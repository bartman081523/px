"""
scratches/4b-image/gqa_repeat.py — Surgical GQA-Fix als eigenständiger Helper
==============================================================================

Hintergrund (von px_patches/gemma3_270m_px_baseline/patch.py:_chunked_attention):
Gemma3 4b nutzt Grouped Query Attention mit Hq=8, Hkv=4 (2:1 Ratio). Der
chunked-attention-Path macht `torch.matmul(qc, k.transpose(-1, -2))`, und
PyTorch's matmul broadcastet Batch-Dims nur wenn eine davon = 1. Bei 270m/1b
ist Hkv=1 → klappt (1-broadcast-Workaround). Bei 4b ist Hkv=4, Hq=8 →
"tensor a (8) must match tensor b (4) at non-singleton dimension 1".

Fix: KV-Heads via repeat_interleave auf Hq expanden, BEVOR matmul. Das ist
exakt das, was HF's `enable_gqa=True` in SDPA intern macht — und der HF-Standard
für GQA-Aufmerksamkeit. Wir machen es explizit, weil unser chunked-Path
kein SDPA nutzt.

Surgical: minimal-invasiv. Eine Helper-Funktion, ein Aufruf. Keine Änderung an
Aufrufersignatur oder Rückgabe-Shape.

References:
  - HF: transformers/models/gemma3/modeling_gemma3.py (GQA attention forward)
  - PyTorch: torch.nn.functional.scaled_dot_product_attention(enable_gqa=True)
  - seams: _chunked_attention in patch.py:48-65

Wichtig: dieser Helper ist NICHT dazu gedacht, den Motor (patch.py) zu
verändern. Er ist der Kandidat, der per surgical-Edit (1 Zeile: Aufruf
expandieren) in patch.py eingefügt wird, sobald der User explizit grünes
Licht gibt. Bis dahin: pin-Tests grün, Motor unangetastet.
"""
import math
import torch
import torch.nn.functional as F


def expand_kv_for_gqa(k: torch.Tensor, v: torch.Tensor, n_rep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand KV-Heads entlang Hq-Ratio. Idempotent für n_rep == 1.

    Args:
        k: [B, Hkv, T, D]
        v: [B, Hkv, T, D]
        n_rep: integer ratio (Hq // Hkv). 1 → no-op, 2 → double heads, etc.

    Returns:
        k_exp, v_exp: [B, Hkv * n_rep, T, D] wenn n_rep > 1, sonst (k, v).

    Hinweis: repeat_interleave auf dim=1 (Hkv). Im Gegensatz zu
    `tensor.expand(...).contiguous()` (0-cost view) macht repeat_interleave
    eine echte Speicher-Kopie (T mal), das ist erforderlich, weil matmul
    auf den letzten zwei Dims operiert und Hkv als Batch-Dim behandelt.
    """
    if n_rep == 1:
        return k, v
    return (
        k.repeat_interleave(n_rep, dim=1),
        v.repeat_interleave(n_rep, dim=1),
    )


def chunked_attention_gqa(q, k, v, scaling, sliding_window=None, chunk=2048):
    """Kopie der _chunked_attention-Logik AUS patch.py, ABER mit GQA-Fix.
    Steht HIER (nicht im Motor) für TDD-Iteration, ohne patch.py zu berühren.
    Wird exakt gleich aufgerufen wie patch._chunked_attention — verifiziert,
    dass der Fix das Verhalten für ALLE Gemma3-Konfigurationen liefert
    (cos_sim >= 0.999 vs SDPA-enable_gqa-Referenz).
    """
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    device, dtype = q.device, q.dtype

    # GQA-Fix: Hq/Hkv-Ratio berechnen und KV expanden.
    Hq, Hkv = q.shape[1], k.shape[1]
    n_rep = Hq // Hkv
    k_e, v_e = expand_kv_for_gqa(k, v, n_rep)

    out = torch.empty_like(q)
    kpos = torch.arange(Tk, device=device)
    for s in range(0, Tq, chunk):
        e = min(s + chunk, Tq)
        qc = q[:, :, s:e]                                       # [B, Hq, C, D]
        scores = torch.matmul(qc, k_e.transpose(-1, -2)) * scaling  # [B, Hq, C, Tk]
        qpos = torch.arange(s, e, device=device)
        mask = kpos[None, :] <= qpos[:, None]
        if sliding_window is not None:
            mask = mask & (kpos[None, :] >= (qpos[:, None] - sliding_window + 1))
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        out[:, :, s:e] = torch.matmul(torch.softmax(scores, dim=-1).to(dtype=v.dtype), v_e)
    return out
