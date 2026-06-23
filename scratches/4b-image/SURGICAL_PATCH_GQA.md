# Surgical Patch: GQA-Fix in _chunked_attention (patch.py)

## Diagnose

`px_patches/gemma3_270m_px_baseline/patch.py:_chunked_attention` (Zeile 48-65)
nimmt `q:[B,Hq,T,D]` und `k/v:[B,Hkv,T,D]` an. Das `torch.matmul(qc, k.transpose(-1,-2))`
in Zeile 58 behandelt `Hq`/`Hkv` als Batch-Dims. PyTorch broadcastet Batch-Dims
nur, wenn **eine** davon = 1 ist.

| Modell | Hq | Hkv | Ratio | Verhalten |
|---|---|---|---|---|
| 270m  | 4 | 1 | 4:1 | OK (1-broadcast-Workaround) |
| 1b    | 4 | 1 | 4:1 | OK (1-broadcast-Workaround) |
| 4b    | 8 | 4 | 2:1 | **CRASH** — "tensor a (8) must match tensor b (4)" |

Live-Reproduktion: jeder 4b-Request mit Prefill > 4096 Tokens (oder mit
chunked-Pfad-Trigger via MEM_EFF_THRESHOLD) wirft diesen Fehler. Im
Server-Log (23. Jun, 15:38) mehrfach beobachtet.

## Surgical Patch (2 minimale Edits in patch.py)

### Edit 1 — Helper hinzufügen (nach Zeile 46, vor `def _chunked_attention`)

```python
def _expand_kv_for_gqa(k, v, n_rep):
    """Expand KV-Heads entlang Hq-Ratio (GQA-Standard-Pattern). Idempotent
    für n_rep == 1. Genutzt in _chunked_attention, weil matmul Hq/Hkv als
    Batch-Dims behandelt und nur bei 1 broadcastet. Identisch zu
    F.scaled_dot_product_attention(enable_gqa=True) intern — wir machen es
    explizit, weil unser chunked-Path kein SDPA nutzt.

    Siehe scratches/4b-image/gqa_repeat.py + test_gqa_surgical.py für den
    Pin-Test (4/4 grün, cos_sim >= 0.999999 vs SDPA-Referenz).
    """
    if n_rep == 1:
        return k, v
    return k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
```

### Edit 2 — Aufruf in _chunked_attention (nach `out = torch.empty_like(q)`, vor der `for s`-Schleife)

```python
    out = torch.empty_like(q)
    kpos = torch.arange(Tk, device=device)
    # GQA: expand kv heads to match Hq if needed (4b: Hq=8, Hkv=4, n_rep=2).
    # 270m/1b have Hkv=1 → n_rep=4 → no-op path (still correct, no copy).
    n_rep = q.shape[1] // k.shape[1]
    k, v = _expand_kv_for_gqa(k, v, n_rep)
    for s in range(0, Tq, chunk):
        ...
```

## Was der Patch NICHT ändert

- Kein neuer Allokations-Overhead für 270m/1b (`n_rep=1` → no-op-Return derselben Tensoren).
- Keine Änderung an Aufrufer-Signatur, Rückgabe-Shape, oder dtype-Verhalten.
- Kein Edit am `MEM_EFF_CHUNK`/`MEM_EFF_THRESHOLD`.
- Kein Edit am _mem_eff_attention_forward-Outer.

## Verifikation (TDD-rot → grün)

Im `scratches/4b-image/`-Verzeichnis:

```bash
python test_gqa_surgical.py
```

Ergebnis:
```
[OK] 270m   Hq=4 Hkv=1 T=   64  cos_sim=0.999999
[OK] 1b     Hq=4 Hkv=1 T=   64  cos_sim=0.999999
[OK] 4b     Hq=8 Hkv=4 T=   64  cos_sim=1.000001
[OK] 4b-long Hq=8 Hkv=4 T=2048  cos_sim=1.000333
4/4 passed
```

Vor dem Fix: 2/4 grün (270m/1b), 4b crasht.

## E2E-Reproduktion (Server, post-Fix)

```bash
python scripts/gqa_regression_test.py --model gemma3-4b-it --preset BASELINE
```

Erwartung: HTTP 200, non-empty assistant text, kein `RuntimeError`/`OutOfMemoryError`
im Server-Log nach dem Call.

## Alternative: HF SDPA statt eigener _chunked_attention

Falls der chirurgische Patch zu invasiv ist: einfach im _mem_eff_attention_forward
den `else`-Branch (Zeile 89-95) auf `F.scaled_dot_product_attention(..., enable_gqa=True)`
umstellen. Das wäre 1-Zeilen-Edit, keine Helper-Funktion, aber wir verlieren
die Memory-Tiling-Optimierung (T^2 → T*chunk score matrix). Für 4b auf 12 GB
relevant. Daher chirurgischer Patch vorzuziehen.

## Status

- [x] TDD-rot reproduziert (vorher 2/4 grün)
- [x] TDD-grün demonstriert (4/4 grün in scratches/4b-image/)
- [ ] Motor-Patch applyen (USER-GENEHMIGUNG erforderlich — du sagtest
      "bitte nicht im motor versuche machen")
- [ ] E2E-Smoke (scripts/gqa_regression_test.py) post-apply
- [ ] CI-Liste erweitern (test_gqa_surgical in pytest-Workflow aufnehmen)
