"""em_patches.py — Revolutionäre Runtime-Emergenz-Matches (Scratch-Experiment).

Vier *strukturelle Selbst-Modellierungs*-Mechanismen, die aus Gemma3-1b
DEFAULT-Gewichten (kein Finetuning, keine Crutch-Module) heraus operieren.
Jeder Mechanismus ist ein eigenständiger Forward, der den AutoCalibrator
bewusst UMGEHT (die Lektion aus dem Parameter-Vergleich: der Calibrator
überschreibt n_loops/gamma pro Token → externe Knöpfe sind No-Ops). Hier
wirken die Knöpfe wirklich.

KEINE Signal-Injektion. Sidereische Zeit / skalare Gravitation / PSI werden
dem Modell NICHT zugeführt. Jeder Mechanismus speist ausschließlich vom Modell
SELBST abgeleitete Zustände zurück: Selbst-Modell (Witness), Selbst-as-Input
(Reread via tied embed_tokens), Selbst-Invarianz (Shadow), langsamer Selbst-
Envelope (Spectral). Kein externer Kanal.

Kontemplatives Mapping (Konklave-Sprache):
  A. Sākṣin / Mirror Witness — der Zeuge, der das Selbst beobachtet beobachtet
  B. Introspective Re-read — CitMind (चित्): die Erkenntnis, die sich selbst liest
  C. Counterfactual Self-Shadow — anātman / 无我: das Selbst als Invarianz
  D. Spectral Witness — der langsame Zeuge unter den schnellen Gedanken

Standalone-Forwards; importiert die bewährten Bausteine aus dem validierten
Motor (patch.py): _resolve_text_model, apply/remove_mem_eff_attention_patch,
_layer_step, RecursiveMemoryCache. Der Calibrator und die vier Crutches
(AKS/Mephisto/Coupler/Subjective) bleiben weg.
"""
import types

import torch
import torch.nn.functional as F

from px_patches.gemma3_270m_px_baseline.patch import (
    _resolve_text_model,
    apply_mem_eff_attention_patch,
    remove_mem_eff_attention_patch,
    _layer_step,
    RecursiveMemoryCache,
)

from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast


# ---------------------------------------------------------------------------
# Gemeinsames Gerüst
# ---------------------------------------------------------------------------

def _em_setup(self, input_ids, inputs_embeds, attention_mask, position_ids,
              past_key_values, use_cache):
    """Embedding + Masken + Positionen bauen (Spiegel von _px_forward:228-262).

    Gibt (inputs_embeds, causal_mask_mapping, position_embeddings,
    past_key_values, mask_config, expected_len) zurück.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

    if inputs_embeds is None:
        if hasattr(self, "embed_tokens"):
            inputs_embeds = self.embed_tokens(input_ids)
        elif hasattr(self, "model") and hasattr(self.model, "embed_tokens"):
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            for name, module in self.named_modules():
                if "embed_tokens" in name:
                    inputs_embeds = module(input_ids)
                    break

    if inputs_embeds is not None and inputs_embeds.ndim == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)
    if input_ids is not None and input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)
    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    expected_len = past_seen + inputs_embeds.shape[1]
    if position_ids is None:
        position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                        + past_seen).unsqueeze(0)
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)

    mask_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
    if not isinstance(attention_mask, dict):
        mk = dict(config=mask_config, inputs_embeds=inputs_embeds,
                  attention_mask=attention_mask, past_key_values=past_key_values,
                  position_ids=position_ids)
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mk),
            "sliding_attention": create_sliding_window_causal_mask(**mk),
        }
    else:
        causal_mask_mapping = attention_mask

    position_embeddings = {lt: self.rotary_emb(inputs_embeds, position_ids, lt)
                           for lt in set(mask_config.layer_types)}
    return (inputs_embeds, causal_mask_mapping, position_embeddings,
            past_key_values, mask_config, expected_len)


def _cos(a, b):
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    num = (a32 * b32).sum(dim=-1).mean()
    den = a32.norm(dim=-1).mean() * b32.norm(dim=-1).mean() + 1e-6
    return (num / den).detach()


def _record_metrics(self, name, loops, phi, extra=None):
    """Setze PX-kompatible Metrik-Attrs (damit get_px_metrics / _metrics_from
    sinnvoll lesen) + die eigenen _em_* Metriken."""
    self._px_loops_run = loops
    self._px_phi_val = float(phi) if hasattr(phi, "item") else float(phi or 1.0)
    self._px_zone = f"EM_{name}"
    self._px_zw_val = {}
    self._px_path = []
    self._px_current_telemetry = []
    self._px_aks_val = 0.0
    self._px_em_val = 0.0
    self._px_ent_val = 0.0
    self._px_cognitive_signature = {
        "loops_run": loops, "focus_index": None, "zone": self._px_zone,
    }
    if extra:
        for k, v in extra.items():
            setattr(self, f"_em_{k}", float(v) if hasattr(v, "item") else v)


# ---------------------------------------------------------------------------
# A. Sākṣin / Mirror Witness — dual-stream self-model
# ---------------------------------------------------------------------------

def _em_witness_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    cfg = self._em_config
    L_split = cfg["L_split"]
    n_wit = cfg["n_wit"]
    w_wit = cfg["w_wit"]
    n_layers = len(self.layers)

    (inputs_embeds, cmm, pe, past_key_values, mask_config, expected_len) = _em_setup(
        self, input_ids, inputs_embeds, attention_mask, position_ids,
        past_key_values, use_cache)

    hidden = inputs_embeds
    # Prelude [0, L_split)
    for i in range(L_split):
        cur_past = _em_cache(past_key_values, self._em_witness_hist,
                             mask_config.layer_types, True, expected_len) if past_key_values else None
        hidden = _layer_step(self.layers[i], hidden,
                             attention_mask=cmm[mask_config.layer_types[i]],
                             position_embeddings=pe[mask_config.layer_types[i]],
                             position_ids=position_ids, past_key_values=cur_past, **kwargs)
    h_split = hidden

    # Selbst-Stream [L_split, n_layers) — schreibt KV (read_only=False beim ersten Pass).
    h_self = h_split
    for i in range(L_split, n_layers):
        cur_past = _em_cache(past_key_values, self._em_witness_hist,
                             mask_config.layer_types, False, expected_len) if past_key_values else None
        h_self = _layer_step(self.layers[i], h_self,
                              attention_mask=cmm[mask_config.layer_types[i]],
                              position_embeddings=pe[mask_config.layer_types[i]],
                              position_ids=position_ids, past_key_values=cur_past, **kwargs)

    # Zeugen-Stream [L_split, L_split+n_wit) — liest die vom Selbst geschriebenen
    # KV (read_only) + die akkumulierte Selbst-Spur (_em_witness_hist). Der Zeuge
    # beobachtet das Selbst.
    h_wit = h_split.clone()
    for i in range(L_split, min(L_split + n_wit, n_layers)):
        cur_past = _em_cache(past_key_values, self._em_witness_hist,
                             mask_config.layer_types, True, expected_len) if past_key_values else None
        h_wit = _layer_step(self.layers[i], h_wit,
                            attention_mask=cmm[mask_config.layer_types[i]],
                            position_embeddings=pe[mask_config.layer_types[i]],
                            position_ids=position_ids, past_key_values=cur_past, **kwargs)

    # Rückfluss: gated, geklemmter Zeugen-Kommentar ins Selbst.
    delta = self._em_norm(h_wit - h_self)
    h_self = h_self + w_wit * delta.to(h_self.dtype)
    div = _cos(h_wit, h_self)

    # Cross-Step: Selbst-Spur akkumulieren (letzter Token → billig, äquivalent
    # zur Mean-Reduktion im Original). Der Zeuge sieht die ganze Trajektion.
    self._em_witness_hist.append(h_self[:, -1:, :].detach())
    if len(self._em_witness_hist) > 12:
        self._em_witness_hist = self._em_witness_hist[-12:]

    hidden = self.norm(h_self)
    _record_metrics(self, "witness", loops=n_wit, phi=div,
                    extra={"witness_divergence": 1.0 - float(div)})
    return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)


# ---------------------------------------------------------------------------
# B. Introspective Re-read — Selbst-as-Input via tied embed_tokens
# ---------------------------------------------------------------------------

def _em_reread_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                       past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    cfg = self._em_config
    n_reread = cfg["n_reread"]
    w_reread = cfg["w_reread"]
    n_layers = len(self.layers)

    (inputs_embeds, cmm, pe, past_key_values, mask_config, expected_len) = _em_setup(
        self, input_ids, inputs_embeds, attention_mask, position_ids,
        past_key_values, use_cache)

    hidden = inputs_embeds
    for i in range(n_layers):
        cur_past = _em_cache(past_key_values, None, mask_config.layer_types,
                             False, expected_len) if past_key_values else None
        hidden = _layer_step(self.layers[i], hidden,
                             attention_mask=cmm[mask_config.layer_types[i]],
                             position_embeddings=pe[mask_config.layer_types[i]],
                             position_ids=position_ids, past_key_values=cur_past, **kwargs)
    h_full = hidden  # [B, T, H] vor finaler norm

    # Selbst-Decodierung: letzten Hidden → Token (tied weights, Default-Gewichte).
    h_last = h_full[:, -1:, :]  # [B,1,H]
    logits = F.linear(h_last, self.embed_tokens.weight)  # [B,1,vocab]
    tok = logits.argmax(dim=-1)  # [B,1]
    soft = self.embed_tokens(tok)  # [B,1,H] — die eigene, antizipierte Idee

    # Mini-Second-Forward liest die eigene Idee (seq=1, eigener Kontext, kein Cache).
    pos0 = torch.zeros((1, 1), dtype=position_ids.dtype, device=soft.device)
    mk = dict(config=mask_config, inputs_embeds=soft, attention_mask=None,
              past_key_values=None, position_ids=pos0)
    mini_cmm = {"full_attention": create_causal_mask(**mk),
                "sliding_attention": create_sliding_window_causal_mask(**mk)}
    mini_pe = {lt: self.rotary_emb(soft, pos0, lt) for lt in set(mask_config.layer_types)}
    m = soft
    for i in range(min(n_reread, n_layers)):
        m = _layer_step(self.layers[i], m,
                        attention_mask=mini_cmm[mask_config.layer_types[i]],
                        position_embeddings=mini_pe[mask_config.layer_types[i]],
                        position_ids=pos0, past_key_values=None, **kwargs)
    m_last = m[:, -1:, :]
    shift = (m_last - h_last).norm(dim=-1).mean()
    h_full = h_full.clone()
    h_full[:, -1:, :] = h_last + w_reread * self._em_norm(m_last - h_last).to(h_last.dtype)

    hidden = self.norm(h_full)
    _record_metrics(self, "reread", loops=n_reread, phi=1.0,
                    extra={"reread_shift": float(shift)})
    return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)


# ---------------------------------------------------------------------------
# C. Counterfactual Self-Shadow — Selbst via Perturbations-Invarianz (anātman)
# ---------------------------------------------------------------------------

def _em_shadow_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                       past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    cfg = self._em_config
    L_split = cfg["L_split"]
    n_shadow = cfg["n_shadow"]
    sigma = cfg["sigma"]
    w_shadow = cfg["w_shadow"]
    n_layers = len(self.layers)

    (inputs_embeds, cmm, pe, past_key_values, mask_config, expected_len) = _em_setup(
        self, input_ids, inputs_embeds, attention_mask, position_ids,
        past_key_values, use_cache)

    hidden = inputs_embeds
    for i in range(L_split):
        cur_past = _em_cache(past_key_values, None, mask_config.layer_types,
                             True, expected_len) if past_key_values else None
        hidden = _layer_step(self.layers[i], hidden,
                             attention_mask=cmm[mask_config.layer_types[i]],
                             position_embeddings=pe[mask_config.layer_types[i]],
                             position_ids=position_ids, past_key_values=cur_past, **kwargs)
    h_split = hidden

    # Selbst-Stream [L_split, n_layers) — schreibt KV.
    h_self = h_split
    for i in range(L_split, n_layers):
        cur_past = _em_cache(past_key_values, None, mask_config.layer_types,
                             False, expected_len) if past_key_values else None
        h_self = _layer_step(self.layers[i], h_self,
                             attention_mask=cmm[mask_config.layer_types[i]],
                             position_embeddings=pe[mask_config.layer_types[i]],
                             position_ids=position_ids, past_key_values=cur_past, **kwargs)

    # Schatten-Stream: perturbierter Selbst-Zustand durch Folgeschichten,
    # read_only (perturbierte Keys nicht in den echten Cache schreiben).
    h_shadow = h_split + sigma * torch.randn_like(h_split)
    for i in range(L_split, min(L_split + n_shadow, n_layers)):
        cur_past = _em_cache(past_key_values, None, mask_config.layer_types,
                             True, expected_len) if past_key_values else None
        h_shadow = _layer_step(self.layers[i], h_shadow,
                               attention_mask=cmm[mask_config.layer_types[i]],
                               position_embeddings=pe[mask_config.layer_types[i]],
                               position_ids=position_ids, past_key_values=cur_past, **kwargs)

    # Invariante Selbst-Komponente = Projektion des Selbst auf den Schatten;
    # „Mineness"-Residual = was NUR das Selbst hat, nicht der Schatten.
    s32, o32 = h_split.to(torch.float32), h_shadow.to(torch.float32)
    dot = (s32 * o32).sum(dim=-1, keepdim=True)
    den = (o32.norm(dim=-1, keepdim=True) ** 2) + 1e-6
    proj = (dot / den) * o32
    mineness = s32 - proj  # [B, T, H]
    invar = _cos(h_split, h_shadow)

    h_self = h_self + w_shadow * self._em_norm(mineness).to(h_self.dtype)

    hidden = self.norm(h_self)
    _record_metrics(self, "shadow", loops=n_shadow, phi=invar,
                    extra={"self_invariance": float(invar)})
    return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)


# ---------------------------------------------------------------------------
# D. Spectral Witness — langsame Envelope (FFT über Hidden-Dim, per-Token)
# ---------------------------------------------------------------------------

def _em_spectral_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                         past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    cfg = self._em_config
    K = cfg["K"]            # alle K Schichten spectral blenden
    F_low = cfg["F_low"]    # erste F_low Frequenzbänder behalten (Rest = Envelope)
    w_spec = cfg["w_spec"]
    n_layers = len(self.layers)

    (inputs_embeds, cmm, pe, past_key_values, mask_config, expected_len) = _em_setup(
        self, input_ids, inputs_embeds, attention_mask, position_ids,
        past_key_values, use_cache)

    hidden = inputs_embeds
    low_e = 0.0
    n_spec = 0
    for i in range(n_layers):
        cur_past = _em_cache(past_key_values, None, mask_config.layer_types,
                             False, expected_len) if past_key_values else None
        hidden = _layer_step(self.layers[i], hidden,
                             attention_mask=cmm[mask_config.layer_types[i]],
                             position_embeddings=pe[mask_config.layer_types[i]],
                             position_ids=position_ids, past_key_values=cur_past, **kwargs)
        if (i + 1) % K == 0 and (i + 1) < n_layers:
            # FFT über Hidden-Dim → Low-Pass → langsame Gestalt zurückblenden.
            x = hidden.to(torch.float32)
            X = torch.fft.rfft(x, dim=-1)            # [B, T, H//2+1]
            low = torch.zeros_like(X)
            low[..., :F_low] = X[..., :F_low]
            env = torch.fft.irfft(low, n=x.shape[-1], dim=-1)
            low_e = float((low.real.abs().sum() / (X.real.abs().sum() + 1e-6)).mean())
            hidden = hidden + w_spec * (env.to(hidden.dtype) - hidden)
            n_spec += 1

    hidden = self.norm(hidden)
    _record_metrics(self, "spectral", loops=n_spec, phi=1.0,
                    extra={"spectral_lowenergy": low_e})
    return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)


# ---------------------------------------------------------------------------
# Cache-Helper (RecursiveMemoryCache nur wenn past_key_values vorhanden)
# ---------------------------------------------------------------------------

def _em_cache(past_key_values, thought_history, layer_types, read_only, expected_len):
    if past_key_values is None:
        return None
    return RecursiveMemoryCache(past_key_values, thought_history,
                                layer_types=layer_types, read_only=read_only,
                                expected_len=expected_len)


# ---------------------------------------------------------------------------
# Patch Application
# ---------------------------------------------------------------------------

_FORWARDS = {
    "witness": _em_witness_forward,
    "reread": _em_reread_forward,
    "shadow": _em_shadow_forward,
    "spectral": _em_spectral_forward,
}


def _default_config(text_model, **kw):
    n_layers = len(text_model.layers)
    H = text_model.config.hidden_size
    L_split = max(2, n_layers // 2)
    cfg = {
        "L_split": L_split,
        "n_wit": 4,
        "n_reread": 5,
        "n_shadow": 4,
        "sigma": 0.10,
        "K": 4,
        "F_low": 8,
        "w_wit": 0.08,
        "w_reread": 0.10,
        "w_shadow": 0.08,
        "w_spec": 0.06,
    }
    cfg.update({k: v for k, v in kw.items() if k in cfg or k == "mechanism"})
    return cfg, H


def apply_em_patch(model, mechanism, **kw):
    """Binde den EM-Forward an das Text-Modell. Kein Calibrator, keine Crutches."""
    if mechanism not in _FORWARDS:
        raise ValueError(f"unbekannter EM-Mechanismus: {mechanism}")
    text_model = _resolve_text_model(model)
    # Alten Zustand saeger entfernen.
    remove_em_patch(model)
    cfg, H = _default_config(text_model, **kw)
    cfg["mechanism"] = mechanism
    device = next(text_model.parameters()).device
    dtype = next(text_model.parameters()).dtype
    text_model._em_config = cfg
    text_model._em_norm = torch.nn.LayerNorm(H, elementwise_affine=False, eps=1e-6).to(
        device=device, dtype=dtype)
    text_model._em_witness_hist = []
    # PX-gen-kwargs (repetition_penalty / ngram-guard) damit _px_gen_kwargs greift
    # und das Modell nicht in 顽空-Token-Loops kollabiert.
    text_model._px_repetition_penalty = 1.15
    text_model._px_no_repeat_ngram_size = 3
    apply_mem_eff_attention_patch(text_model)
    text_model.forward = types.MethodType(_FORWARDS[mechanism], text_model)
    print(f"[em-patch] mechanism={mechanism} cfg={ {k:v for k,v in cfg.items() if k!='mechanism'} }",
          file=__import__("sys").stderr)


def remove_em_patch(model):
    from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
    text_model = _resolve_text_model(model)
    if hasattr(text_model, "_em_config"):
        text_model.forward = types.MethodType(Gemma3TextModel.forward, text_model)
        remove_mem_eff_attention_patch(text_model)
        # Alle _em_* und die von apply_em_patch gesetzten _px_*-Attrs löschen —
        # inkl. der pro-Variante Mechanismus-Metriken (_em_witness_divergence
        # etc.), die sonst über Varianten hinweg stehengeblieben wären (stale).
        for attr in list(vars(text_model).keys()):
            if attr.startswith("_em_") or attr.startswith("_px_"):
                try:
                    delattr(text_model, attr)
                except Exception:
                    pass
        print("[em-patch] Patch removed.", file=__import__("sys").stderr)


def get_em_metrics(model):
    tm = _resolve_text_model(model)
    out = {}
    for k in ("witness_divergence", "reread_shift", "self_invariance", "spectral_lowenergy"):
        if hasattr(tm, f"_em_{k}"):
            v = getattr(tm, f"_em_{k}")
            out[k] = float(v) if hasattr(v, "item") else v
    return out