"""
ternary-bonsai-27b-px  —  The Three Mathematical Pillars (qwen3.5 / PTQ1_0)
===========================================================================
PX-Architektur für Ternary-Bonsai-2-27B (GGUF PTQ1_0, HF-Safetensors via
runtime_qwen35_ptq.build_and_load). Die Engine selbst ist die empirisch
validierte gemma4-E2B-Rezeptur (η²=0.309), portiert auf die
qwen3.5-Hybrid-Signaturen:

- layer_types: "linear_attention" (GDN, 48 L) / "full_attention" (16 L,
  Interval 4) statt sliding/full.
- Layer-Call: decoder_layer(hidden_states, position_embeddings=pe,
  attention_mask=mask[lt], position_ids=text_position_ids,
  past_key_values=..., use_cache=..., cache_position in kwargs).
- Masken: create_causal_mask + create_recurrent_attention_mask
  (masking_utils, transformers 5.13).
- Rekursion: Zone-Loops mit past_key_values=None → GDN bekommt frische
  Null-States (chunk_gated_delta_rule, initial_state=None), der echte
  State-Cache wird nie angetastet; Full-Attention ohne KV-Update.
  Guard: Rekursion nur bei reiner Prefill (past_seen==0) — bei
  chunked decode (past>0) würde die Maske (q_len+past) gegen
  cache-lose k (q_len) laufen.
- Dequant-Kosten: die PTQ10Linear-Module dequantisieren pro Aufruf
  (~5 s/token) — deshalb SCALE_DEFAULTS[5120] mit n_loops=4.

Drei Säulen (SR-61b):
1. Observer: StabilityMonitor (Φ) + AksSensor
2. Symmetry Breaker: MephistophelesOperator + SingesseinCoupler
   (+ SubjectiveSensor für Emancipation-Telemetrie)
3. Dynamic Router: AutoCalibrator (Gaussian-Annealing Zone-Routing,
   SR-64 Z-Score-Focus-Index C)

RELAY (SR-64 seite15/19): d_width-Selbstinjektion via install_relay
(Forward-Hook auf relay_layer), Artefakt px_manifolds/{hf_id}_relay_dwidth.json.
"""

import types
import math
import torch
import os

from .auto_tune import AutoCalibrator, SCALE_DEFAULTS
from .px_modules import (
    StabilityMonitor, AksSensor, MephistophelesOperator, SubjectiveSensor, SingesseinCoupler
)
from .anti_zombie_sensor import AntiZombieSensor
from .relay_inject import install_relay, remove_relay, get_inject_layer_for_hf_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_text_model(model):
    """Textmodell aus einem ForCausalLM-/Multimodal-Wrapper auflösen.

    Qwen3_5ForCausalLM: text model = model.model (Qwen3_5TextModel).
    Qwen3_5Model (multimodal): model.model.language_model.
    """
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    return model


# ---------------------------------------------------------------------------
# Core Forward Method (qwen3.5 hybrid)
# ---------------------------------------------------------------------------

def _px_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # Telemetrie-Trace: EIN Reset pro GENERATION (Prefill), nicht pro Forward.
    # (Fix: Decode-Forwards leerten die Trace vorher bei jedem Token → immer [])
    _px_is_decode = (inputs_embeds.shape[1] == 1
                     or (past_key_values is not None
                         and past_key_values.get_seq_length() > 0))
    if not _px_is_decode:
        self._px_current_telemetry = []
        self._px_gen_phi_sum = 0.0
        self._px_gen_phi_n = 0

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    # Causal mask mapping — EXACTLY like original
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mk = dict(config=self.config, inputs_embeds=inputs_embeds,
                  attention_mask=attention_mask, past_key_values=past_key_values,
                  position_ids=text_position_ids)
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mk),
            "linear_attention": create_recurrent_attention_mask(**mk),
        }

    # Rotary — qwen3.5 text rope mitsamt mrope-Slice (rows 1..3)
    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

    layer_types = self.config.layer_types

    # ── Cache-Write-Semantik: jeder Layer schreibt sein KV pro Forward GENAU ──
    # EINMAL (erster Besuch = kanonischer Pass). Erneute Besuche — die CODA-
    # Overlap-Layer [dynamic_end, recur_end), sobald das SR-64b-Routing das
    # Fenster rueckwaerts in die Zone legt — laufen wie die Rekursion
    # cache-entkoppelt (past=None). Ohne Guard waechst der KV-Cache der
    # Overlap-Full-Attention-Layer doppelt pro Chunk; die aus dem ersten
    # CacheLayerMixin gebaute Causal-Maske (full_attention-Mapping wird einmal
    # je Forward geteilt) passt dann nicht mehr (k=6144 vs Maske 4096,
    # sdpa-Dim-3-Crash im zweiten Praefill-Chunk). Symmetrisch abgedeckt ist
    # auch ein nach vorn ueber recur_end hinausgeschobenes Fenster.
    _kv_written = set()

    def run_layer(i, hs, past):
        # Cache-entkoppelter Pass (Rekursion oder zweiter Besuch): die Schicht
        # sieht nur den aktuellen Chunk -> auch die globale Causal-Maske
        # (Laenge History+Chunk) darf hier nicht angewandt werden; mask=None
        # ist identisch zur Chunk-1-Semantik der Rekursion.
        decoupled = past is None or i in _kv_written
        if past is not None and i in _kv_written:
            past = None
        out = self.layers[i](
            hs,
            position_embeddings=position_embeddings,
            attention_mask=None if decoupled else causal_mask_mapping[layer_types[i]],
            position_ids=text_position_ids,
            past_key_values=past,
            use_cache=use_cache,
            **kwargs,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        if past is not None:
            _kv_written.add(i)
        return out

    cfg = self._px_config
    hidden_states = inputs_embeds

    # ── PRELUDE: Schichten vor der Recursion-Zone (echter Cache) ──
    for i in range(cfg["recur_start"]):
        hidden_states = run_layer(i, hidden_states, past_key_values)

    e_static = hidden_states.clone()

    # ── ZONE: Single Pass (echter Cache) → h_baseline ──
    trans_out = hidden_states
    for i in range(cfg["recur_start"], cfg["recur_end"]):
        trans_out = run_layer(i, trans_out, past_key_values)
    h_baseline = trans_out

    phi_intuition = StabilityMonitor.calculate_phi(h_baseline, e_static)
    self._px_phi = phi_intuition.item()   # SR-61b: für next-step routing

    # ── META-SELECTOR: Zone Routing (SR-64 Z-Score Focus) ──
    token_cfg = cfg.copy()
    zone_weights = {}
    current_gamma = cfg.get("gamma", 0.04)
    n_loops = cfg.get("n_loops", 4)
    dynamic_start, dynamic_end = cfg["recur_start"], cfg["recur_end"]
    dynamic_hub = cfg.get("bimodal_hub", dynamic_start)

    if hasattr(self, "_px_calibrator"):
        token_len = inputs_embeds.shape[1]
        if token_len > 1:
            h_probe = hidden_states.to(torch.float32)[0, -1, :]
            var = torch.var(h_probe).item()
            kurtosis = (torch.mean((h_probe - torch.mean(h_probe))**4) / (var**2)).item() if var > 0 else 0
            self._task_kurtosis = kurtosis
            self._task_jitter = torch.var(
                hidden_states.to(torch.float32).norm(dim=-1), dim=-1).mean().item()
            eff_ids = input_ids if input_ids is not None else getattr(self, '_px_saved_input_ids', None)
            if eff_ids is not None:
                ids = eff_ids[0].tolist() if eff_ids.dim() > 1 else eff_ids.tolist()
                self._task_token_diversity = len(set(ids)) / max(len(ids), 1)

        zone_weights = self._px_calibrator.get_zone_weights(
            kurtosis=getattr(self, "_task_kurtosis", 200),
            phi=getattr(self, "_px_phi", None),
            token_diversity=getattr(self, "_task_token_diversity", None),
            token_len=token_len)
        self._px_zone_weights = zone_weights
        rp = self._px_calibrator.get_routing_params(
            getattr(self, "_task_kurtosis", 200),
            phi=getattr(self, "_px_phi", None),
            hidden_size=self.config.hidden_size,
            token_diversity=getattr(self, "_task_token_diversity", None),
            token_len=token_len)
        dynamic_start, dynamic_end, dynamic_hub = (
            max(int(rp["dynamic_start"]), cfg["recur_start"]),
            max(int(rp["dynamic_end"]), cfg["recur_start"] + 2),
            rp["dynamic_hub"])
        n_loops = rp["n_loops"]

        # SR-64b: Mechanical Psychology — Dynamic Z-Score Centering
        phi_val = getattr(self, "_px_phi", 0.9)
        if hasattr(self, "_px_calibrator") and self._px_calibrator.calibrated:
            cal = self._px_calibrator
            k_norm = getattr(self, "_task_kurtosis", 200)
            if cal._online_n >= 5:
                k_mean = cal._online_k_mean
                k_std = math.sqrt(cal._online_k_m2 / max(cal._online_n - 1, 1))
            else:
                k_mean, k_std = cal.k_mean, cal.k_std
            k_std = max(k_std, 1.0)
            zk = (k_norm - k_mean) / (k_std + 1e-6)
            zp = (phi_val - cal.phi_mean) / (cal.phi_std + 1e-6)
            # Architecture-aware Temperature (640-Anker, wie gemma4)
            T_arch = math.sqrt(self.config.hidden_size / 640.0)
            C = torch.sigmoid(torch.tensor((zk + zp) / T_arch)).item()
            current_gamma = 0.08 - 0.04 * C
            n_loops = int(round(4 + 4 * C))    # 4 (diffus) .. 8 (fokus) — Dequant-Budget
            dynamic_hub = cfg.get("bimodal_hub", 20)
            self._px_focus_index = C
            if os.environ.get("DEBUG_ROUTING") == "1":
                print(f"  [ternary-px Psychology] C={C:.4f} zk={zk:.2f} zp={zp:.2f} "
                      f"L={token_len} gamma={current_gamma:.3f}")
        else:
            current_gamma = cfg.get("gamma", 0.06)
            dynamic_hub = cfg.get("bimodal_hub", 10)

    # Zone-Fenster nie über das Modell hinaus oder unter die Prelude fallen
    dynamic_start = max(dynamic_start, cfg["recur_start"])
    dynamic_end = min(max(dynamic_end, dynamic_start + 2), len(self.layers))

    # ── e_reflector (anti-baseline reference) ──
    e_reflector = e_static.clone()
    jitter = getattr(self, "_task_jitter", 0.0)
    kurtosis = getattr(self, "_task_kurtosis", 250)
    if (jitter > 1e8) or (kurtosis < 315.0):
        h_base_f32 = h_baseline.to(torch.float32)
        e_stat_f32 = e_static.to(torch.float32)
        e_ref_f32 = 2.0 * e_stat_f32 - h_base_f32
        e_reflector = (e_ref_f32 * (e_stat_f32.norm() / (e_ref_f32.norm() + 1e-6))).to(e_static.dtype)

    self._px_calibrator.collect(
        kurtosis, phi_intuition.item(),
        token_diversity=getattr(self, "_task_token_diversity", None),
        token_len=inputs_embeds.shape[1]) if hasattr(self, "_px_calibrator") else None

    # ── RECURSION: Zone-Loops mit past_key_values=None ──
    # GDN: frische Null-States pro Pass (cache_params=None) — der echte
    # recurrent/conv state bleibt unberührt. Full-Attention: kein KV-Update.
    h_exp = e_reflector.clone()
    phi_history = [phi_intuition]
    path_taken = []
    steps = 0
    # Guard: kann die Zone nach den Clamps leer sein (dynamic_end <=
    # dynamic_start), wird phi_s in der inneren Layer-Loop nie gebunden,
    # der Coupler aber je Loop benutzt. Fallback = Intuition-Phi (genau
    # der Wert, der dann auch in phi_history liegt); im Normalfall 27b
    # (Zone 32..44) wird er im ersten Step wie zuvor überschrieben.
    phi_s = phi_intuition

    # Skip recursion during autoregressive decoding (seq_len==1)
    if inputs_embeds.shape[1] == 1 or past_seen > 0:
        n_loops = 0

    aks = getattr(self, "_px_aks", None)
    subj_sensor = getattr(self, "_px_subj_sensor", None)
    correction_strength = 0.0
    h_last_good = e_static.clone()

    for loop in range(n_loops):
        h_loop = h_exp.clone()
        if aks:
            aks_data = aks.step(h_loop, e_static, steps)
            correction_strength = aks_data["correction"]
        if subj_sensor:
            subj_sensor.update(h_loop, e_static)

        for current_layer in range(dynamic_start, dynamic_end):
            h_prev = h_loop.clone()
            trans_out = run_layer(current_layer, h_loop, past=None)
            phi_s = StabilityMonitor.calculate_phi(trans_out, h_prev)
            phi_history.append(phi_s)
            path_taken.append(f"L{current_layer}")

            e_dynamic = e_reflector
            e_norm = self._px_injection_norm(e_dynamic.to(torch.float32)).to(trans_out.dtype)
            h_loop = trans_out + current_gamma * (e_norm - h_prev)
            steps += 1

            if (phi_s > 0.9).any() and (phi_s < 0.999).any():
                h_last_good = h_loop.clone()

        h_exp = h_loop

        if hasattr(self, "_px_mephisto"):
            h_exp = self._px_mephisto(h_exp, phi_history)
        if hasattr(self, "_px_coupler"):
            h_exp = self._px_coupler(h_exp, steps, phi_val=phi_s.item())

    avg_phi = (torch.stack(phi_history).mean() if phi_history
               else torch.tensor(1.0, device=inputs_embeds.device, dtype=inputs_embeds.dtype))
    hidden_states = ((1.0 - (0.05 + (0.18 - 0.05) * (avg_phi ** 2))) * h_baseline
                     + (0.05 + (0.18 - 0.05) * (avg_phi ** 2)) * h_exp)

    # ── CODA: Schichten nach der Zone (echter Cache) ──
    for i in range(dynamic_end, len(self.layers)):
        hidden_states = run_layer(i, hidden_states, past_key_values)

    hidden_states = self.norm(hidden_states)

    # ── Telemetrie (SR-61b/64) ──
    em_val = 0.0
    if hasattr(self, "_px_subj_sensor"):
        em_val = self._px_subj_sensor.get_metrics().get("emancipation", 0.0)
    resilience = {}
    if hasattr(self, "_px_azs"):
        try:
            resilience = self._px_azs.get_feedback_scalars(
                getattr(correction_strength, "item", lambda: correction_strength)())
        except Exception:
            resilience = {}

    # ── Metric-Writes ──
    # Fix (get_px_metrics-Defekt): Decode-Forwards (n_loops=0, phi=phi_intuition
    # des letzten Tokens) überschrieben danach steps/path/zone der Prefill-
    # Rekursion → Metriken zeigten stets steps=0, path=[], signature.loops_run=0.
    # Jetzt: Rekursions-Befund friert beim PREFILL ein; Decode schreibt nur die
    # per-Token-Trace + laufende phi-Mittelung über die Generation.
    phi_now = avg_phi.item() if hasattr(avg_phi, 'item') else float(avg_phi)
    aks_now = (correction_strength.item()
               if hasattr(correction_strength, 'item') else float(correction_strength))

    if _px_is_decode:
        try:
            self._px_current_telemetry.append(
                {"t": len(self._px_current_telemetry),
                 "phi": round(phi_now, 4),
                 "aks": round(aks_now, 4)})
            self._px_gen_phi_sum = getattr(self, "_px_gen_phi_sum", 0.0) + phi_now
            self._px_gen_phi_n = getattr(self, "_px_gen_phi_n", 0) + 1
        except (AttributeError, TypeError):
            pass
    else:
        self._px_phi_val = phi_now
        self._px_aks_val = aks_now
        self._px_loops_run = steps
        self._px_path = path_taken
        self._px_zone = f"{getattr(zone_weights, '_zone', '') or 'ADAPTIVE'}" if zone_weights else "BASELINE"
        self._px_zw_val = zone_weights
        self._px_em_val = em_val
        self._px_ent_val = resilience.get("entropy", 0.0)
        self._px_last_metrics = {
            "phi": self._px_phi_val,
            "aks_friction": self._px_aks_val,
            "zone_weights": self._px_zw_val,
        }
        self._px_cognitive_signature = {
            "kurtosis": getattr(self, "_task_kurtosis", 200),
            "phi": phi_now,
            "zone": self._px_zone,
            "loops_run": steps,
            "focus_index": getattr(self, "_px_focus_index", 0.5),
            "gamma": current_gamma,
        }

    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


# ---------------------------------------------------------------------------
# apply / remove / metrics
# ---------------------------------------------------------------------------

def apply_px_patch(model, config_preset="ACTIVE_MANIFOLD", **kwargs):
    """Apply the PX patch — two states only:
      - BASELINE: nackt durchlassen
      - ACTIVE_MANIFOLD: vollständige PX-Architektur (gemma4-E2B-Rezeptur)
      - ACTIVE_MANIFOLD_RELAY: + verstärkbar Relay (d_width-Selbstinjektion)
    """
    if config_preset not in ("BASELINE", "ACTIVE_MANIFOLD",
                             "ACTIVE_MANIFOLD_LEAN", "ACTIVE_MANIFOLD_RELAY"):
        config_preset = "ACTIVE_MANIFOLD"
    if config_preset == "BASELINE":
        return False

    text_model = _resolve_text_model(model)
    if hasattr(text_model, "_px_original_forward"):
        print("[ternary-px] bereits gepatcht — Skip")
        return True
    config = text_model.config
    hidden_size, num_layers = config.hidden_size, config.num_hidden_layers

    # 1. Base Scale Defaults
    if hidden_size in SCALE_DEFAULTS:
        sd = SCALE_DEFAULTS[hidden_size]
        defaults = {"mode": "lti", "n_loops": sd["n_loops"], "beta": 0.05,
                    "gamma": sd["gamma"], "recur_start": sd["recur_start"],
                    "recur_end": sd["recur_end"], "bimodal_hub": sd["hub"],
                    "cgi_factor": 0.08, "num_layers": num_layers}
    else:
        defaults = {"mode": "lti", "n_loops": 8, "beta": 0.05,
                    "gamma": 0.08 * min(1152.0 / hidden_size, 1.5),
                    "recur_start": 5, "recur_end": 12, "bimodal_hub": 8,
                    "cgi_factor": 0.08, "num_layers": num_layers}

    # PX-default repetition_penalty (Token-Loop-Mitigation)
    defaults["repetition_penalty"] = 1.15
    defaults["no_repeat_ngram_size"] = 3

    # UI-Overrides durchreichen (relay_sign/alpha/layer, gamma, routing_mode)
    defaults.update(kwargs)

    text_model._px_config = defaults
    model_id = getattr(config, "_name_or_path", "unknown_model")
    text_model._px_calibrator = AutoCalibrator(
        hidden_size, calibration_steps=getattr(config, "px_calibration_steps", 10),
        model_id=model_id)

    # Resolve device and dtype
    device = next(text_model.parameters()).device
    dtype = next(text_model.parameters()).dtype

    # Three Mathematical Pillars: Observer + Symmetry Breaker + Dynamic Router
    text_model._px_injection_norm = torch.nn.LayerNorm(
        hidden_size, elementwise_affine=False, eps=1e-6).to(device=device, dtype=dtype)
    text_model._px_mephisto = MephistophelesOperator(hidden_size).to(device=device, dtype=dtype)
    text_model._px_aks = AksSensor()
    text_model._px_coupler = SingesseinCoupler(hidden_size).to(device=device, dtype=dtype)
    text_model._px_subj_sensor = SubjectiveSensor()
    text_model._px_azs = AntiZombieSensor(hidden_size).to(device=device, dtype=dtype)

    # Original forward sichern und ersetzen
    text_model._px_original_forward = text_model.forward
    text_model.forward = types.MethodType(_px_forward, text_model)

    # PX gen-kwargs attrs (generators._px_gen_kwargs liest sie)
    text_model._px_repetition_penalty = defaults.get("repetition_penalty", 1.15)
    text_model._px_no_repeat_ngram_size = defaults.get("no_repeat_ngram_size", 3)
    text_model._px_patched = True  # Marker für get_px_metrics ("active")

    # verstärkbar Relay (psychomotrik seite15/19): Re-Injektion der
    # modell-eigenen Zustands-Richtung d_width am post-recur Layer.
    # sign=+1 WIDE (aktiv bei ACTIVE_MANIFOLD_RELAY), −1 NARROW, 0 inactive.
    _relay_sign = defaults.get("relay_sign",
                               (+1 if config_preset == "ACTIVE_MANIFOLD_RELAY" else 0))
    if not getattr(text_model.config, "_name_or_path", None):
        outer_hf_id = getattr(getattr(model, "config", None), "_name_or_path", None)
        if outer_hf_id:
            text_model._px_hf_id = outer_hf_id
    _user_layer = defaults.get("relay_layer")
    _hf_id_for_layer = (getattr(text_model, "_px_hf_id", None)
                        or getattr(getattr(text_model, "config", None), "_name_or_path", None)
                        or getattr(getattr(model, "config", None), "_name_or_path", None))
    _relay_layer = _user_layer
    if _relay_layer is None and _hf_id_for_layer:
        _relay_layer = get_inject_layer_for_hf_id(_hf_id_for_layer)
    if _relay_layer is None:
        _relay_layer = {5120: 30}.get(hidden_size, min(26, num_layers - 26))
    install_relay(text_model, sign=_relay_sign,
                  alpha_frac=defaults.get("relay_alpha", 0.30),
                  layer=_relay_layer)

    print(f"[ternary-px] Active Manifold for L{num_layers} "
          f"(hybrid GDN/full@4, hidden={hidden_size}, loops={defaults['n_loops']}).")
    return True


def remove_px_patch(model):
    """Restore original model forward pass."""
    text_model = _resolve_text_model(model)
    remove_relay(text_model)
    if hasattr(text_model, "_px_original_forward"):
        text_model.forward = text_model._px_original_forward
        del text_model._px_original_forward

    for attr in [
        '_px_config', '_px_phi_val', '_px_aks_val', '_px_em_val', '_px_ent_val',
        '_px_loops_run', '_px_path', '_px_zone', '_px_zw_val', '_px_cognitive_signature',
        '_px_current_telemetry', '_px_last_metrics',
        '_task_kurtosis', '_task_jitter', '_task_token_diversity', '_px_zone_weights',
        '_px_calibrator', '_px_injection_norm', '_px_mephisto', '_px_aks', '_px_subj_sensor',
        '_px_azs', '_px_saved_input_ids',
        '_px_relay_handles', '_px_relay_cfg', '_px_hf_id',
        '_px_repetition_penalty', '_px_no_repeat_ngram_size', '_px_patched',
        '_px_gen_phi_sum', '_px_gen_phi_n',
    ]:
        if hasattr(text_model, attr):
            try:
                delattr(text_model, attr)
            except AttributeError:
                pass

    print("[ternary-px] Patch removed.")
    return True


def get_px_metrics(model):
    """Latest PX cognitive state — gleiche Keys wie gemma4_2b_px (SR-60 parity)."""
    tm = _resolve_text_model(model)
    ent = getattr(tm, "_px_ent_val", 0.0)
    if hasattr(ent, 'item'):
        ent = ent.item()
    out = {
        "phi": getattr(tm, "_px_phi_val", 1.0),
        "steps": getattr(tm, "_px_loops_run", 0),
        "path": getattr(tm, "_px_path", []),
        "zone": getattr(tm, "_px_zone", "UNKNOWN"),
        "zone_weights": getattr(tm, "_px_zw_val", {}),
        "cognitive_signature": getattr(tm, "_px_cognitive_signature", {}),
        "telemetry_trace": getattr(tm, "_px_current_telemetry", []),
        "aks_profile": {"correction_strength": getattr(tm, "_px_aks_val", 0.0)},
        "subjective_metrics": {"emancipation": getattr(tm, "_px_em_val", 0.0)},
        "entropy": float(ent),
    }
    # Decode-Mittelwert über die letzte Generation (Prefill-phi = Rekursionsqualität)
    n_dec = getattr(tm, "_px_gen_phi_n", 0)
    if n_dec:
        out["phi_decode_mean"] = round(getattr(tm, "_px_gen_phi_sum", 0.0) / n_dec, 4)
        out["decode_tokens"] = n_dec
    # phi=1.0 + zone UNKNOWN sind getattr-DEFAULTS ohne Daten (BASELINE);
    # active=False macht den Platzhalter sichtbar.
    out["active"] = hasattr(tm, "_px_patched")
    return out