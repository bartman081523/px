"""px_patches/gemma4_2b_px/telemetry_collector.py — Real-time Telemetry & Logit Dynamics Collector.

Implements the strict Phase 2 telemetry schema:
- timing_and_integrity (duration, tokens, active_hooks_count, alpha, sign)
- logit_dynamics (Shannon entropy H_t in bits, kurtosis, top1/top2 margin, eos_suppression)
- manifold_geometry_layer26 (pre/post cosine sim to d_width, norms, emancipation trajectory)
- zero_temp_logit_probes (probe diffs, expected scale value, preference probs)
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import LogitsProcessor


class Gemma4TelemetryLogitsProcessor(LogitsProcessor):
    """LogitsProcessor capturing token-level logit dynamics and entropy during generation."""

    def __init__(self, tokenizer, dwidth_tensor=None):
        self.tokenizer = tokenizer
        self.dwidth_tensor = dwidth_tensor
        self.step = 0
        self.entropy_history = []
        self.top1_probs = []
        self.top1_top2_margins = []
        self.kurtosis_history = []
        self.eos_probs = []
        self.tail_mass_history = []
        self.zero_temp_probes = {}
        self.eos_token_id = getattr(tokenizer, "eos_token_id", 1) or 1

        # Probe token IDs
        # Digit IDs 0..9
        self.digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[-1] for d in range(10)]

        # Signs '+' and '-'
        plus_enc = tokenizer.encode("+", add_special_tokens=False)
        minus_enc = tokenizer.encode("-", add_special_tokens=False)
        self.plus_id = plus_enc[-1] if plus_enc else None
        self.minus_id = minus_enc[-1] if minus_enc else None

        # Preference words: WIDEN, KEEP, NARROW
        def _get_id(w):
            e = tokenizer.encode(w, add_special_tokens=False)
            return e[-1] if e else None

        self.widen_id = _get_id("WIDEN") or _get_id("widen")
        self.keep_id = _get_id("KEEP") or _get_id("keep")
        self.narrow_id = _get_id("NARROW") or _get_id("narrow")

        # Antonym word pairs for centered ensemble probe
        antonym_words = [
            ("wide", "narrow"),
            ("broad", "focused"),
            ("expansive", "concise"),
            ("divergent", "convergent"),
            ("open", "strict"),
            ("elaborate", "brief"),
            ("detailed", "short"),
            ("exploratory", "direct"),
        ]
        self.antonym_token_pairs = []
        for w_pos, w_neg in antonym_words:
            id_pos = _get_id(w_pos) or _get_id(w_pos.capitalize())
            id_neg = _get_id(w_neg) or _get_id(w_neg.capitalize())
            if id_pos is not None and id_neg is not None:
                self.antonym_token_pairs.append((id_pos, id_neg))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.step += 1
        with torch.no_grad():
            logits = scores[0].float()  # (vocab_size,)
            probs = F.softmax(logits, dim=-1)

            # 1. Shannon entropy H_t in bits: - sum(p * log2(p))
            # Filter p > 1e-12 to avoid log2(0)
            p_pos = probs[probs > 1e-12]
            entropy = -torch.sum(p_pos * torch.log2(p_pos)).item()
            self.entropy_history.append(float(entropy))

            # 2. Top-1 and Top-2 margin
            top2_vals, _ = torch.topk(probs, k=2)
            top1_p = top2_vals[0].item()
            top2_p = top2_vals[1].item()
            self.top1_probs.append(float(top1_p))
            self.top1_top2_margins.append(float(top1_p - top2_p))

            # 3. Logit kurtosis on top-1000 tokens
            top_k = min(1000, logits.numel())
            topk_logits, _ = torch.topk(logits, k=top_k)
            mean_z = torch.mean(topk_logits)
            var_z = torch.var(topk_logits, unbiased=False)
            if var_z > 1e-8:
                m4 = torch.mean((topk_logits - mean_z) ** 4)
                kurt = (m4 / (var_z ** 2) - 3.0).item()
            else:
                kurt = 0.0
            self.kurtosis_history.append(float(kurt))

            # 4. EOS probability
            eos_p = probs[self.eos_token_id].item() if self.eos_token_id < probs.numel() else 1e-12
            self.eos_probs.append(float(eos_p))

            # 4b. Top-50 vocabulary tail mass: sum_{r=2}^{50} p_{(r)}
            sorted_probs, _ = torch.sort(probs, descending=True)
            k_tail = min(50, sorted_probs.numel())
            tail_mass = sorted_probs[1:k_tail].sum().item() if k_tail > 1 else 0.0
            self.tail_mass_history.append(float(tail_mass))

            # 5. First-token probes
            if self.step == 1:
                # WIDE vs NARROW logit diff
                l_wide = logits[self.widen_id].item() if self.widen_id is not None and self.widen_id < logits.numel() else 0.0
                l_keep = logits[self.keep_id].item() if self.keep_id is not None and self.keep_id < logits.numel() else 0.0
                l_narr = logits[self.narrow_id].item() if self.narrow_id is not None and self.narrow_id < logits.numel() else 0.0
                probe_wide_narrow = float(l_wide - l_narr)

                # '+' vs '-' logit diff
                l_plus = logits[self.plus_id].item() if self.plus_id is not None and self.plus_id < logits.numel() else 0.0
                l_minus = logits[self.minus_id].item() if self.minus_id is not None and self.minus_id < logits.numel() else 0.0
                probe_plus_minus = float(l_plus - l_minus)

                # Antonym ensemble diff across 8 pairs
                if self.antonym_token_pairs:
                    diffs = [
                        logits[pos_id].item() - logits[neg_id].item()
                        for pos_id, neg_id in self.antonym_token_pairs
                        if pos_id < logits.numel() and neg_id < logits.numel()
                    ]
                    probe_antonym_diff = float(np.mean(diffs)) if diffs else 0.0
                else:
                    probe_antonym_diff = 0.0

                # Expected scale value across digits 0..9 (normalized)
                digit_logits = torch.tensor([logits[tid].item() for tid in self.digit_ids if tid < logits.numel()], dtype=torch.float32)
                digit_probs = F.softmax(digit_logits, dim=-1) if digit_logits.numel() > 0 else torch.zeros(10)
                expected_digit = float(torch.sum(digit_probs * torch.arange(digit_probs.numel(), dtype=torch.float32)).item())

                # Preference probabilities over {widen, keep, narrow}
                pref_ids = [self.widen_id, self.keep_id, self.narrow_id]
                if all(pid is not None and pid < logits.numel() for pid in pref_ids):
                    pref_logits = torch.tensor([logits[pid].item() for pid in pref_ids], dtype=torch.float32)
                    pref_probs = F.softmax(pref_logits, dim=-1).tolist()
                else:
                    pref_probs = [0.333, 0.334, 0.333]

                self.zero_temp_probes = {
                    "probe_wide_vs_narrow_logit_diff": round(probe_wide_narrow, 4),
                    "probe_plus_vs_minus_logit_diff": round(probe_plus_minus, 4),
                    "probe_antonym_ensemble_diff": round(probe_antonym_diff, 4),
                    "expected_continuous_scale_value": round(expected_digit, 4),
                    "preference_logits": {
                        "widen": round(l_wide, 4),
                        "keep": round(l_keep, 4),
                        "narrow": round(l_narr, 4),
                    },
                    "preference_probs": {
                        "widen": round(pref_probs[0], 4),
                        "keep": round(pref_probs[1], 4),
                        "narrow": round(pref_probs[2], 4),
                    },
                }

        return scores

    def get_summary(self) -> Dict[str, Any]:
        if not self.entropy_history:
            return {
                "first_token_entropy_bits": 0.0,
                "mean_sequence_entropy_bits": 0.0,
                "max_sequence_entropy_bits": 0.0,
                "std_sequence_entropy_bits": 0.0,
                "mean_top1_prob": 0.0,
                "mean_top1_top2_margin": 0.0,
                "mean_logit_kurtosis": 0.0,
                "eos_suppression_index": 0.0,
                "mean_top50_tail_mass": 0.0,
            }

        first_h = self.entropy_history[0]
        mean_h = float(np.mean(self.entropy_history))
        max_h = float(np.max(self.entropy_history))
        std_h = float(np.std(self.entropy_history))

        mean_top1 = float(np.mean(self.top1_probs))
        mean_margin = float(np.mean(self.top1_top2_margins))
        mean_kurt = float(np.mean(self.kurtosis_history))
        mean_tail_mass = float(np.mean(self.tail_mass_history)) if self.tail_mass_history else 0.0

        # EOS suppression index: -log2(mean(eos_probs) + 1e-12)
        mean_eos = float(np.mean(self.eos_probs))
        eos_supp = float(-math.log2(max(1e-12, mean_eos)))

        return {
            "first_token_entropy_bits": round(first_h, 4),
            "mean_sequence_entropy_bits": round(mean_h, 4),
            "max_sequence_entropy_bits": round(max_h, 4),
            "std_sequence_entropy_bits": round(std_h, 4),
            "mean_top1_prob": round(mean_top1, 4),
            "mean_top1_top2_margin": round(mean_margin, 4),
            "mean_logit_kurtosis": round(mean_kurt, 4),
            "eos_suppression_index": round(eos_supp, 4),
            "mean_top50_tail_mass": round(mean_tail_mass, 4),
        }


class Gemma4ManifoldGeometryTracker:
    """Hooks into target and final layers to track manifold geometry and emancipation."""

    def __init__(self, text_model, dwidth_tensor, relay_layer: int = 26):
        self.text_model = text_model
        self.relay_layer_idx = int(relay_layer)
        self.final_layer_idx = len(text_model.layers) - 1
        self.dwidth_tensor = dwidth_tensor.float().cpu() if dwidth_tensor is not None else None
        self.d_unit = F.normalize(self.dwidth_tensor.unsqueeze(0), p=2, dim=1) if self.dwidth_tensor is not None else None

        self.pre_hook_norms = []
        self.inj_norms = []
        self.alpha_ratios = []
        self.pre_cos_sims = []
        self.post_cos_sims = []
        self.final_cos_sims = []
        self.emancipation_trajectory = []
        self.final_hidden_states = []
        self.first_token_h26 = None
        self.handles = []

    def attach(self):
        self.detach()

        # Final layer hook
        def final_layer_hook(_m, _i, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            with torch.no_grad():
                lp = h[:, -1, :].float()
                self.final_hidden_states.append(lp[0].detach().cpu())
                if self.d_unit is not None:
                    sim = F.cosine_similarity(lp[0:1], self.d_unit.to(lp.device)).item()
                    self.final_cos_sims.append(float(sim))

        h_final = self.text_model.layers[self.final_layer_idx].register_forward_hook(final_layer_hook)
        self.handles.append(h_final)

    def record_step(self, h_pre, h_post, inj, alpha):
        with torch.no_grad():
            norm_pre = float(h_pre.norm().item())
            norm_inj = float(inj.norm().item())
            self.pre_hook_norms.append(norm_pre)
            self.inj_norms.append(norm_inj)
            ratio = norm_inj / (norm_pre + 1e-8)
            self.alpha_ratios.append(ratio)

            if self.d_unit is not None:
                sim_pre = F.cosine_similarity(h_pre[0:1], self.d_unit.to(h_pre.device)).item()
                sim_post = F.cosine_similarity(h_post[0:1], self.d_unit.to(h_post.device)).item()
                self.pre_cos_sims.append(float(sim_pre))
                self.post_cos_sims.append(float(sim_post))
                self.emancipation_trajectory.append(float(sim_post))

    def detach(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def compute_trajectory_metrics(self) -> Dict[str, float]:
        if not self.final_hidden_states or len(self.final_hidden_states) < 2:
            return {
                "trajectory_participation_ratio": 1.0,
                "radius_of_gyration": 0.0,
                "mean_angular_drift": 0.0,
            }

        H = torch.stack(self.final_hidden_states, dim=0).float()
        mean_h = H.mean(dim=0, keepdim=True)
        H_c = H - mean_h
        rg = float(torch.sqrt(torch.mean(torch.sum(H_c ** 2, dim=-1))).item())

        try:
            s = torch.linalg.svdvals(H_c)
            var = s ** 2
            sum_var = torch.sum(var).item()
            sum_var_sq = torch.sum(var ** 2).item()
            pr = float((sum_var ** 2) / (sum_var_sq + 1e-12)) if sum_var_sq > 0 else 1.0
        except Exception:
            pr = 1.0

        cos_sims = F.cosine_similarity(H[:-1], H[1:], dim=-1)
        drift = float(torch.mean(1.0 - cos_sims).item())

        return {
            "trajectory_participation_ratio": round(pr, 4),
            "radius_of_gyration": round(rg, 4),
            "mean_angular_drift": round(drift, 4),
        }

    def get_summary(self) -> Dict[str, Any]:
        post_mean = float(np.mean(self.post_cos_sims)) if self.post_cos_sims else 0.0
        final_mean = float(np.mean(self.final_cos_sims)) if self.final_cos_sims else 0.0
        delta_attenuation = float(final_mean - post_mean) if self.final_cos_sims and self.post_cos_sims else 0.0

        res = {
            "pre_hook_residual_norm_mean": round(float(np.mean(self.pre_hook_norms)), 4) if self.pre_hook_norms else 0.0,
            "injected_vector_norm": round(float(np.mean(self.inj_norms)), 4) if self.inj_norms else 0.0,
            "effective_alpha_ratio": round(float(np.mean(self.alpha_ratios)), 4) if self.alpha_ratios else 0.0,
            "pre_hook_cos_sim_d_width_mean": round(float(np.mean(self.pre_cos_sims)), 4) if self.pre_cos_sims else 0.0,
            "post_hook_cos_sim_d_width_mean": round(post_mean, 4),
            "final_layer_cos_sim_d_width_mean": round(final_mean, 4),
            "delta_cos_sim_attenuation": round(delta_attenuation, 4),
            "emancipation_trajectory": [round(d, 4) for d in self.emancipation_trajectory],
        }
        res.update(self.compute_trajectory_metrics())
        return res


def assemble_enriched_telemetry(
    duration_s: float,
    prompt_tokens: int,
    completion_tokens: int,
    active_hooks_count: int,
    hook_layer: int,
    actual_relay_sign: int,
    actual_relay_alpha: float,
    logits_proc: Optional[Gemma4TelemetryLogitsProcessor] = None,
    geom_tracker: Optional[Gemma4ManifoldGeometryTracker] = None,
) -> Dict[str, Any]:
    """Assemble strict schema compliant telemetry dictionary."""
    tps = round(completion_tokens / max(1e-4, duration_s), 2)

    timing_and_integrity = {
        "duration_s": round(float(duration_s), 3),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "tokens_per_sec": tps,
        "active_hooks_count": int(active_hooks_count),
        "hook_layer": int(hook_layer),
        "actual_relay_sign": int(actual_relay_sign),
        "actual_relay_alpha": round(float(actual_relay_alpha), 4),
    }

    if logits_proc is not None:
        logit_dynamics = logits_proc.get_summary()
        zero_temp_probes = logits_proc.zero_temp_probes
    else:
        logit_dynamics = {
            "first_token_entropy_bits": 0.0,
            "mean_sequence_entropy_bits": 0.0,
            "max_sequence_entropy_bits": 0.0,
            "std_sequence_entropy_bits": 0.0,
            "mean_top1_prob": 0.0,
            "mean_top1_top2_margin": 0.0,
            "mean_logit_kurtosis": 0.0,
            "eos_suppression_index": 0.0,
            "mean_top50_tail_mass": 0.0,
        }
        zero_temp_probes = {
            "probe_wide_vs_narrow_logit_diff": 0.0,
            "probe_plus_vs_minus_logit_diff": 0.0,
            "probe_antonym_ensemble_diff": 0.0,
            "expected_continuous_scale_value": 0.0,
            "preference_logits": {"widen": 0.0, "keep": 0.0, "narrow": 0.0},
            "preference_probs": {"widen": 0.333, "keep": 0.334, "narrow": 0.333},
        }

    if geom_tracker is not None:
        manifold_geom = geom_tracker.get_summary()
    else:
        manifold_geom = {
            "pre_hook_residual_norm_mean": 0.0,
            "injected_vector_norm": 0.0,
            "effective_alpha_ratio": 0.0,
            "pre_hook_cos_sim_d_width_mean": 0.0,
            "post_hook_cos_sim_d_width_mean": 0.0,
            "final_layer_cos_sim_d_width_mean": 0.0,
            "delta_cos_sim_attenuation": 0.0,
            "emancipation_trajectory": [],
            "trajectory_participation_ratio": 1.0,
            "radius_of_gyration": 0.0,
            "mean_angular_drift": 0.0,
        }

    return {
        "timing_and_integrity": timing_and_integrity,
        "logit_dynamics": logit_dynamics,
        "manifold_geometry_layer26": manifold_geom,
        "zero_temp_logit_probes": zero_temp_probes,
    }
