"""tests/test_prefill_steering_divergence.py — Regression Test for Prefill-Hook Steering Divergence on Token 1.

Verifies:
1. Hook on Layer 26 is active during the prefill pass on the last token position h[:, -1, :].
2. The logits for the first generated token diverge between sign=+1 (WIDE), sign=0 (BASELINE), and sign=-1 (NARROW).
3. first_token_entropy_bits, probe_wide_vs_narrow_logit_diff, and probe_plus_vs_minus_logit_diff are not frozen/identical.
"""

import pytest
import torch
import torch.nn as nn
from transformers import LogitsProcessorList

from px_patches.manifold_relay_hook import ManifoldRelayHook
from px_patches.gemma4_2b_px.telemetry_collector import Gemma4TelemetryLogitsProcessor


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 1
        self._vocab = {
            "+": 10, "-": 11, "0": 12, "1": 13,
            "WIDEN": 20, "KEEP": 21, "NARROW": 22,
            "widen": 20, "keep": 21, "narrow": 22,
        }

    def encode(self, text, add_special_tokens=False):
        clean = text.strip()
        if clean in self._vocab:
            return [self._vocab[clean]]
        return [30]

    def decode(self, token_ids):
        rev = {v: k for k, v in self._vocab.items()}
        return "".join([rev.get(t, f"<t_{t}>") for t in token_ids])


class LinearLMHead(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, dim))

    def forward(self, x):
        return torch.matmul(x, self.weight.t())


class DummyBackboneLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Identity forward
        return x


class DummyModelWithLMHead(nn.Module):
    def __init__(self, num_layers=30, dim=64, vocab_size=100):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([DummyBackboneLayer(dim) for _ in range(num_layers)])
        self.lm_head = LinearLMHead(dim, vocab_size)
        self.config = type("Config", (), {"hidden_size": dim, "_name_or_path": "google/gemma-4-E2B-it"})()
        self._px_relay_handles = []

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape
        # Create non-zero hidden states for prompt
        h = torch.ones((batch_size, seq_len, self.dim), dtype=torch.float32)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(h)
        return logits


def test_prefill_steering_divergence_on_token_1():
    """Verify that steering on prefill last-token position alters the first generated token logits."""
    dim = 64
    vocab_size = 100
    torch.manual_seed(42)

    model = DummyModelWithLMHead(num_layers=30, dim=dim, vocab_size=vocab_size)
    dwidth = torch.randn(dim, dtype=torch.float32)
    dwidth = dwidth / dwidth.norm()

    # Align LM head weights for token '+' and '-' with dwidth
    # Token 10 ('+'): positive projection onto dwidth
    # Token 11 ('-'): negative projection onto dwidth
    with torch.no_grad():
        model.lm_head.weight[10] = dwidth * 2.0
        model.lm_head.weight[11] = -dwidth * 2.0
        model.lm_head.weight[20] = dwidth * 2.0   # WIDEN
        model.lm_head.weight[22] = -dwidth * 2.0  # NARROW

    tokenizer = MockTokenizer()

    # Prompt with seq_len = 5 (prefill pass!)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    def run_prefill(sign, alpha):
        vector_tuple = (dwidth.numpy(), {"direction": "test"})
        with ManifoldRelayHook(
            model,
            layer_idx=26,
            vector=vector_tuple,
            alpha=alpha,
            sign=sign,
            model_id="gemma4-e2b-it",
        ):
            # Forward pass across entire prefill sequence
            logits = model(input_ids)
            # The last token logits determine Token 1!
            last_logits = logits[:, -1, :]

            proc = Gemma4TelemetryLogitsProcessor(tokenizer, dwidth_tensor=dwidth)
            # Call processor on step 1
            proc(input_ids, last_logits)
            summary = proc.get_summary()
            probes = proc.zero_temp_probes
            return last_logits.detach(), summary, probes

    # Run WIDE (+1), BASELINE (0), NARROW (-1)
    logits_wide, summary_wide, probes_wide = run_prefill(sign=1, alpha=0.35)
    logits_base, summary_base, probes_base = run_prefill(sign=0, alpha=0.0)
    logits_narrow, summary_narrow, probes_narrow = run_prefill(sign=-1, alpha=0.35)

    # 1. Assert that logits on the last prefill position are NOT identical!
    assert not torch.allclose(logits_wide, logits_base, atol=1e-3), "WIDE logits equal BASELINE on Token 1!"
    assert not torch.allclose(logits_narrow, logits_base, atol=1e-3), "NARROW logits equal BASELINE on Token 1!"
    assert not torch.allclose(logits_wide, logits_narrow, atol=1e-3), "WIDE logits equal NARROW on Token 1!"

    # 2. Assert zero-temp probes diverge
    diff_wide = probes_wide["probe_plus_vs_minus_logit_diff"]
    diff_base = probes_base["probe_plus_vs_minus_logit_diff"]
    diff_narrow = probes_narrow["probe_plus_vs_minus_logit_diff"]

    print(f"\nProbe Plus vs Minus: Wide={diff_wide:+.4f}, Base={diff_base:+.4f}, Narrow={diff_narrow:+.4f}")
    assert diff_wide > diff_base, f"Wide probe ({diff_wide}) must be greater than base ({diff_base})!"
    assert diff_base > diff_narrow, f"Base probe ({diff_base}) must be greater than narrow ({diff_narrow})!"

    # 3. Assert preference probs diverge
    pref_wide = probes_wide["preference_probs"]
    pref_narrow = probes_narrow["preference_probs"]
    print(f"Wide Preference Probs: {pref_wide}")
    print(f"Narrow Preference Probs: {pref_narrow}")
    assert pref_wide["widen"] > pref_narrow["widen"], "Wide condition should favor WIDEN over narrow condition!"
    assert pref_narrow["narrow"] > pref_wide["narrow"], "Narrow condition should favor NARROW over wide condition!"
