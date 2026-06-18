"""
Layer B — InfLLMCache + ReAttention (architectural infinite-context KV cache)
=============================================================================
Corrected, self-contained re-implementation of the InfLLM (block memory +
representative tokens) + ReAttention (position-agnostic top-k retrieval, RoPE
re-applied after selection) approach from the technical document
("LLMs_ Unendlicher Kontext durch Architektur.md").

This fixes the bug in the previous wip's `infinite_context.py`: that version
BYPASSED block retrieval for large prefills (T_new > 1024) and returned the full
un-rotated KV, producing exactly the N^2 prefill attention that OOMs on the
RTX 2060 (see bug_context.txt). Here the context is ALWAYS bounded:

    K_used = [ Sinks | Top-k Retrieved Blocks | Local Window ]

so the KV footprint is O(sinks + top_k_blocks*block_size + window_size) regardless
of total history length — the "infinite" property. Prefill is handled with the
same bounded context (sliding local window + retrieval) instead of a full-KV
bypass.

Correctness fixes vs the wip:
  * No prefill bypass — bounded context on every call.
  * LTM eviction cap (`max_ltm_blocks`) so CPU memory is bounded too.
  * Representative keys kept on CPU; brought to GPU only for scoring.
  * `get_seq_length` (real total, for position_ids / mask creation in the PX
    forward) is separated from `get_usable_length` (bounded attention length),
    fixing the position/mask length mismatch.
  * Multi-query retrieval (all query tokens) instead of only the last token.
  * `remove_reattention_patch` so `remove_px_patch` can cleanly revert.

Validated with mock rotary in test_inf_llm.py / test_layer_b_bounds.py.
GPU/model end-to-end validation is a separate, guarded step (see README).
"""

from typing import List, Dict, Any, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# RoPE helpers (operate on un-rotated KV stored in the cache)
# ---------------------------------------------------------------------------
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(x, cos, sin):
    """x: [B, H, T, D]; cos/sin: [B, T, D] or [T, D]."""
    if cos.ndim == 3:        # [B, T, D]
        cos = cos.unsqueeze(1)   # [B, 1, T, D]
        sin = sin.unsqueeze(1)
    elif cos.ndim == 2:      # [T, D]
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, T, D]
        sin = sin.unsqueeze(0).unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class InfLLMCache:
    """
    InfLLM (block memory) + ReAttention (decoupled RoPE) cache.

    All stored K/V are UN-ROTATED; RoPE is applied to the reassembled bounded
    context after retrieval (ReAttention).
    """

    def __init__(
        self,
        config,
        block_size: int = 128,
        r_tokens: int = 8,
        top_k_blocks: int = 16,
        sinks_count: int = 4,
        window_size: int = 512,
        max_ltm_blocks: int = 256,
    ):
        self.config = config
        self.block_size = block_size
        self.r_tokens = r_tokens
        self.top_k_blocks = top_k_blocks
        self.sinks_count = sinks_count
        self.window_size = window_size        # local attention window cap
        self.max_ltm_blocks = max_ltm_blocks  # CPU LTM eviction cap

        num_layers = getattr(config, "num_hidden_layers", 28)
        self.ltm_k: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]   # CPU
        self.ltm_v: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]   # CPU
        self.ltm_rk: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]  # CPU (rep keys)

        self.buffer_k: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        self.buffer_v: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]

        self.sinks_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self.sinks_v: List[Optional[torch.Tensor]] = [None] * num_layers

        self.seen_tokens = 0  # real total tokens seen (for get_seq_length)

    # -- length accounting -------------------------------------------------
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Real total tokens seen — used by the PX forward for position_ids /
        causal mask creation (must reflect history, not the bounded attention set)."""
        archived = len(self.ltm_k[layer_idx]) * self.block_size
        local = sum(x.size(-2) for x in self.buffer_k[layer_idx])
        sinks = self.sinks_count if self.sinks_k[layer_idx] is not None else 0
        # sinks are a copy of the first tokens already counted in archived(0)/local
        # at the very first call, so we avoid double counting by subtracting overlap.
        return self.seen_tokens

    def get_usable_length(self, layer_idx: int = 0) -> int:
        """Bounded attention length actually used this step:
        sinks + retrieved + local-window. This is what the K tensor will be."""
        l = 0
        if self.sinks_k[layer_idx] is not None:
            l += self.sinks_k[layer_idx].size(-2)
        n_blocks = min(len(self.ltm_k[layer_idx]), self.top_k_blocks)
        l += n_blocks * self.block_size
        local = sum(x.size(-2) for x in self.buffer_k[layer_idx])
        l += min(local, self.window_size)
        return l

    def get_max_length(self) -> Optional[int]:
        return None

    # -- core ReAttention hook --------------------------------------------
    def prepare_reattention(self, q, k, v, layer_idx, rotary_emb_module, read_only=False, **kwargs):
        """
        q, k, v: [B, H, T_new, D] — UN-ROTATED. Returns (q_rot, k_rot, v) where
        k_rot/v are the bounded, re-assembled, RoPE'd context [Sinks|Ret|Local].
        """
        B, H, T_new, D = q.shape
        device = q.device

        # 1. Sinks (one-time, from the first tokens ever seen)
        if self.sinks_k[layer_idx] is None and T_new >= self.sinks_count:
            self.sinks_k[layer_idx] = k[:, :, :self.sinks_count, :].clone()
            self.sinks_v[layer_idx] = v[:, :, :self.sinks_count, :].clone()

        # 2. Append current KV to the local buffer and archive full blocks to LTM
        if not read_only:
            self.buffer_k[layer_idx].append(k)
            self.buffer_v[layer_idx].append(v)
            self.seen_tokens += T_new
            if sum(x.size(-2) for x in self.buffer_k[layer_idx]) >= self.block_size:
                self._archive_block(layer_idx)

        # 3. Retrieval (multi-query) — bounded global memory
        ret_k, ret_v = None, None
        if len(self.ltm_k[layer_idx]) > 0:
            ret_k, ret_v = self._retrieve(layer_idx, q)

        # 4. Local window (capped) — most recent tokens, un-rotated. Buffer may be
        # empty if the current chunk was an exact multiple of block_size and got
        # fully archived; in that case there is no local contribution.
        k_parts, v_parts = [], []
        if self.sinks_k[layer_idx] is not None:
            k_parts.append(self.sinks_k[layer_idx])
            v_parts.append(self.sinks_v[layer_idx])
        if ret_k is not None:
            k_parts.append(ret_k)
            v_parts.append(ret_v)
        if self.buffer_k[layer_idx]:
            local_k = torch.cat(self.buffer_k[layer_idx], dim=-2)
            local_v = torch.cat(self.buffer_v[layer_idx], dim=-2)
            if local_k.size(-2) > self.window_size:
                local_k = local_k[:, :, -self.window_size:, :]
                local_v = local_v[:, :, -self.window_size:, :]
            k_parts.append(local_k)
            v_parts.append(local_v)

        if not k_parts:  # degenerate: nothing to attend to yet
            return q, k, v
        final_k = torch.cat(k_parts, dim=-2)
        final_v = torch.cat(v_parts, dim=-2)

        # 6. Re-apply RoPE with FRESH sequential positions (ReAttention): the
        # retrieved/sinks blocks ignore their original absolute distance.
        T_final = final_k.size(-2)
        k_positions = torch.arange(T_final, device=device).unsqueeze(0)
        q_positions = torch.arange(T_final - T_new, T_final, device=device).unsqueeze(0)

        cos, sin = rotary_emb_module(final_k, k_positions)
        final_k_rot = apply_rotary_pos_emb_single(final_k, cos, sin)

        q_cos, q_sin = rotary_emb_module(q, q_positions)
        q_rot = apply_rotary_pos_emb_single(q, q_cos, q_sin)

        return q_rot, final_k_rot, final_v

    # -- block archival / retrieval ---------------------------------------
    def _archive_block(self, layer_idx: int):
        all_k = torch.cat(self.buffer_k[layer_idx], dim=-2)
        all_v = torch.cat(self.buffer_v[layer_idx], dim=-2)

        while all_k.size(-2) >= self.block_size:
            block_k = all_k[:, :, :self.block_size, :].clone()
            block_v = all_v[:, :, :self.block_size, :].clone()
            # Representative tokens: top-r by key norm (InfLLM-style max-pool proxy)
            magnitudes = block_k.norm(dim=-1)                      # [B, H, T_block]
            _, indices = magnitudes.topk(self.r_tokens, dim=-1)     # [B, H, r]
            r_k = torch.gather(
                block_k, -2,
                indices.unsqueeze(-1).expand(-1, -1, -1, block_k.size(-1)),
            )
            self.ltm_k[layer_idx].append(block_k.cpu())
            self.ltm_v[layer_idx].append(block_v.cpu())
            self.ltm_rk[layer_idx].append(r_k.cpu())
            all_k = all_k[:, :, self.block_size:, :]
            all_v = all_v[:, :, self.block_size:, :]

        self.buffer_k[layer_idx] = [all_k] if all_k.size(-2) > 0 else []
        self.buffer_v[layer_idx] = [all_v] if all_v.size(-2) > 0 else []

        # Eviction: bound CPU LTM (FIFO of oldest blocks)
        while len(self.ltm_k[layer_idx]) > self.max_ltm_blocks:
            self.ltm_k[layer_idx].pop(0)
            self.ltm_v[layer_idx].pop(0)
            self.ltm_rk[layer_idx].pop(0)

    def _retrieve(self, layer_idx: int, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-query block scoring against representative keys."""
        n_blocks = len(self.ltm_rk[layer_idx])
        k_blocks = min(self.top_k_blocks, n_blocks)
        if k_blocks == 0:
            return None, None

        # Stack representative keys: [n_blocks, B, H, r, D] on GPU for scoring
        rk_stack = torch.stack(self.ltm_rk[layer_idx], dim=0).to(q.device)  # [N, B, H, r, D]
        # Score: q [B, H, T_q, D] vs rk [N, B, H, r, D] -> [N, B, H, T_q, r]
        # Use the max relevance over query tokens and rep tokens, then mean over heads.
        scores = torch.einsum("bhtd,nbhrd->nbhtr", q, rk_stack)              # [N, B, H, T, r]
        block_scores = scores.amax(dim=-1).amax(dim=-1).mean(dim=(1, 2))   # [N]
        k_blocks = min(k_blocks, n_blocks)
        top = torch.topk(block_scores, k=k_blocks).indices.tolist()
        top.sort()  # chronological order (ReAttention preserves relative order)

        ret_k = torch.cat([self.ltm_k[layer_idx][i].to(q.device) for i in top], dim=-2)
        ret_v = torch.cat([self.ltm_v[layer_idx][i].to(q.device) for i in top], dim=-2)
        return ret_k, ret_v

    # -- fallback Cache interface -----------------------------------------
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.buffer_k[layer_idx].append(key_states)
        self.buffer_v[layer_idx].append(value_states)
        return torch.cat(self.buffer_k[layer_idx], dim=-2), torch.cat(self.buffer_v[layer_idx], dim=-2)


# ---------------------------------------------------------------------------
# Attention patch (Gemma3) — surgical, with bounded-context mask handling
# ---------------------------------------------------------------------------
_PATCHED_FLAG = "_px_reattention_patched"
_ORIG_FORWARD_ATTR = "_px_reattention_orig_forward"


def _px_attention_forward(self, hidden_states, position_embeddings=None,
                           attention_mask=None, past_key_values=None, **kwargs):
    """Surgical Gemma3Attention forward supporting ReAttention."""
    import torch.nn.functional as F
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

    if past_key_values is not None and hasattr(past_key_values, "prepare_reattention"):
        read_only = getattr(past_key_values, "_read_only", False)
        query_states, key_states, value_states = past_key_values.prepare_reattention(
            query_states, key_states, value_states, self.layer_idx,
            self.rotary_emb, read_only=read_only,
        )
        # Rebuild mask to the bounded K length (Sinks prepended => pad left with 0).
        T_q, T_k = query_states.size(-2), key_states.size(-2)
        if attention_mask is not None:
            T_orig = attention_mask.size(-1)
            if T_k > T_orig:
                attention_mask = F.pad(attention_mask, (T_k - T_orig, 0), value=0.0)
            elif T_k < T_orig:
                attention_mask = attention_mask[..., -T_k:]
        elif T_q > 1:
            # Build a fresh causal mask matching the bounded context.
            mask = torch.full((T_q, T_k), float("-inf"), device=query_states.device, dtype=query_states.dtype)
            mask = torch.triu(mask, diagonal=T_k - T_q + 1)
            attention_mask = mask.unsqueeze(0).unsqueeze(0)
    else:
        from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    from transformers.models.gemma3.modeling_gemma3 import (
        ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
    )
    attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward,
    )

    attn_output, attn_weights = attn_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def apply_reattention_patch(model):
    """Patch all Gemma3Attention modules to use ReAttention. Idempotent."""
    import types
    patched = 0
    for _, module in model.named_modules():
        if "Gemma3Attention" in type(module).__name__:
            if not getattr(module, _PATCHED_FLAG, False):
                setattr(module, _ORIG_FORWARD_ATTR, module.forward)
            module.forward = types.MethodType(_px_attention_forward, module)
            setattr(module, _PATCHED_FLAG, True)
            patched += 1
    print(f"[InfLLM] Patched {patched} attention modules with ReAttention.")
    return patched


def remove_reattention_patch(model):
    """Revert ReAttention on all patched modules (clean BASELINE restore)."""
    reverted = 0
    for _, module in model.named_modules():
        if getattr(module, _PATCHED_FLAG, False):
            orig = getattr(module, _ORIG_FORWARD_ATTR, None)
            if orig is not None:
                module.forward = orig
            delattr(module, _PATCHED_FLAG)
            if hasattr(module, _ORIG_FORWARD_ATTR):
                delattr(module, _ORIG_FORWARD_ATTR)
            reverted += 1
    print(f"[InfLLM] Reverted ReAttention on {reverted} attention modules.")
    return reverted