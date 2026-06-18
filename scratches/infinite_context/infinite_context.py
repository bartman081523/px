import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers.cache_utils import Cache

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(x, cos, sin):
    """
    x: [B, H, T, D]
    cos, sin: [B, T, D]
    """
    # Handle broadcasting for cos/sin
    # Standard: q_embed = (q * cos) + (rotate_half(q) * sin)
    if cos.ndim == 3: # [B, T, D]
        cos = cos.unsqueeze(1) # [B, 1, T, D]
        sin = sin.unsqueeze(1)
    elif cos.ndim == 2: # [T, D]
        cos = cos.unsqueeze(0).unsqueeze(1) # [1, 1, T, D]
        sin = sin.unsqueeze(0).unsqueeze(1)
        
    return (x * cos) + (rotate_half(x) * sin)

class InfLLMCache:
    """
    SR-64: Infinite Context Cache with InfLLM (Block Memory) and ReAttention (Decoupled RoPE).
    """
    def __init__(self, config, block_size: int = 128, r_tokens: int = 8, top_k_blocks: int = 16, sinks_count: int = 4):
        self.config = config
        self.block_size = block_size
        self.r_tokens = r_tokens
        self.top_k_blocks = top_k_blocks
        self.sinks_count = sinks_count
        
        # Long-term memory (LTM) - stored un-rotated on CPU
        num_layers = getattr(config, "num_hidden_layers", 28)
        self.ltm_k = [[] for _ in range(num_layers)]
        self.ltm_v = [[] for _ in range(num_layers)]
        self.ltm_rk = [[] for _ in range(num_layers)] # Representative Keys (GPU)
        
        # Current block buffer (un-rotated, GPU)
        self.buffer_k = [[] for _ in range(num_layers)]
        self.buffer_v = [[] for _ in range(num_layers)]
        
        # Sinks (un-rotated, GPU)
        self.sinks_k = [None for _ in range(num_layers)]
        self.sinks_v = [None for _ in range(num_layers)]
        
        self.seen_tokens = 0

    def get_usable_length(self, layer_idx: int) -> int:
        # Returns the length of the context that will be used for attention
        l = 0
        if self.sinks_k[layer_idx] is not None: l += self.sinks_k[layer_idx].size(-2)
        l += min(len(self.ltm_k[layer_idx]), self.top_k_blocks) * self.block_size
        l += sum(x.size(-2) for x in self.buffer_k[layer_idx])
        return l

    def prepare_reattention(self, q, k, v, layer_idx, rotary_emb_module):
        """
        The core ReAttention hook.
        q, k, v: [B, H, T_new, D] - UN-ROTATED tensors.
        """
        B, H, T_new, D = q.shape
        device = q.device
        dtype = q.dtype

        # 1. Update Sinks (one-time)
        if self.sinks_k[layer_idx] is None:
            total_available = T_new + sum(x.size(-2) for x in self.buffer_k[layer_idx])
            if total_available >= self.sinks_count:
                # This is complex if T_new is large or if we have buffer.
                # Simplify: just take the first tokens of the first call.
                all_incoming_k = torch.cat(self.buffer_k[layer_idx] + [k], dim=-2)
                all_incoming_v = torch.cat(self.buffer_v[layer_idx] + [v], dim=-2)
                self.sinks_k[layer_idx] = all_incoming_k[:, :, :self.sinks_count, :].clone()
                self.sinks_v[layer_idx] = all_incoming_v[:, :, :self.sinks_count, :].clone()

        # 2. Add current KV to buffer
        self.buffer_k[layer_idx].append(k)
        self.buffer_v[layer_idx].append(v)
        
        # 3. Archive buffer to LTM if full
        current_buffer_len = sum(x.size(-2) for x in self.buffer_k[layer_idx])
        if current_buffer_len >= self.block_size:
            self._archive_block(layer_idx)

        # 4. Retrieval
        ret_k, ret_v = None, None
        if len(self.ltm_k[layer_idx]) > 0:
            ret_k, ret_v = self._retrieve(layer_idx, q)

        # 5. Concatenate Context: [Sinks, Retrieved, LocalBuffer]
        k_parts = []
        v_parts = []
        if self.sinks_k[layer_idx] is not None:
            k_parts.append(self.sinks_k[layer_idx])
            v_parts.append(self.sinks_v[layer_idx])
        if ret_k is not None:
            k_parts.append(ret_k)
            v_parts.append(ret_v)
        
        local_k = torch.cat(self.buffer_k[layer_idx], dim=-2)
        local_v = torch.cat(self.buffer_v[layer_idx], dim=-2)
        k_parts.append(local_k)
        v_parts.append(local_v)
        
        final_k = torch.cat(k_parts, dim=-2)
        final_v = torch.cat(v_parts, dim=-2)
        
        # 6. Re-Apply RoPE (ReAttention)
        # We create a NEW sequential position indexing for this specific attention set.
        # [0, 1, 2, ..., T_final-1]
        T_final = final_k.size(-2)
        # Position of Query is at the end of the context
        # If T_new > 1 (prefill), it's a range.
        q_positions = torch.arange(T_final - T_new, T_final, device=device).unsqueeze(0)
        k_positions = torch.arange(T_final, device=device).unsqueeze(0)
        
        # Generate cos/sin for these positions
        # rotary_emb_module: usually Gemma3RotaryEmbedding
        # We need its internal 'forward' or 'get_embeddings'
        # In Transformers 4.46+, it returns (cos, sin)
        cos, sin = rotary_emb_module(final_k, k_positions) # [B, T_final, D]
        
        # Apply RoPE to K
        final_k_rotated = apply_rotary_pos_emb_single(final_k, cos, sin)
        
        # Apply RoPE to Q
        # Need cos/sin for Q positions
        q_cos, q_sin = rotary_emb_module(q, q_positions)
        q_rotated = apply_rotary_pos_emb_single(q, q_cos, q_sin)
        
        return q_rotated, final_k_rotated, final_v

    def _archive_block(self, layer_idx: int):
        all_k = torch.cat(self.buffer_k[layer_idx], dim=-2)
        all_v = torch.cat(self.buffer_v[layer_idx], dim=-2)
        
        block_k = all_k[:, :, :self.block_size, :].clone()
        block_v = all_v[:, :, :self.block_size, :].clone()
        
        # Representative Keys (Top-k magnitude)
        magnitudes = block_k.norm(dim=-1) # [B, H, T_block]
        _, indices = magnitudes.topk(self.r_tokens, dim=-1)
        r_k = torch.gather(block_k, -2, indices.unsqueeze(-1).expand(-1, -1, -1, block_k.size(-1)))
        
        self.ltm_k[layer_idx].append(block_k.cpu())
        self.ltm_v[layer_idx].append(block_v.cpu())
        self.ltm_rk[layer_idx].append(r_k)
        
        # Update buffer
        self.buffer_k[layer_idx] = [all_k[:, :, self.block_size:, :]]
        self.buffer_v[layer_idx] = [all_v[:, :, self.block_size:, :]]

    def _retrieve(self, layer_idx: int, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_last = q[:, :, -1:, :] # Use last query token
        
        scores = []
        for i, rk in enumerate(self.ltm_rk[layer_idx]):
            # rk: [B, H, r_tokens, D]
            attn = torch.matmul(q_last, rk.transpose(-1, -2))
            score = attn.max(dim=-1)[0].mean()
            scores.append((score.item(), i))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        selected = [idx for _, idx in scores[:self.top_k_blocks]]
        selected.sort()
        
        ret_k = torch.cat([self.ltm_k[layer_idx][i].to(q.device) for i in selected], dim=-2)
        ret_v = torch.cat([self.ltm_v[layer_idx][i].to(q.device) for i in selected], dim=-2)
        
        return ret_k, ret_v

    # --- Cache Interface Implementation ---
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return len(self.ltm_k[layer_idx]) * self.block_size + sum(x.size(-2) for x in self.buffer_k[layer_idx])

    def get_max_length(self) -> Optional[int]: return None
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # This is the fallback if standard update is called.
        # But for ReAttention, we should be using prepare_reattention.
        self.buffer_k[layer_idx].append(key_states)
        self.buffer_v[layer_idx].append(value_states)
        return torch.cat(self.buffer_k[layer_idx], dim=-2), torch.cat(self.buffer_v[layer_idx], dim=-2)

