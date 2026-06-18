import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union
from transformers.cache_utils import DynamicCache

class InfLLMCache(DynamicCache):
    """
    SR-64: Infinite Context Cache based on InfLLM and ReAttention.
    Integrates block-based memory and positionsagnostic retrieval.
    """
    def __init__(self, config, block_size: int = 128, r_tokens: int = 4, top_k_blocks: int = 8, sinks_count: int = 4):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.r_tokens = r_tokens
        self.top_k_blocks = top_k_blocks
        self.sinks_count = sinks_count
        
        # Long-term memory storage per layer
        # List of lists (one per layer)
        self.ltm_k: List[List[torch.Tensor]] = [] 
        self.ltm_v: List[List[torch.Tensor]] = []
        self.ltm_rk: List[List[torch.Tensor]] = [] # Representative keys
        
        # Buffer for incoming tokens per layer
        self.buffer_k: List[List[torch.Tensor]] = []
        self.buffer_v: List[List[torch.Tensor]] = []
        
        # Sinks per layer
        self.sinks_k: List[Optional[torch.Tensor]] = []
        self.sinks_v: List[Optional[torch.Tensor]] = []
        
        self.num_layers = config.num_hidden_layers
        for _ in range(self.num_layers):
            self.ltm_k.append([])
            self.ltm_v.append([])
            self.ltm_rk.append([])
            self.buffer_k.append([])
            self.buffer_v.append([])
            self.sinks_k.append(None)
            self.sinks_v.append(None)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overrides DynamicCache.update to handle block archival and retrieval.
        key_states: [B, H, T_new, D]
        """
        # 1. Store Sinks (first N tokens ever seen)
        if self.sinks_k[layer_idx] is None:
            # Check if we have enough tokens for sinks
            if key_states.size(-2) >= self.sinks_count:
                self.sinks_k[layer_idx] = key_states[:, :, :self.sinks_count, :].clone()
                self.sinks_v[layer_idx] = value_states[:, :, :self.sinks_count, :].clone()
            else:
                # Not enough yet, but we'll wait for the next update or just take what we have
                # For simplicity, let's just take the first ones if it's the very first call
                if self.get_seq_length(layer_idx) == 0:
                    self.sinks_k[layer_idx] = key_states.clone()
                    self.sinks_v[layer_idx] = value_states.clone()

        # 2. Add to buffer
        self.buffer_k[layer_idx].append(key_states)
        self.buffer_v[layer_idx].append(value_states)
        
        # 3. If buffer exceeds block size, archive to LTM
        current_buffer_len = sum(k.size(-2) for k in self.buffer_k[layer_idx])
        if current_buffer_len >= self.block_size:
            self._archive_block(layer_idx)
            
        # 4. Retrieval for ReAttention
        # We need the current Query to perform retrieval. 
        # But wait, 'update' is called with K, V. Where is Q?
        # Usually, Attention calls update with current K, V.
        # If we want to return a "Selected" KV set, we need to know the query.
        # We can pass 'q' in cache_kwargs if we patch the attention call.
        
        q = None
        if cache_kwargs and "query_states" in cache_kwargs:
            q = cache_kwargs["query_states"] # [B, H, T_q, D]
            
        # If we have a query, we perform retrieval
        if q is not None and len(self.ltm_k[layer_idx]) > 0:
            ret_k, ret_v = self._retrieve(layer_idx, q)
            # Concatenate Sinks + Retrieved + Current Buffer
            # Note: We return the "view" that attention will use
            k_parts = []
            v_parts = []
            if self.sinks_k[layer_idx] is not None:
                k_parts.append(self.sinks_k[layer_idx])
                v_parts.append(self.sinks_v[layer_idx])
            if ret_k is not None:
                k_parts.append(ret_k)
                v_parts.append(ret_v)
            
            # Local window (current buffer)
            local_k = torch.cat(self.buffer_k[layer_idx], dim=-2)
            local_v = torch.cat(self.buffer_v[layer_idx], dim=-2)
            k_parts.append(local_k)
            v_parts.append(local_v)
            
            return torch.cat(k_parts, dim=-2), torch.cat(v_parts, dim=-2)
        
        # Fallback: Just return the current buffer (acting like a normal cache)
        # Note: DynamicCache usually returns the full history.
        # To be "Infinite", we must only return what fits in VRAM.
        res_k = torch.cat(self.buffer_k[layer_idx], dim=-2)
        res_v = torch.cat(self.buffer_v[layer_idx], dim=-2)
        return res_k, res_v

    def _archive_block(self, layer_idx: int):
        all_k = torch.cat(self.buffer_k[layer_idx], dim=-2)
        all_v = torch.cat(self.buffer_v[layer_idx], dim=-2)
        
        # Take the first 'block_size' tokens
        block_k = all_k[:, :, :self.block_size, :]
        block_v = all_v[:, :, :self.block_size, :]
        
        # Keep the rest in buffer
        self.buffer_k[layer_idx] = [all_k[:, :, self.block_size:, :]]
        self.buffer_v[layer_idx] = [all_v[:, :, self.block_size:, :]]
        
        # Representative Tokens (Top-k magnitude)
        magnitudes = block_k.norm(dim=-1) # [B, H, T_block]
        _, indices = magnitudes.topk(self.r_tokens, dim=-1)
        r_k = torch.gather(block_k, -2, indices.unsqueeze(-1).expand(-1, -1, -1, block_k.size(-1)))
        
        self.ltm_k[layer_idx].append(block_k.cpu()) # Offload to CPU
        self.ltm_v[layer_idx].append(block_v.cpu())
        self.ltm_rk[layer_idx].append(r_k) # Keep RK on GPU for fast retrieval

    def _retrieve(self, layer_idx: int, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q: [B, H, T_q, D]
        # Calculate similarity with all RKs in LTM
        rk_stack = torch.cat(self.ltm_rk[layer_idx], dim=0) # [NumBlocks, B, H, r_tokens, D]
        # This indexing is tricky if B > 1. Assuming B=1 for now.
        # rk_stack: [NumBlocks, H, r_tokens, D]
        
        # Sim: [NumBlocks, H, T_q, r_tokens]
        # For now, use the last query token for retrieval
        q_last = q[:, :, -1:, :] # [B, H, 1, D]
        
        scores = []
        for i, rk in enumerate(self.ltm_rk[layer_idx]):
            # rk: [B, H, r_tokens, D]
            attn = torch.matmul(q_last, rk.transpose(-1, -2)) # [B, H, 1, r_tokens]
            score = attn.max(dim=-1)[0].mean()
            scores.append((score.item(), i))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        selected = [idx for _, idx in scores[:self.top_k_blocks]]
        selected.sort() # Chronological order
        
        ret_k = torch.cat([self.ltm_k[layer_idx][i].to(q.device) for i in selected], dim=-2)
        ret_v = torch.cat([self.ltm_v[layer_idx][i].to(q.device) for i in selected], dim=-2)
        
        return ret_k, ret_v

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # Sequence length from the perspective of the attention mechanism
        # (Sinks + retrieved blocks + local buffer)
        # But for 'position_ids' generation, we need the "real" total length.
        # DynamicCache usually tracks the real length.
        return sum(k.size(-2) for k in self.ltm_k[layer_idx]) * self.block_size + sum(k.size(-2) for k in self.buffer_k[layer_idx])

