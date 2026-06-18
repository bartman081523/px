import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple

class InfLLMBlockMemory:
    """
    InfLLM-style block-based memory for KV cache.
    Stores historical blocks and provides representative tokens for retrieval.
    """
    def __init__(self, block_size: int = 128, r_tokens: int = 4, device: str = "cpu"):
        self.block_size = block_size
        self.r_tokens = r_tokens # Number of representative tokens per block
        self.device = device
        
        # storage: List of dicts { 'k': tensor, 'v': tensor, 'r_k': tensor }
        self.blocks: List[Dict[str, torch.Tensor]] = []
        
        # Buffer for the current filling block
        self.current_k: List[torch.Tensor] = []
        self.current_v: List[torch.Tensor] = []

    def add_kv(self, k: torch.Tensor, v: torch.Tensor):
        """
        k, v: [B, H, 1, D] - Single token KV
        """
        self.current_k.append(k.to(self.device))
        self.current_v.append(v.to(self.device))
        
        if len(self.current_k) >= self.block_size:
            self._finalize_block()

    def _finalize_block(self):
        block_k = torch.cat(self.current_k, dim=-2) # [B, H, T_block, D]
        block_v = torch.cat(self.current_v, dim=-2)
        
        # Select representative tokens (InfLLM uses max-pooling or top-k magnitude)
        # Here we use a simple top-k magnitude across the sequence dimension for each head
        # k: [B, H, T_block, D]
        magnitudes = block_k.norm(dim=-1) # [B, H, T_block]
        _, indices = magnitudes.topk(self.r_tokens, dim=-1) # [B, H, r_tokens]
        
        # Gather representative keys
        # We need to broadcast indices to D dimension
        # indices shape: [B, H, r_tokens]
        # block_k shape: [B, H, T_block, D]
        r_k = torch.gather(block_k, -2, indices.unsqueeze(-1).expand(-1, -1, -1, block_k.size(-1)))
        
        self.blocks.append({
            'k': block_k,
            'v': block_v,
            'r_k': r_k
        })
        
        self.current_k = []
        self.current_v = []

    def retrieve(self, q: torch.Tensor, top_k: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: [B, H, 1, D]
        returns concatenated K, V for the most relevant blocks.
        """
        if not self.blocks:
            return None, None
            
        q = q.to(self.device)
        
        scores = []
        for i, block in enumerate(self.blocks):
            # r_k: [B, H, r_tokens, D]
            # q: [B, H, 1, D]
            # attn: [B, H, 1, r_tokens]
            attn = torch.matmul(q, block['r_k'].transpose(-1, -2))
            max_score = attn.max(dim=-1)[0].mean() # Average max score across heads for simplicity
            scores.append((max_score.item(), i))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        selected_indices = [idx for score, idx in scores[:top_k]]
        # Maintain chronological order for ReAttention
        selected_indices.sort()
        
        ret_k = torch.cat([self.blocks[i]['k'] for i in selected_indices], dim=-2)
        ret_v = torch.cat([self.blocks[i]['v'] for i in selected_indices], dim=-2)
        
        return ret_k, ret_v

class ReAttentionManager:
    """
    Implements ReAttention: Decoupling Retrieval from Positioning.
    """
    def __init__(self, block_memory: InfLLMBlockMemory, sinks_count: int = 4):
        self.block_memory = block_memory
        self.sinks_count = sinks_count
        self.sinks_k: Optional[torch.Tensor] = None
        self.sinks_v: Optional[torch.Tensor] = None
        self.total_tokens_seen = 0

    def update_sinks(self, k: torch.Tensor, v: torch.Tensor):
        if self.sinks_k is None:
            self.sinks_k = k[:, :, :self.sinks_count, :].clone()
            self.sinks_v = v[:, :, :self.sinks_count, :].clone()

    def get_context(self, q: torch.Tensor, local_k: torch.Tensor, local_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves relevant historical blocks and concatenates with sinks and local context.
        """
        ret_k, ret_v = self.block_memory.retrieve(q)
        
        k_parts = []
        v_parts = []
        
        if self.sinks_k is not None:
            k_parts.append(self.sinks_k)
            v_parts.append(self.sinks_v)
            
        if ret_k is not None:
            k_parts.append(ret_k)
            v_parts.append(ret_v)
            
        k_parts.append(local_k)
        v_parts.append(local_v)
        
        full_k = torch.cat(k_parts, dim=-2)
        full_v = torch.cat(v_parts, dim=-2)
        
        return full_k, full_v

# Test Mock
if __name__ == "__main__":
    B, H, D = 1, 8, 64
    mem = InfLLMBlockMemory(block_size=10, r_tokens=2)
    
    # Fill memory
    for i in range(50):
        k = torch.randn(B, H, 1, D)
        v = torch.randn(B, H, 1, D)
        mem.add_kv(k, v)
        
    print(f"Blocks finalized: {len(mem.blocks)}")
    
    # Retrieval
    q = torch.randn(B, H, 1, D)
    ret_k, ret_v = mem.retrieve(q, top_k=2)
    print(f"Retrieved K shape: {ret_k.shape}") # Should be [1, 8, 20, 64]
    
    reattn = ReAttentionManager(mem, sinks_count=2)
    reattn.update_sinks(torch.randn(B, H, 10, D), torch.randn(B, H, 10, D))
    
    local_k = torch.randn(B, H, 5, D)
    local_v = torch.randn(B, H, 5, D)
    
    full_k, full_v = reattn.get_context(q, local_k, local_v)
    print(f"Full K shape: {full_k.shape}") # 2 (sinks) + 20 (retrieved) + 5 (local) = 27
