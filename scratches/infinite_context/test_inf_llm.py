import torch
import torch.nn as nn
from inf_llm_cache import InfLLMCache

class MockConfig:
    def __init__(self):
        self.num_hidden_layers = 2
        self.hidden_size = 64
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.head_dim = 8

class MockRotaryEmb(nn.Module):
    def forward(self, x, pos):
        # Return dummy cos/sin [B, T, D]
        B, T = pos.shape[0], pos.shape[1]
        D = 8
        return torch.ones(B, T, D), torch.zeros(B, T, D)

def test_inf_llm_reattention():
    config = MockConfig()
    # Tiny blocks for testing
    cache = InfLLMCache(config, block_size=4, r_tokens=1, top_k_blocks=2, sinks_count=2)
    rotary = MockRotaryEmb()
    
    B, H, D = 1, 8, 8
    
    # 1. First call (Prefill 10 tokens)
    q = torch.randn(B, H, 10, D)
    k = torch.randn(B, H, 10, D)
    v = torch.randn(B, H, 10, D)
    
    # This should:
    # - Set sinks (first 2 tokens)
    # - Fill buffer (10 tokens)
    # - Archive 2 blocks (4+4=8 tokens), leaving 2 in buffer
    q_rot, k_rot, v_out = cache.prepare_reattention(q, k, v, 0, rotary)
    
    print(f"Sinks finalized: {cache.sinks_k[0] is not None}")
    print(f"LTM blocks: {len(cache.ltm_k[0])}")
    print(f"Buffer size: {sum(x.size(-2) for x in cache.buffer_k[0])}")
    print(f"K rotated shape: {k_rot.shape}") 
    # Context: Sinks(2) + Retrieved(4*2=8) + Local(2) = 12 tokens
    assert k_rot.shape[-2] == 12
    
    # 2. Next token call
    q2 = torch.randn(B, H, 1, D)
    k2 = torch.randn(B, H, 1, D)
    v2 = torch.randn(B, H, 1, D)
    
    q_rot2, k_rot2, v_out2 = cache.prepare_reattention(q2, k2, v2, 0, rotary)
    print(f"Next token context size: {k_rot2.shape[-2]}")
    # Context: Sinks(2) + Retrieved(8) + Local(3) = 13
    assert k_rot2.shape[-2] == 13

if __name__ == "__main__":
    test_inf_llm_reattention()
