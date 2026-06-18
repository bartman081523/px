import torch
import sys
sys.path.append(".")
from infinite_context import InfLLMCache

class Config:
    num_hidden_layers = 1
cache = InfLLMCache(Config(), block_size=128, r_tokens=8, top_k_blocks=16, sinks_count=4)

q = torch.randn(1, 1, 14523, 64)
k = torch.randn(1, 1, 14523, 64)
v = torch.randn(1, 1, 14523, 64)

def rotary(x, pos):
    return torch.ones_like(x), torch.zeros_like(x)

q_rot, k_rot, v_rot = cache.prepare_reattention(q, k, v, 0, rotary)
print("Q out:", q_rot.shape)
print("K out:", k_rot.shape)
