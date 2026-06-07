
import unittest
import torch
import torch.nn as nn
import sys
import os

# Ensure we can import from all_space
sys.path.insert(0, os.getcwd())

class MockLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, h, attention_mask=None, past_key_values=None, **kwargs):
        if past_key_values is not None:
            # Simulate DynamicCache update
            k = torch.randn(h.shape[0], 1, h.shape[1], 64)
            v = torch.randn(h.shape[0], 1, h.shape[1], 64)
            past_key_values.update(k, v, self.index)
        return h, past_key_values

class MockCache:
    def __init__(self):
        self.key_cache = []
    def update(self, k, v, i):
        while len(self.key_cache) <= i: self.key_cache.append(None)
        if self.key_cache[i] is None: self.key_cache[i] = k
        else: self.key_cache[i] = torch.cat([self.key_cache[i], k], dim=-2)
        return self.key_cache[i], None
    def get_seq_length(self, i=0):
        return self.key_cache[i].shape[-2] if i < len(self.key_cache) and self.key_cache[i] is not None else 0

class TestOverlapBug(unittest.TestCase):
    def test_prelude_reasoning_overlap(self):
        """Test if overlapping prelude and reasoning pass-through double the cache."""
        cache = MockCache()
        layers = [MockLayer(i) for i in range(10)]
        
        prelude_end = 5
        dynamic_start = 3 # OVERLAP with 3, 4
        dynamic_end = 8
        
        T = 10
        h = torch.randn(1, T, 64)
        
        # Simulate Prelude
        updated_layers = set()
        for i in range(prelude_end):
            updated_layers.add(i)
            h, _ = layers[i](h, past_key_values=cache)
            
        self.assertEqual(cache.get_seq_length(3), T)
        
        # Simulate Reasoning Pass-Through (The Buggy Part)
        # It currently uses raw cache without checking updated_layers
        for i in range(dynamic_start, dynamic_end):
            # updated_layers.add(i) # This is done in the code, but doesn't protect the raw cache
            h, _ = layers[i](h, past_key_values=cache)
            
        # Layers 3 and 4 should still have length T, but they will have 2*T!
        len_3 = cache.get_seq_length(3)
        print(f"Layer 3 cache length: {len_3}")
        
        # This will trigger the RuntimeError in SDPA because mask expects T, but cache has 2*T
        self.assertLessEqual(len_3, T, "Cache size exceeded! Overlap bug detected.")

if __name__ == "__main__":
    unittest.main()
