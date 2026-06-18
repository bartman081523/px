"""Diagnostic: PX at 14k with thought-injection DISABLED vs ENABLED.
If disabling fixes the degenerate output, the culprit is the RecursiveMemoryCache
thought-injection corrupting the full prefill KV (it blends into -T_curr: which at
prefill = the whole 14k). Fix would be to cap the injection to recent tokens only.
"""
import sys, os, json, time, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from transformers import AutoModelForCausalLM, AutoTokenizer
from px_lossless import patch as pxpatch
from px_lossless import patch as _px  # for RecursiveMemoryCache

MODEL = "google/gemma-3-1b-it"; SESSION = os.path.join(HERE, "..", "..", "sessions", "3479b4f9.json")

def _is_sliding(self, layer_idx):
    if self._layer_types and layer_idx < len(self._layer_types):
        return self._layer_types[layer_idx] == "sliding_attention"
    return False

def _update_noblend(self, key_states, value_states, layer_idx, cache_kwargs=None):
    """RecursiveMemoryCache.update WITHOUT the thought-injection blend (read_only logic kept)."""
    import torch
    if self._read_only:
        past_k, past_v = None, None
        if hasattr(self._real, "key_cache") and len(self._real.key_cache) > layer_idx:
            past_k, past_v = self._real.key_cache[layer_idx], self._real.value_cache[layer_idx]
        if past_k is None:
            past_k = torch.empty(0, device=key_states.device, dtype=key_states.dtype)
            past_v = torch.empty(0, device=value_states.device, dtype=value_states.dtype)
        past_seq, cur_seq = past_k.shape[-2] if past_k.numel() > 0 else 0, key_states.shape[-2]
        if past_seq >= self._expected_len: res_k, res_v = past_k, past_v
        elif past_seq == 0: res_k, res_v = key_states, value_states
        elif self._is_sliding(layer_idx) and cur_seq > 1: res_k, res_v = key_states, value_states
        else: res_k, res_v = torch.cat([past_k, key_states], dim=-2), torch.cat([past_v, value_states], dim=-2)
    else:
        res_k, res_v = self._real.update(key_states, value_states, layer_idx, cache_kwargs)
    return res_k, res_v

def run(disable_injection):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    if disable_injection:
        _px.RecursiveMemoryCache._is_sliding = _is_sliding
        _px.RecursiveMemoryCache.update = _update_noblend
    pxpatch.apply_px_patch(model, config_preset="ACTIVE_MANIFOLD")
    hist = json.load(open(SESSION))["history"]
    ids = tok(tok.apply_chat_template(hist, tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=48, do_sample=False, repetition_penalty=1.15,
                             no_repeat_ngram_size=3, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
    del model; torch.cuda.empty_cache()
    return txt

print("ENABLED (current PX):"); print(repr(run(False)[:300]))
print("-"*60)
print("DISABLED injection:"); print(repr(run(True)[:300]))