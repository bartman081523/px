"""Isolate output-quality: (A) BASE model + chunked attention (no PX) on full 3479b4f9,
(B) lossless PX patch on a SHORT prompt (PX functionality sanity vs 294191f).
"""
import sys, os, json, time, types, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

MODEL = "google/gemma-3-1b-it"
SESSION = os.path.join(HERE, "..", "..", "sessions", "3479b4f9.json")
CHUNK = 2048

def chunked_attention(q, k, v, scaling, sliding_window=None, chunk=CHUNK):
    B, H, Tq, D = q.shape; Tk = k.shape[-2]; device, dtype = q.device, q.dtype
    out = torch.empty_like(q); kpos = torch.arange(Tk, device=device)
    for s in range(0, Tq, chunk):
        e = min(s+chunk, Tq); qc = q[:,:,s:e]
        scores = torch.matmul(qc, k.transpose(-1,-2))*scaling
        qpos = torch.arange(s, e, device=device)
        mask = kpos[None,:] <= qpos[:,None]
        if sliding_window is not None: mask = mask & (kpos[None,:] >= (qpos[:,None]-sliding_window+1))
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        out[:,:,s:e] = torch.matmul(torch.softmax(scores, dim=-1).to(dtype=v.dtype), v)
    return out

def patch_chunked(model):
    attns = [m for m in model.model.modules() if "Gemma3Attention" in type(m).__name__]
    def fwd(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
        ish = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_proj(hidden_states).view(ish).transpose(1,2)
        k = self.k_proj(hidden_states).view(ish).transpose(1,2)
        v = self.v_proj(hidden_states).view(ish).transpose(1,2)
        q = self.q_norm(q); k = self.k_norm(k)
        cos, sin = position_embeddings; q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None: k, v = past_key_values.update(k, v, self.layer_idx)
        Tq, Tk = q.shape[-2], k.shape[-2]; sw = getattr(self,"sliding_window",None)
        if Tq==1 or (Tq<=4096 and Tk<=4096):
            from transformers.models.gemma3.modeling_gemma3 import ALL_ATTENTION_FUNCTIONS, eager_attention_forward
            ai = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
            o,_ = ai(self,q,k,v,attention_mask,dropout=0.0,scaling=self.scaling,sliding_window=sw)
        else:
            o = chunked_attention(q,k,v,self.scaling,sliding_window=sw)
        o = o.transpose(1,2).reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.o_proj(o), None
    for m in attns: m.forward = types.MethodType(fwd, m)

tok = AutoTokenizer.from_pretrained(MODEL)

# (A) BASE + chunked, full 3479b4f9
print("="*70); print("(A) BASE + chunked attention, FULL 3479b4f9")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
patch_chunked(model)
hist = json.load(open(SESSION))["history"]
prompt = tok.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
ids = tok(prompt, return_tensors="pt").to("cuda")
print("tokens:", ids["input_ids"].shape[1])
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache(); t0=time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=64, do_sample=False, repetition_penalty=1.15, no_repeat_ngram_size=3, pad_token_id=tok.eos_token_id)
print(f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB t={time.time()-t0:.1f}s")
print("BASE OUT:", repr(tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)[:300]))
del model; torch.cuda.empty_cache()

# (B) lossless PX patch, SHORT prompt
print("="*70); print("(B) lossless PX patch, SHORT prompt")
from px_lossless import patch as pxpatch
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
pxpatch.apply_px_patch(model, config_preset="ACTIVE_MANIFOLD")
short = [{"role":"user","content":"Erkläre in zwei Sätzen, was ein Schwarzes Loch ist."}]
ids = tok(tok.apply_chat_template(short, tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
print("tokens:", ids["input_ids"].shape[1])
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=80, do_sample=False, repetition_penalty=1.15, no_repeat_ngram_size=3, pad_token_id=tok.eos_token_id)
print("PX SHORT OUT:", repr(tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)[:400]))