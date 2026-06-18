"""Probe: manual query-chunked attention (lossless, memory-bounded, kernel-independent).
Patches Gemma3Attention.forward on the BASE model. Tests:
 (a) correctness vs stock SDPA at T=4000 (outputs must match within fp tolerance)
 (b) no-OOM + speed at T=16000 (stock OOMs here)
Chunked attention is EXACT causal attention, just computed in q-tiles so peak score
memory is O(chunk * T_k) instead of O(T_q * T_k).
"""
import torch, time, os, types, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

CHUNK = 2048

def chunked_attention(q, k, v, scaling, sliding_window=None, chunk=CHUNK):
    """q,k,v: [B,H,T,D]. EXACT causal attention (+sliding_window if set), q-tiled. Lossless."""
    B, H, Tq, D = q.shape
    Tk = k.shape[-2]
    device, dtype = q.device, q.dtype
    out = torch.empty_like(q)
    kpos = torch.arange(Tk, device=device)
    for s in range(0, Tq, chunk):
        e = min(s + chunk, Tq)
        qc = q[:, :, s:e]
        scores = torch.matmul(qc, k.transpose(-1, -2)) * scaling
        qpos = torch.arange(s, e, device=device)
        mask = kpos[None, :] <= qpos[:, None]
        if sliding_window is not None:
            mask = mask & (kpos[None, :] >= (qpos[:, None] - sliding_window + 1))
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        out[:, :, s:e] = torch.matmul(attn, v)
    return out

def make_mem_eff_forward(orig_forward):
    def _fwd(self, hidden_states, position_embeddings=None, attention_mask=None,
             past_key_values=None, **kwargs):
        B, T, _ = hidden_states.shape
        hs = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hs).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hs).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
        q = self.q_norm(q); k = self.k_norm(k)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        out = chunked_attention(q, k, v, self.scaling, sliding_window=getattr(self, "sliding_window", None))
        out = out.transpose(1, 2).reshape(B, T, -1).contiguous()
        return self.o_proj(out), None
    return _fwd

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
tm = model.model
attn_modules = [m for m in tm.modules() if "Gemma3Attention" in type(m).__name__]
print("attn modules:", len(attn_modules))

def patch_on():
    for m in attn_modules:
        if not hasattr(m, "_orig_fwd"):
            m._orig_fwd = m.forward
        m.forward = types.MethodType(make_mem_eff_forward(m.forward), m)

def patch_off():
    for m in attn_modules:
        if hasattr(m, "_orig_fwd"):
            m.forward = m._orig_fwd

def gen(T):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    ids = torch.randint(10, tok.vocab_size, (1, T), device="cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
    return out[0].tolist(), torch.cuda.max_memory_allocated() / 1e9, time.time() - t0

# (a) correctness via LOGIT cosine at T=4000 (stock vs chunked both fit)
print("--- correctness T=4000 (logit cosine) ---")
ids4 = torch.randint(10, tok.vocab_size, (1, 4000), device="cuda")
with torch.no_grad():
    stock_logits = model(ids4).logits[0, -1].float()
patch_on()
with torch.no_grad():
    chunk_logits = model(ids4).logits[0, -1].float()
patch_off()
def cos(a, b):
    return (a @ b / (a.norm() * b.norm())).item()
print("cos(stock, chunk) =", round(cos(stock_logits, chunk_logits), 6))
print("argmax stock:", stock_logits.argmax().item(), "argmax chunk:", chunk_logits.argmax().item(),
      "max_abs_diff:", round((stock_logits - chunk_logits).abs().max().item(), 4))

# (b) 16k no-OOM + speed (chunked only; stock OOMs)
print("--- 16k chunked ---")
patch_on()
try:
    o, peak, dt = gen(16000)
    print(f"T=16k chunked: OK peak={peak:.2f}GB t={dt:.1f}s out={o[:8]}")
except Exception as e:
    print(f"T=16k chunked: ERR {type(e).__name__}: {str(e)[:120]}")
patch_off()