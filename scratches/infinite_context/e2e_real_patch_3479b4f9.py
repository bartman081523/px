"""FINAL e2e through the REAL production patch path (not the scratch).
Imports px_patches.gemma3_270m_px_baseline.patch directly, applies ACTIVE_MANIFOLD
to gemma3-1b-it, runs the FULL 3479b4f9 session (no truncation), measures VRAM/time/output.
This is the deliverable validation.
"""
import sys, os, json, time, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
ROOT = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, ROOT)                       # so `px_patches.*` package imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from px_patches.gemma3_270m_px_baseline import patch as pxpatch   # REAL module

MODEL = "google/gemma-3-1b-it"
SESSION = os.path.join(ROOT, "sessions", "3479b4f9.json")

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    pxpatch.apply_px_patch(model, config_preset="ACTIVE_MANIFOLD")
    tm = model.model
    print("REAL patch applied. mem_eff present:", any(hasattr(m, "_px_mem_eff_orig") for m in tm.modules()))

    with open(SESSION) as f:
        hist = json.load(f)["history"]
    prompt = tok.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to("cuda")
    T = ids["input_ids"].shape[1]
    print(f"prompt tokens: {T}  (lossless: full context, no truncation)")

    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=64, do_sample=False,
                repetition_penalty=getattr(tm, "_px_repetition_penalty", 1.15),
                no_repeat_ngram_size=getattr(tm, "_px_no_repeat_ngram_size", 3),
                pad_token_id=tok.eos_token_id,
            )
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"RESULT: OK  peak={peak:.2f}GB  time={dt:.1f}s  new_tokens={out.shape[1]-T}")
        print(f"FREE_AFTER: {torch.cuda.mem_get_info()[0]/1e9:.2f}GB free")
        print("OUTPUT:", repr(text[:400]))
    except torch.OutOfMemoryError as e:
        print(f"RESULT: OOM  {str(e)[:160]}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"RESULT: ERR {type(e).__name__}: {str(e)[:200]}")

if __name__ == "__main__":
    main()