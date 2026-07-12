"""Diagnose: Welcher PX-Mechanismus degeneriert 270m?

TDD-RED: 
- baseline soll "OK" sein (kein Looping).
- active_manifold soll auch "OK" sein nach Fix.

Heuristik schärfer: phrase "X is the number" ≥ 2 mal = LOOPY.
"""
import sys, os, json
sys.path.insert(0, '/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages')
sys.path.insert(0, '/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/rigor_zone_v3')

os.environ['HF_HUB_OFFLINE'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch

# Test-Task (HLE Math)
TASK_PROMPT = """Let $G_2$ be the classifying space of the Lie group $G_2$. What is the reduced 12-th dimensional Spin bordism of $G_2$? Please answer with a single mathematical expression."""
GT = "Z+Z+Z+Z+Z"  # known HLE answer
MAX_NEW = 60

def load_model(preset, **kwargs):
    tok = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", dtype=torch.bfloat16).to("cuda").eval()
    if preset != "BASELINE":
        # Patch das INNERE text_model
        apply_px_patch(model.model, config_preset=preset, **kwargs)
    return tok, model

def generate(tok, model, prompt, max_new=MAX_NEW):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    new_tokens = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)

def analyze(text, gt=GT):
    """Schärfere Heuristik: leer/loopy/contains-gt."""
    if not text.strip():
        return {"degenerated": True, "reason": "EMPTY", "len": 0, "contains_gt": False}
    
    # Loopy: jede 4-word phrase ≥ 3 mal
    words = text.split()
    is_loopy = False
    reason = "OK"
    for plen in [3, 4, 5]:
        for i in range(len(words) - plen*3):
            phrase = " ".join(words[i:i+plen])
            n_repeats = sum(1 for j in range(i+plen, len(words)-plen+1, plen) if " ".join(words[j:j+plen]) == phrase)
            if n_repeats >= 3:
                is_loopy = True
                reason = f"LOOPY_p{plen}"
                break
        if is_loopy: break
    
    # Numeric garbage: 0, 0, 0, 0, 0, ...
    if ", 0," in text and text.count(", 0,") > 3:
        is_loopy = True
        reason = "NUMERIC_GARBAGE"
    if "0, 0, 0" in text or "1, 1, 1" in text:
        is_loopy = True
        reason = "NUMERIC_GARBAGE"
    
    return {
        "degenerated": is_loopy,
        "reason": reason,
        "len": len(text),
        "contains_gt": gt.lower() in text.lower(),
    }

# === Haupttest ===
print("=" * 70)
print("PX-ENGINE DIAGNOSE — 1 Math-Task (HLE 669402)")
print("=" * 70)
print(f"Task: {TASK_PROMPT[:100]!r}")
print(f"GT: {GT!r}")
print()

tests = [
    ("A_baseline", "BASELINE", {}),
    ("B_active_manifold_full", "ACTIVE_MANIFOLD", {}),
]

results = {}
for name, preset, kwargs in tests:
    print(f"--- Test: {name} ---")
    try:
        tok, model = load_model(preset, **kwargs)
        text = generate(tok, model, TASK_PROMPT)
        r = analyze(text)
        results[name] = {"text": text[:300], **r}
        print(f"  degen={r['degenerated']:5}  reason={r['reason']:15}  len={r['len']:3}  contains_gt={r['contains_gt']}")
        print(f"  output: {text[:200]!r}")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        traceback.print_exc()
        results[name] = {"error": str(e), "degenerated": True, "reason": "ERROR"}
        print(f"  ERROR: {e}")

# Summary
print()
print("=" * 70)
print("DIAGNOSE")
print("=" * 70)
for name, r in results.items():
    if "error" in r:
        print(f"  {name:30s}: ERROR")
    else:
        status = "DEGEN" if r["degenerated"] else "OK"
        print(f"  {name:30s}: {status:5s}  ({r['reason']:15s})  contains_gt={r['contains_gt']}")
