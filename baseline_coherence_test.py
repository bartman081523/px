
import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_patched_baseline_test(model_id="google/gemma-3-270m-it"):
    print(f"--- Patched (Commit 2fdc442) Coherence Test: {model_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    
    from all_space.px_patches.gemma3_270m_px_baseline.patch import apply_px_patch
    apply_px_patch(model, config_preset="SUBJECTIVE")
    
    test_prompts = [
        "What is the capital of France?",
        "Was ist der Sinn von Kunst?",
        "Solve: 15 + 27 * 2"
    ]
    
    results = []
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        duration = time.time() - start_time
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        
        tokens = tokenizer.encode(response)
        repetition_score = len(tokens) / len(set(tokens)) if tokens else 0
        
        is_coherent = True
        if repetition_score > 3.0:
            is_coherent = False
            print("!! Coherence Issue detected !!")
            
        results.append({
            "prompt": prompt,
            "response": response,
            "is_coherent": is_coherent,
            "tps": len(tokens) / duration
        })
        
    return results

if __name__ == "__main__":
    results = run_patched_baseline_test()
    all_ok = all(r["is_coherent"] for r in results)
    print(f"\nBaseline Stability: {'✅ OK' if all_ok else '❌ BROKEN'}")
