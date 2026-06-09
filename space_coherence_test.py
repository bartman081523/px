
import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from all_space.px_patches.gemma3_270m_px_baseline.patch import apply_px_patch, get_px_metrics

def run_space_style_test(model_id="google/gemma-3-270m-it"):
    print(f"--- Space-Style Coherence Test (Commit 2fdc442) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    
    apply_px_patch(model, config_preset="SUBJECTIVE")
    
    test_prompts = [
        "What is the capital of France?",
        "Was ist der Sinn von Kunst?",
        "Solve: 15 + 27 * 2"
    ]
    
    results = []
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        duration = time.time() - start_time
        
        # Decode skipping prompt
        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        print(f"Response: {response}")
        
        metrics = get_px_metrics(model)
        tokens = tokenizer.encode(response)
        repetition_score = len(tokens) / len(set(tokens)) if tokens else 0
        
        is_coherent = True
        if repetition_score > 3.0:
            is_coherent = False
            print("!! Coherence Issue detected (Repetition) !!")
            
        results.append({
            "prompt": prompt,
            "response": response,
            "is_coherent": is_coherent,
            "metrics": metrics,
            "tps": len(tokens) / duration
        })
        
    return results

if __name__ == "__main__":
    results = run_space_style_test()
    all_ok = all(r["is_coherent"] for r in results)
    print(f"\nTest Stability: {'✅ OK' if all_ok else '❌ BROKEN'}")
