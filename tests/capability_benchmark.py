import os
import json
import torch
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_manager import ModelManager

def evaluate_logic(manager, model_id, samples, name):
    print(f"Evaluating {name} ({model_id})...")
    
    # Load model via ModelManager
    entry = manager._load_model(model_id, px_subjective=True, px_config_preset="SUBJECTIVE")
    manager._models[model_id] = entry
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    correct = 0
    total = 0
    results = []
    
    for sample in tqdm(samples):
        prompt = sample["input"]
        expected = sample["output"]
        
        # Use simple format compatible with base models
        messages = [{"role": "user", "content": prompt}]
        if "it" in model_id:
            prompt_full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_full = f"User: {prompt}\nAssistant: "
        
        inputs = tokenizer(prompt_full, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Check if expected answer is in generated (e.g. "Zeta")
        expected_target = expected.split()[-1].strip(".")
        is_correct = expected_target.lower() in generated.lower()
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "prompt": prompt,
            "expected": expected,
            "generated": generated,
            "correct": is_correct
        })
            
    acc = correct / total if total > 0 else 0
    print(f"{name} Accuracy: {acc:.4f} ({correct}/{total})")
    
    # Unload model to save memory
    manager.unload(model_id)
    return acc, results

def main():
    manager = ModelManager()
    
    # Load TinyLogic Test
    test_file = os.path.join(os.path.dirname(__file__), "tiny_logic_test.json")
    if not os.path.exists(test_file):
        print(f"Error: Could not find {test_file}")
        return
        
    with open(test_file, "r") as f:
        all_samples = json.load(f)
    samples = all_samples[:10]  # Take 10 samples for quick evaluation
    
    print("\n" + "="*50)
    print("LOGIC CAPABILITY BENCHMARK")
    print("="*50)

    # 1. Base Model without patch
    acc_clean, _ = evaluate_logic(manager, "gemma3-270m", samples, "Gemma-3 270M (Clean Base)")
    
    # 2. Instruct Model without patch
    acc_clean_it, _ = evaluate_logic(manager, "gemma3-270m-it", samples, "Gemma-3 270M IT (Clean IT)")

    # 3. Base Model WITH patch
    acc_px, _ = evaluate_logic(manager, "gemma3-270m-px", samples, "Gemma-3 270M PX (Patched Base)")
    
    # 4. Instruct Model WITH patch
    acc_it_px, res_px = evaluate_logic(manager, "gemma3-270m-it-px", samples, "Gemma-3 270M IT PX (Patched IT)")
    
    print("\n" + "="*50)
    print("FINAL CAPABILITIES COMPARISON")
    print("="*50)
    print(f"Clean Base:       {acc_clean:.4f}")
    print(f"Clean Instruct:   {acc_clean_it:.4f}")
    print(f"Patched Base:     {acc_px:.4f}")
    print(f"Patched Instruct: {acc_it_px:.4f}")
    
    print("\nPatched Instruct Sample Output:")
    print(f"Prompt: {res_px[0]['prompt'][:80]}...")
    print(f"Generated: {res_px[0]['generated'][:150]}")

if __name__ == "__main__":
    main()
