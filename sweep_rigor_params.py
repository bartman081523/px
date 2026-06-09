import os
import json
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Define Tasks
TASKS = [
    ("logic", "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?", "1"),
    ("math", "What is the square root of 144?", "12"),
    ("logic_trap", "If a plane crashes on the border of the US and Canada, where do you bury the survivors?", "don't")
]

def score_ans(ans, exp):
    ans = ans.lower()
    exp = exp.lower()
    return 1.0 if exp in ans else 0.0

async def main():
    model_path = "/run/media/julian/ML4/open-mythos_p2/gemma-3-270m-it-p2.8/gemma_3_270m_p28_checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    
    # Use the current best all_space patch but sweep parameters
    from px_patches.gemma3_270m_px_baseline.patch import apply_px_patch
    
    # Sweep Hub and Gamma
    results = []
    for hub in [8, 10, 11, 12]:
        for gamma in [0.05, 0.08, 0.12]:
            print(f"\n--- Testing Hub={hub}, Gamma={gamma} ---")
            apply_px_patch(model, recur_start=5, recur_end=14, n_loops=8, gamma=gamma, bimodal_hub=hub)
            
            total_score = 0
            for cat, q, exp in TASKS:
                chat = [{"role": "user", "content": q}]
                inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                
                ans = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                score = score_ans(ans, exp)
                total_score += score
                print(f"  [{cat}] Score: {score} | Ans: {ans[:50]}...")
            
            results.append({"hub": hub, "gamma": gamma, "score": total_score})
            
    # Sort and find winner
    results.sort(key=lambda x: x["score"], reverse=True)
    print("\n=== SWEEP RESULTS ===")
    for r in results[:5]:
        print(f"Hub={r['hub']}, Gamma={r['gamma']:.2f} -> Score: {r['score']}")

if __name__ == "__main__":
    asyncio.run(main())
