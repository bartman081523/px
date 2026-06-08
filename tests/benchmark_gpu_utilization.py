import torch
import time
import asyncio
from model_manager import ModelManager
from transformers import AutoTokenizer

async def run_benchmark():
    manager = ModelManager()
    model_id = "gemma3-270m-it"
    
    print("Loading model...")
    model_entry = await manager.get_model(model_id, px_subjective=True)
    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]
    
    prompt = "Explain the concept of recursion in computer science using an analogy of nested boxes."
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    print("\nWarmup (1 token)...")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    
    print("\nBenchmarking generation (50 tokens)...")
    start_time = time.time()
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=50)
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    tokens_generated = 50
    tps = tokens_generated / duration
    
    metrics = manager.get_px_metrics(model_id)
    steps = metrics.get("steps", 0)
    
    print(f"\n--- Benchmark Results ---")
    print(f"Time: {duration:.2f} s")
    print(f"Tokens/sec: {tps:.2f}")
    print(f"PX Recursion Steps per token (last token): {steps}")
    print(f"Average time per token: {(duration/tokens_generated)*1000:.2f} ms")
    print("-------------------------\n")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
