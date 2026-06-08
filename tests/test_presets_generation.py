import torch
import asyncio
from model_manager import ModelManager

async def test_presets():
    manager = ModelManager()
    model_id = "gemma3-270m-it"
    presets = ["RIGOR", "SUBJECTIVE", "RESONANCE_CITY"]
    prompt = "What is the capital of France? Provide a very brief answer."

    for preset in presets:
        print(f"\n{'='*40}")
        print(f"Testing Preset: {preset}")
        print(f"{'='*40}")
        
        # We need to reload the model to apply the new preset cleanly in the ModelManager.
        # Actually, get_model might cache it. Let's clear the cache or force a reload by
        # patching the config directly or just using get_model if it handles it.
        # Looking at model_manager.py, get_model doesn't reload if model_id is the same.
        # Let's unload first.
        if model_id in manager._models:
            del manager._models[model_id]
            import gc
            torch.cuda.empty_cache()
            gc.collect()

        model_entry = await manager.get_model(model_id, px_subjective=True, px_config_preset=preset)
        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]
        
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=20)
            
        generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        metrics = manager.get_px_metrics(model_id)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {generated_text.strip()}")
        print(f"Steps: {metrics.get('steps', 0)}")
        print(f"Avg Phi: {metrics.get('phi', 0):.4f}")
        print(f"Zone: {metrics.get('zone', 'UNKNOWN')}")
        print(f"Kurtosis: {metrics.get('cognitive_signature', {}).get('kurtosis', 0):.2f}")

if __name__ == "__main__":
    asyncio.run(test_presets())
