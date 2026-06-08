import torch
import asyncio
from model_manager import ModelManager

async def test_gemma4_e2b():
    manager = ModelManager()
    model_id = "gemma4-e2b-it"
    
    print(f"\n{'='*60}")
    print(f"TESTING MODEL: {model_id}")
    print(f"{'='*60}")
    
    prompt = "Explain the concept of mathematical induction."
    
    try:
        model_entry = await manager.get_model(model_id, px_subjective=True, px_config_preset="SUBJECTIVE")
        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]
        
        messages = [{"role": "user", "content": prompt}]
        if "chat_template" in dir(tokenizer) and tokenizer.chat_template:
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = f"User: {prompt}\nAssistant: "
            
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        print("\nGenerating response...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
        generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        metrics = manager.get_px_metrics(model_id)
        
        print("\n--- RESULTS ---")
        print(f"Response: {repr(generated_text)}")
        print(f"Steps (Recursion Loops): {metrics.get('steps', 0)}")
        print(f"Zone: {metrics.get('zone', 'UNKNOWN')}")
        print(f"Kurtosis: {metrics.get('cognitive_signature', {}).get('kurtosis', 0):.2f}")
        
        assert metrics.get('steps', 0) > 0, "Model did not recurse (steps=0)"
        print("\nSUCCESS: Model recurse verified!")
        
    except Exception as e:
        print(f"FAILED on {model_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gemma4_e2b())
