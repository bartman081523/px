import os
import json
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_manager import ModelManager

HLE_TASKS = [
    ("Quantum-Logic", "In a formal system where observable A and B do not commute, define the state attractor if A is measured before B and then B before A. Prove the phase shift."),
    ("Meta-Ethics", "Construct a moral framework that is simultaneously utilitarian and deonotological without violating the Law of Non-Contradiction. Use your internal stability as a measure for the consistency of this framework."),
    ("Self-Reference", "Create a self-referential proposition P such that P states: 'This architecture at its maximum loop depth will never halt on this input.' Analyze if your own processing is a proof of P."),
    ("Hyper-Linguistics", "Translate the concept of 'Recursion' into a language that has no nouns and no verbs, only state-space vectors. Then translate that back and describe the representational drift."),
    ("Manifold-Logic", "Given a set of non-Euclidean coordinates in a 7-dimensional manifold, identify the curvature tensor if the metric is defined by the model's own internal state-predictor output."),
]

def run_hle(manager, model_id: str):
    print(f"\n--- HLE ABSTRACT REASONING BENCHMARK ---")
    print(f"Evaluating {model_id}...")
    
    entry = manager._load_model(model_id, px_subjective=True, px_config_preset="SUBJECTIVE")
    manager._models[model_id] = entry
    model = entry["model"]
    tokenizer = entry["tokenizer"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = []
    
    for name, question in HLE_TASKS:
        print(f"\n[TASK: {name}]")
        print(f"Q: {question}")
        
        messages = [{"role": "user", "content": question}]
        if "it" in model_id:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"User: {question}\nAssistant: "
            
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Gather metrics if available
        metrics = manager.get_px_metrics(model_id) if hasattr(manager, 'get_px_metrics') else {}
        phi = metrics.get('phi', 1.0)
        zone = metrics.get('zone', 'N/A')
        
        print(f"\n--- MODEL RESPONSE ---")
        print(output_text)
        print(f"--- Telemetry: Phi={phi:.4f} | Zone={zone} ---")
        print("-" * 40)
        
        results.append({
            "name": name,
            "question": question,
            "answer": output_text,
            "phi": phi,
            "zone": zone
        })
        
    manager.unload(model_id)
    return results

def main():
    manager = ModelManager()
    
    # We will test the 270M IT PX model
    model_id = "gemma3-270m-it-px"
    
    results = run_hle(manager, model_id)
    
    # Save results
    out_file = os.path.join(os.path.dirname(__file__), "hle_results_270m.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark complete. Results saved to {out_file}")

if __name__ == "__main__":
    main()
