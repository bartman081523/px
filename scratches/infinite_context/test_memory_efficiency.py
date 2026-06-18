import torch
import json
import os
import sys

# Ensure we can import from all_space
sys.path.insert(0, os.getcwd())

from model_manager import ModelManager

import asyncio

async def test_large_session():
    # Limit visible devices just in case
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("Initializing ModelManager...")
    manager = ModelManager()
    
    print("Loading model gemma3-270m-it...")
    # This automatically patches it with ACTIVE_MANIFOLD preset
    model_entry = await manager.get_model("gemma3-270m-it", px_subjective=True, px_config_preset="ACTIVE_MANIFOLD")
    
    print("Loading large session '3479b4f9'...")
    session_file = "sessions/3479b4f9.json"
    if not os.path.exists(session_file):
        print(f"Session file {session_file} not found. Creating a mock large history.")
        # Create a mock long history (e.g. 500 messages)
        history = []
        for i in range(100):
            history.append({"role": "user", "content": [{"type": "text", "text": "Can you explain the theory of relativity?"}]})
            history.append({"role": "assistant", "content": [{"type": "text", "text": "The theory of relativity, developed by Albert Einstein, has two main parts: special relativity and general relativity. Special relativity states that the laws of physics are the same for all non-accelerating observers, and that the speed of light in a vacuum is independent of the motion of all observers."}]})
    else:
        with open(session_file, "r") as f:
            session_data = json.load(f)
            history = session_data.get("history", [])
            print(f"Loaded {len(history)} messages from session.")

    print(f"History length: {len(history)} messages.")

    try:
        from generators import generate_chat_completion_stream
        
        print("Starting generation...")
        import time
        start_t = time.time()
        
        # Generator
        gen = generate_chat_completion_stream(
            model_entry=model_entry,
            messages=history,
            max_tokens=100,
            temperature=0.7,
            top_p=0.95,
        )
        
        text = ""
        async for chunk in gen:
            if isinstance(chunk, dict):
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            else:
                content = chunk
            text += content

            
        dur = time.time() - start_t
        print(f"Generation successful! Length: {len(text)} characters. Time: {dur:.2f}s")
        print("Memory footprint is stable!")
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_large_session())
