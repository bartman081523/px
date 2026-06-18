import gradio as gr
import os
import sys
from gradio import processing_utils
import asyncio

async def test_postprocess():
    # 2MB long string
    b64_str = "data:image/png;base64," + "i" * (2 * 1024 * 1024)
    data = [
        {"role": "assistant", "content": [{"type": "image", "image": {"path": b64_str}}]}
    ]
    print(f"Testing postprocess with length: {len(b64_str)}")
    
    with gr.Blocks() as demo:
        cb = gr.Chatbot()
        
    try:
        # Gradio 6.x internal postprocessing logic
        await processing_utils.async_move_files_to_cache(data, demo)
        print("SUCCESS: Postprocess finished without crash.")
    except OSError as e:
        print(f"CRASH DETECTED (OSError): {e}")
    except Exception as e:
        print(f"CRASH DETECTED (Other): {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_postprocess())
