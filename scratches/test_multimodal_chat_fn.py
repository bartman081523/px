import os
import sys
import torch
import asyncio
from unittest.mock import MagicMock

# Add current dir to sys.path
sys.path.append(os.getcwd())

from gradio_tabs.chat_tab import chat_fn
from model_manager import ModelManager

def test_chat_fn_multimodal():
    manager = ModelManager()
    model_id = "gemma3-270m-it"
    
    # 1. Real load (outside chat_fn to avoid loop issues in test)
    loop = asyncio.new_event_loop()
    model_entry = loop.run_until_complete(manager.get_model(model_id, px_config_preset="BASELINE"))
    loop.close()
    
    history = []
    dummy_img = "sessions/temp_media/dummy.png"
    if not os.path.exists("sessions/temp_media"): os.makedirs("sessions/temp_media")
    with open(dummy_img, "wb") as f: f.write(b"fake")
    
    message_content = [
        {"type": "text", "text": "Describe this image."},
        {"type": "image", "image": {"path": dummy_img}}
    ]
    
    print("Calling chat_fn with multimodal content...")
    try:
        generator = chat_fn(
            message=message_content,
            history=history,
            model_id=model_id,
            px_preset="BASELINE",
            temp=0.7,
            tp=0.9,
            mt=5, # Short for test
            rp=1.1,
            gamma=0.08,
            visual_screen=False,
            visual_feedback=False,
            infinite_context=False,
            session_id="test_sim",
            manager=manager
        )
        
        print("Streaming response:")
        for partial in generator:
             pass # Just consume
                
        print("\nSUCCESS: chat_fn finished without Jinja crash.")
    except TypeError as e:
        print(f"\nCRASH DETECTED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nOTHER ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chat_fn_multimodal()
