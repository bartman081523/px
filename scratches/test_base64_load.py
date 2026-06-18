import os
import sys
import json
import base64
from PIL import Image
import io

# Add current dir to sys.path
sys.path.append(os.getcwd())

from sessions import load_session, save_session

def test_load_base64_session():
    # Session ID from previous listing
    session_id = "3479b4f9"
    print(f"Testing load for session: {session_id}")
    
    try:
        data = load_session(session_id)
        history = data.get("history", [])
        print(f"Successfully loaded history with {len(history)} messages.")
        
        # Look for image paths in history
        found_images = 0
        for msg in history:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img_val = item.get("image")
                        print(f"Found image value: {img_val[:50]}...")
                        if os.path.exists(img_val):
                            print(f"VERIFIED: File exists at {img_val}")
                            found_images += 1
                        else:
                            print(f"FAILURE: File DOES NOT exist at {img_val}")
                            
        if found_images > 0:
            print(f"SUCCESS: Decoded {found_images} images to temp files.")
        else:
            print("No images found to decode (or decoding failed).")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load_base64_session()
