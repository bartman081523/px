import os
import sys
import json
import base64

sys.path.append(os.getcwd())
from gradio_tabs.chat_tab import handle_import

def test_import_flow():
    # 1. Create a JSON with raw base64
    b64_str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    test_data = {
        "session_id": "import_test_b64",
        "history": [
            {"role": "user", "content": [{"type": "text", "text": "Test"}, {"type": "image", "image": b64_str}]}
        ]
    }
    with open("test_import.json", "w") as f:
        json.dump(test_data, f)
        
    class MockFile:
        def __init__(self, name):
            self.name = name
            
    # 2. Call handle_import
    print("Testing handle_import with base64 JSON...")
    res = handle_import(MockFile("test_import.json"))
    
    # res is (new_id, history, dropdown_update, new_id)
    history = res[1]
    img_item = history[0]["content"][1]
    
    path = img_item["image"]["path"]
    print(f"Resulting Path: {path}")
    
    if path.startswith("data:image"):
        print("CRITICAL FAILURE: Still Base64!")
    elif os.path.exists(path):
        print("SUCCESS: Decoded to local file.")
    else:
        print(f"FAILURE: File missing at {path}")

if __name__ == "__main__":
    test_import_flow()
