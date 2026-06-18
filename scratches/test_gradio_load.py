import os
import sys

sys.path.append(os.getcwd())
from sessions import load_session
from gradio_tabs.chat_tab import _format_chatbot_content

def test():
    session_id = "3479b4f9"
    data = load_session(session_id)
    history = data.get("history", [])
    
    print(f"Loaded {len(history)} messages from session.")
    
    # Check raw loaded data first
    raw_img_found = False
    for m in history:
        if isinstance(m["content"], list):
            for item in m["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    raw_img_found = True
                    print(f"Raw loaded image path: {item['image']}")
    
    if not raw_img_found:
        print("Raw image not found in history!")
        return

    # Now test formatting for Gradio
    formatted_history = []
    for m in history:
        formatted_history.append({"role": m["role"], "content": _format_chatbot_content(m["content"])})
    
    print(f"Formatted {len(formatted_history)} messages for Gradio.")
    
    gradio_img_found = False
    for m in formatted_history:
        content = m["content"]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    gradio_img_found = True
                    # In _format_chatbot_content, images are nested as {"type": "image", "image": {"path": path}}
                    path = item["image"]["path"]
                    print(f"Gradio Image Path: {path}")
                    if path.startswith("data:image"):
                        print("CRITICAL: Still Base64!")
                    elif os.path.exists(path):
                        print("SUCCESS: Local file exists.")

if __name__ == "__main__":
    test()
