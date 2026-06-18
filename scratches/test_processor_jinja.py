from transformers import AutoProcessor
import os
import sys

def test_processor_jinja():
    # Use 270M processor
    processor = AutoProcessor.from_pretrained("google/gemma-3-270m-it")
    
    # Simulate my stringified messages
    messages = [
        {"role": "user", "content": "Hello <image>"}
    ]
    
    print("Testing processor.apply_chat_template with stringified content...")
    try:
        res = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("Result OK:", res)
    except Exception as e:
        print("CRASHED:", type(e).__name__, ":", e)

    # Simulate LIST content (what I'm trying to avoid)
    messages_list = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "image", "image": "path"}]}
    ]
    print("\nTesting processor.apply_chat_template with LIST content...")
    try:
        # Some processors handles this automatically!
        res = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
        print("Result OK (Automated handling):", res)
    except Exception as e:
        print("CRASHED with list:", type(e).__name__, ":", e)

if __name__ == "__main__":
    test_processor_jinja()
