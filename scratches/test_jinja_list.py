from transformers import AutoTokenizer
import os
import sys

# Simulation of the gemma3-270m-it template
chat_template_manual = "{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}\n{% else %}Assistant: {{ message['content'] }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"

def test_jinja_crash():
    # We use a dummy tokenizer but set the template
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    tokenizer.chat_template = chat_template_manual
    
    # Case A: Content is a string (SHOULD WORK)
    messages_ok = [{"role": "user", "content": "Hello"}]
    print("Testing string content...")
    try:
        res = tokenizer.apply_chat_template(messages_ok, tokenize=False)
        print("Result OK:", res.strip())
    except Exception as e:
        print("FAILED string content:", e)

    # Case B: Content is a list (multimodal style)
    messages_fail = [{"role": "user", "content": [{"type": "text", "text": "Describe this"}, {"type": "image", "image": "path"}]}]
    print("\nTesting list content (multimodal style)...")
    try:
        res = tokenizer.apply_chat_template(messages_fail, tokenize=False)
        print("Result OK:", res.strip())
    except Exception as e:
        print("CAUGHT EXPECTED CRASH:", type(e).__name__, ":", e)

if __name__ == "__main__":
    test_jinja_crash()
