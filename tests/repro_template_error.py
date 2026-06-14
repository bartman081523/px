from transformers import AutoTokenizer
import jinja2

# Simulate a text-only template (like Gemma)
template = "{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n' }}{% endfor %}"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it") # Should have a similar template

messages_str = [{"role": "user", "content": "hello"}]
messages_list = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

print("--- Testing with string content ---")
try:
    res = tokenizer.apply_chat_template(messages_str, tokenize=False)
    print("Success")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Testing with list content ---")
try:
    res = tokenizer.apply_chat_template(messages_list, tokenize=False)
    print("Success")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Manual Jinja test to confirm the exact error message
print("\n--- Manual Jinja test ---")
try:
    env = jinja2.Environment()
    t = env.from_string(template)
    t.render(messages=messages_list)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
