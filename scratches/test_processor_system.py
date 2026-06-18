from transformers import AutoProcessor

def test():
    processor = AutoProcessor.from_pretrained("google/gemma-3-270m-it")
    messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello"}]
    print("Testing system prompt...")
    try:
        res = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("OK:", res)
    except Exception as e:
        print("CRASHED:", type(e).__name__, ":", e)

if __name__ == "__main__":
    test()
