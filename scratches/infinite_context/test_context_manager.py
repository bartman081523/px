import unittest
from infinite_context import InfiniteContextManager

class MockTokenizer:
    def apply_chat_template(self, msgs, tokenize=False):
        return " ".join([m["content"] for m in msgs])
    def encode(self, text):
        return text.split() # 1 word = 1 token for mock

class TestInfiniteContextManager(unittest.TestCase):
    def test_token_truncation(self):
        manager = InfiniteContextManager(max_tokens=15) # Force truncation
        tokenizer = MockTokenizer()
        messages = [
            {"role": "system", "content": "System1 System2"},
            {"role": "user", "content": "W1 W2 W3 W4 W5"},
            {"role": "assistant", "content": "W1 W2 W3 W4 W5"},
            {"role": "user", "content": "W1 W2 W3 W4 W5"},
            {"role": "assistant", "content": "W1 W2 W3 W4 W5"},
        ]
        processed = manager.process_history(messages, tokenizer=tokenizer)
        self.assertTrue(len(processed) < len(messages) + 1)
        self.assertEqual(processed[0]["role"], "system")
        self.assertEqual(processed[1]["role"], "system")
        self.assertTrue("archived" in processed[1]["content"])

    def test_no_truncation(self):
        manager = InfiniteContextManager(max_history_messages=4)
        messages = [
            {"role": "system", "content": "You are a bot"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"}
        ]
        processed = manager.process_history(messages)
        self.assertEqual(len(processed), 3)
        self.assertEqual(processed[0]["role"], "system")
        self.assertEqual(processed[1]["role"], "user")

    def test_truncation(self):
        manager = InfiniteContextManager(max_history_messages=2)
        messages = [
            {"role": "system", "content": "You are a bot"},
            {"role": "user", "content": "Hi1"},
            {"role": "assistant", "content": "Hello1!"},
            {"role": "user", "content": "Hi2"},
            {"role": "assistant", "content": "Hello2!"},
            {"role": "user", "content": "Hi3"},
        ]
        # Should keep system, inject notice, and keep last 2 (Hello2! and Hi3)
        processed = manager.process_history(messages)
        self.assertEqual(len(processed), 4) # system + notice + 2 messages
        self.assertEqual(processed[0]["role"], "system")
        self.assertEqual(processed[1]["role"], "system") # The compression notice
        self.assertTrue("archived" in processed[1]["content"])
        self.assertEqual(processed[2]["role"], "assistant")
        self.assertEqual(processed[2]["content"], "Hello2!")
        self.assertEqual(processed[3]["role"], "user")
        self.assertEqual(processed[3]["content"], "Hi3")

if __name__ == '__main__':
    unittest.main()
