"""End-to-End-Test fuer Phase 2 Tool-Use-Mapping.

Startet einen Mock-OpenAI-Upstream (mit tool_calls), startet den Shim gegen
diesen Mock, und schickt einen Anthropic-Request mit tools[] + tool_use in
der History. Prueft: (a) Request-Mapping am Mock korrekt, (b) Response wird
in Anthropic-Schema gemappt inkl. tool_use-Block.
"""
import http.server
import json
import socketserver
import threading
import time
import urllib.request

sys_path = "/run/media/julian/ML4/claude-model-einstellen/tools"
import sys
sys.path.insert(0, sys_path)

import anthropic_openai_shim as shim

# 1) Mock-OpenAI: nimmt /v1/chat/completions an, antwortet mit tool_call
MOCK_PORT = 9877
SHIM_PORT = 9878

received_body: dict = {}

class MockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass
    def do_POST(self):  # noqa
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        received_body.update(body)
        # Simuliere Tool-Call-Response
        resp = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": body.get("model", "test"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Muenchen"}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
        }
        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):  # noqa
        if self.path.startswith("/v1/models"):
            data = json.dumps({"data": [{"id": "minimaxai/minimax-m3"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()

# Mock-Server starten
mock_srv = http.server.HTTPServer(("127.0.0.1", MOCK_PORT), MockHandler)
mock_thread = threading.Thread(target=mock_srv.serve_forever, daemon=True)
mock_thread.start()

# Shim auf MOCK als Upstream starten
import os
os.environ["SHIM_PORT"] = str(SHIM_PORT)
os.environ["SHIM_UPSTREAM"] = f"http://127.0.0.1:{MOCK_PORT}"
os.environ["SHIM_API_KEY"] = "test-key"
os.environ["SHIM_LOG_LEVEL"] = "warning"
# Re-import mit neuem Env
import importlib
importlib.reload(shim)

shim_srv = shim.ThreadingHTTPServer(("127.0.0.1", SHIM_PORT), shim.ShimHandler)
shim_thread = threading.Thread(target=shim_srv.serve_forever, daemon=True)
shim_thread.start()
time.sleep(0.3)

# 2) Anthropic-Request senden
ant_body = {
    "model": "minimaxai/minimax-m3",
    "max_tokens": 200,
    "tools": [{
        "name": "get_weather",
        "description": "Get the current weather",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }],
    "messages": [{
        "role": "user",
        "content": [{"type": "text", "text": "Wie ist das Wetter in Muenchen?"}],
    }],
}
req = urllib.request.Request(
    f"http://127.0.0.1:{SHIM_PORT}/v1/messages",
    data=json.dumps(ant_body).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as resp:
    ant_resp = json.loads(resp.read().decode("utf-8"))

# 3) Assertions
print("=== Anthropic Response ===")
print(json.dumps(ant_resp, indent=2, ensure_ascii=False))

# Mock hat gesehen:
print("\n=== Mock hat folgenden OpenAI-Request gesehen ===")
print(json.dumps(received_body, indent=2, ensure_ascii=False))

# Checks
assert received_body["model"] == "minimaxai/minimax-m3"
assert received_body["tools"][0]["type"] == "function"
assert received_body["tools"][0]["function"]["name"] == "get_weather"
assert received_body["messages"][-1]["content"] == "Wie ist das Wetter in Muenchen?"
assert ant_resp["stop_reason"] == "tool_use"
tool_use_blocks = [b for b in ant_resp["content"] if b["type"] == "tool_use"]
assert len(tool_use_blocks) == 1
assert tool_use_blocks[0]["name"] == "get_weather"
assert tool_use_blocks[0]["input"] == {"city": "Muenchen"}
print("\n[OK] Alle Phase-2-Assertions bestanden.")

mock_srv.shutdown()
shim_srv.shutdown()
