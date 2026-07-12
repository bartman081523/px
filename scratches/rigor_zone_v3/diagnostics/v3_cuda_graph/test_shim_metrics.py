"""Misst Latenz-Logging: Simuliert 800ms-Antwort und prueft, was im Log landet."""
import http.server
import json
import threading
import time
import urllib.request
import sys
import logging

sys.path.insert(0, "/run/media/julian/ML4/claude-model-einstellen/tools")
import anthropic_openai_shim as shim
import importlib

# Capture log-Output
log_capture = []
class CaptureHandler(logging.Handler):
    def emit(self, record):
        log_capture.append(f"{record.levelname}|{record.getMessage()}")

MOCK_PORT = 9891
SHIM_PORT = 9892

class SlowHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        time.sleep(0.8)  # Simuliere 800ms Latenz
        resp = {
            "id": "x", "model": "minimaxai/minimax-m3",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "fertig."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        data = b'{"data":[{"id":"minimaxai/minimax-m3"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

import os
os.environ["SHIM_PORT"] = str(SHIM_PORT)
os.environ["SHIM_UPSTREAM"] = f"http://127.0.0.1:{MOCK_PORT}"
os.environ["SHIM_API_KEY"] = "test"
os.environ["SHIM_LOG_LEVEL"] = "info"
importlib.reload(shim)
shim.log.addHandler(CaptureHandler())

mock = http.server.HTTPServer(("127.0.0.1", MOCK_PORT), SlowHandler)
threading.Thread(target=mock.serve_forever, daemon=True).start()
shim_srv = shim.ThreadingHTTPServer(("127.0.0.1", SHIM_PORT), shim.ShimHandler)
threading.Thread(target=shim_srv.serve_forever, daemon=True).start()
time.sleep(0.3)

req = urllib.request.Request(
    f"http://127.0.0.1:{SHIM_PORT}/v1/messages",
    data=json.dumps({"model": "minimaxai/minimax-m3", "max_tokens": 32,
                     "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8"),
    headers={"Content-Type": "application/json", "x-api-key": "test",
             "anthropic-version": "2023-06-01"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=10) as resp:
    body = json.loads(resp.read().decode("utf-8"))
    final_status = resp.status

mock.shutdown(); shim_srv.shutdown()

print(f"=== Captured {len(log_capture)} log lines ===")
for line in log_capture:
    print(f"  {line}")
print()
print("Response:", body["content"][0]["text"])
