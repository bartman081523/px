"""Testet den Retry-Loop: 2x 503, dann 200. Erwartet: 200 am Client.

Cold-Start via subprocess, weil das Shim-Modul Env-Vars zur Import-Zeit
liest und nach Reload der socket-Wrapper den globalen State doppelt
initialisieren wuerde.
"""
import http.server
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request

MOCK_PORT = 9879
SHIM_PORT = 9880
TEST_DIR = "/run/media/julian/ML4/claude-model-einstellen"

call_count = {"n": 0}
oai_resp_data = {
    "id": "chatcmpl-retry", "model": "minimaxai/minimax-m3",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Antwort nach Retry."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
}


class FlakyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass
    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        call_count["n"] += 1
        if call_count["n"] <= 2:
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"upstream busy")
            return
        data = json.dumps(oai_resp_data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Eigenen Mock-Port + Shim-Port waehlen
mock_port = find_free_port()
shim_port = find_free_port()

mock = http.server.HTTPServer(("127.0.0.1", mock_port), FlakyHandler)
threading.Thread(target=mock.serve_forever, daemon=True).start()

env = {
    **__import__("os").environ,
    "SHIM_PORT": str(shim_port),
    "SHIM_UPSTREAM": f"http://127.0.0.1:{mock_port}",
    "SHIM_API_KEY": "test",
    "SHIM_LOG_LEVEL": "warning",
}
proc = subprocess.Popen(
    [sys.executable, f"{TEST_DIR}/tools/anthropic_openai_shim.py"],
    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

# Auf Shim-Start warten
for _ in range(50):
    try:
        with socket.create_connection(("127.0.0.1", shim_port), timeout=0.1):
            break
    except OSError:
        time.sleep(0.1)
else:
    proc.terminate()
    raise SystemExit("shim did not start in 5s")

try:
    t0 = time.time()
    req = urllib.request.Request(
        f"http://127.0.0.1:{shim_port}/v1/messages",
        data=json.dumps({"model": "minimaxai/minimax-m3", "max_tokens": 16,
                         "messages": [{"role": "user", "content": "hi"}]}).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": "test",
                 "anthropic-version": "2023-06-01"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        final_status = resp.status
    elapsed = time.time() - t0

    assert call_count["n"] == 3, f"erwartet 3 Upstream-Calls, sah {call_count['n']}"
    assert final_status == 200
    assert body["content"][0]["text"] == "Antwort nach Retry."
    assert 1.2 < elapsed < 2.5, f"Backoff-Timing auffaellig: {elapsed:.2f}s"
    print(f"[OK] Retry-Loop: 2x 503 + Backoff 0.5+1.0s + 200 in {elapsed:.2f}s")
finally:
    proc.terminate()
    proc.wait(timeout=2)
    mock.shutdown()
