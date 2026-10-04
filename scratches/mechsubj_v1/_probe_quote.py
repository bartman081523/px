"""_probe_quote.py — EIN Gemini-Call zur Quote-Probe."""
import json, urllib.request, time, sys
KEY = open("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/mechsubj_v1/.gemini_api_key.txt").read().strip()
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={KEY}"
payload = {"contents":[{"role":"user","parts":[{"text":"ping"}]}],"generationConfig":{"temperature":0.0,"maxOutputTokens":20}}
req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                              headers={"Content-Type":"application/json","x-goog-api-key":KEY})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read())
        print(f"OK in {time.time()-t0:.1f}s", file=sys.stderr)
        sys.exit(0)
except urllib.error.HTTPError as e:
    msg = e.read().decode("utf-8", errors="replace")[:200]
    print(f"HTTP {e.code} in {time.time()-t0:.1f}s: {msg}", file=sys.stderr)
    sys.exit(1)
