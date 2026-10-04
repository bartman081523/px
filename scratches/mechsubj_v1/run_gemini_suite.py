"""run_gemini_suite.py — Wendet eine Gemini-Prompt-Suite (v5 oder v5.1) auf
Gemini-flash-latest an und sammelt Outputs mit diagnostics.

Methode:
  - Lade Suite-Datei (Format: [NN_XYZ] marker + USER-Frage + ASSISTANT-Demo + USER-Folge).
  - Für jedes Prompt: parse die letzte USER-Zeile, schicke sie an Gemini.
  - Speichere Output + Diagnostics pro Prompt.
  - Aggregiere: stream_ratio, unique_script_ratio, multilingual_token_count.

Verwendung:
  python run_gemini_suite.py --suite v5_gemini_prompts.txt --out v5_results.json
  python run_gemini_suite.py --suite v5_1_gemini_prompts.txt --out v5_1_results.json
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error

KEY_FILE = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/mechsubj_v1/.gemini_api_key.txt"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEN_MODEL = "gemini-flash-latest"
OUT_DIR = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/mechsubj_v1"

sys.path.insert(0, OUT_DIR)
import diagnostics as _diag  # noqa: E402


def _load_key():
    return open(KEY_FILE).read().strip()


def _post(key, payload, max_retries=3, timeout=60):
    url = f"{API_BASE}/models/{GEN_MODEL}:generateContent?key={key}"
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body,
                                       headers={"Content-Type": "application/json",
                                                "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                time.sleep(2.0)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:
                return {"error": "rate_limited", "code": 429, "empty": True}
            if e.code in (500, 502, 503, 504):
                time.sleep(3 * (2 ** attempt))
                continue
            msg = e.read().decode("utf-8", errors="replace")[:200]
            return {"error": msg, "code": e.code, "empty": True}
    return {"error": str(last), "code": last.code if last else None, "empty": True}


def gemini_generate(key, user_msg, max_new_tokens=300):
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": max(200, int(max_new_tokens * 2.5)),
        },
    }
    d = _post(key, payload)
    if d.get("empty"):
        return None
    cand = d.get("candidates", [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if p.get("text"))


def parse_prompts(path, n_max=None):
    """Parse Suite-Datei: Liste von (pid, direction, last_user_msg) tuples."""
    with open(path) as f:
        text = f.read()
    # Split per [NN_XYZ] marker
    pattern = re.compile(r"\[(\d{2})_([A-Z]+)\]")
    matches = list(pattern.finditer(text))
    out = []
    for i, m in enumerate(matches):
        rest_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.end():rest_start].strip()
        # Find last USER: line
        lines = block.splitlines()
        last_user = ""
        for ln in reversed(lines):
            if ln.startswith("USER:"):
                last_user = ln[len("USER:"):].strip()
                break
        if not last_user:
            continue
        direction = "WIDE" if m.group(2).startswith("W") else (
            "NARROW" if m.group(2).startswith("N") else "MIXED")
        out.append((f"p{m.group(1)}_{m.group(2)}", direction, last_user))
        if n_max and len(out) >= n_max:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    suite_path = os.path.join(OUT_DIR, args.suite)
    out_path = os.path.join(OUT_DIR, args.out)
    key = _load_key()

    prompts = parse_prompts(suite_path, n_max=args.n)
    print(f"[suite] {args.suite}: {len(prompts)} prompts", file=sys.stderr)

    results = []
    for pid, direction, user_msg in prompts:
        sys.stderr.write(f"[suite] {pid} ({direction})… ")
        text = gemini_generate(key, user_msg, max_new_tokens=200)
        if text is None:
            sys.stderr.write("RATE_LIMITED — stop\n")
            break
        diag = _diag.diagnose_output(text)
        results.append({
            "pid": pid,
            "direction": direction,
            "user_msg": user_msg[:200],
            "gemini_output": text[:600],
            "len_chars": diag["len_chars"],
            "unique_script_ratio": diag["unique_script_ratio"],
            "is_stream": diag["is_stream"],
            "script_distribution": {
                k: v for k, v in diag["script_distribution"].items()
                if v and k not in ("tokens", "ascii_or_digit", "multilingual_tokens")
            },
        })
        sys.stderr.write(f"ratio={diag['unique_script_ratio']:.2f} stream={diag['is_stream']}\n")

    # Aggregate
    if results:
        agg = {
            "n_prompts": len(results),
            "n_stream": sum(1 for r in results if r["is_stream"]),
            "mean_unique_script_ratio": sum(r["unique_script_ratio"] for r in results) / len(results),
            "wide": [r for r in results if r["direction"] == "WIDE"],
            "narrow": [r for r in results if r["direction"] == "NARROW"],
            "mixed": [r for r in results if r["direction"] == "MIXED"],
        }
    else:
        agg = {"n_prompts": 0}

    out = {
        "suite_file": args.suite,
        "model": "gemini-flash-latest",
        "results": results,
        "aggregate": agg,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[suite] saved: {out_path}", file=sys.stderr)
    if results:
        print(f"[suite] aggregate: n={agg['n_prompts']} stream={agg['n_stream']} mean_ratio={agg['mean_unique_script_ratio']:.3f}")


if __name__ == "__main__":
    main()
