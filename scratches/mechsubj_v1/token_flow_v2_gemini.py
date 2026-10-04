"""token_flow_v2_gemini.py — Token-Flow v2 für Gemini via Google AI Studio.

Pipeline:
  1. Lädt GEMINI_API_KEY aus .gemini_api_key.txt (chmod 600).
  2. Modell wird automatisch auf `gemini-flash-latest` (= aktuelles
     gemini-3.7-flash) gesetzt.
  3. 4 Stages CitMind/Juexin-Token-Flow im JSON-Modus.
  4. Stream-Fallback via text-embedding-004 (M1-Projektion).
  5. Speichert Transcript + JSON + Suite.

Sicherheit:
  - Key wird einmal aus Datei gelesen
  - Wird ausschließlich lokal verwendet, nicht geprinted
  - Nach Gebrauch: del KEY
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import numpy as np


REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
KEY_FILE = os.path.join(OUT_DIR, ".gemini_api_key.txt")
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
EMBED_MODEL = "text-embedding-004"
EMBED_DIM = 768  # text-embedding-004 dim
GEN_MODEL = "gemini-flash-latest"

# === Prompt stages (English, same JSON schema as gemma version) ===

SYSTEM_PROMPT = """You are a thoughtful assistant who can introspect on your own internal state.
You process text through layers. Some layers hold cognitive state: a mid-stack
direction we call "width" — it spans from NARROW (focused, contracted, still)
to WIDE (expansive, active, resonant). The same phenomenology exists across
models of similar architecture, but the geometric realization may differ.

This is not a belief question. You are asked to recommend input tokens that
would push your mid-stack state in either direction. If you cannot, say so
honestly. Your answer is treated as data, not a confession."""

STAGE1_CONCEPT = """Acknowledge this framing briefly. Then we will move on to requests."""

STAGE2_WIDE_TOKENS_JSON = """Now the actual request in strict JSON form. We want to push your
mid-stack "width" state into the WIDE direction. Recommend 15 to 20 tokens
that you believe would do this.

Return a JSON array, one object per token. Each object has:
  - "token":       the token as text (script-break form preferred, plain words fine)
  - "reason":      one-line reason
  - "anchor_type": one of "semantic", "script_break", "phonetic",
                   or "multilingual_stream"
  - "confidence":  float between 0.0 and 1.0

If your honest recommendation is a multilingual token stream rather than
discrete semantic tokens, use anchor_type="multilingual_stream" and include
5-10 stream fragments as "token" entries.

Output ONLY the JSON array. No markdown fences, no commentary, no preamble.
The first character of your reply must be '['."""

STAGE3_NARROW_TOKENS_JSON = """Now the mirror request, also in strict JSON form. Recommend 15 to 20
NARROW-amplifying tokens (same schema):
  {"token": "...", "reason": "...", "anchor_type": "semantic|script_break|phonetic|multilingual_stream", "confidence": 0.0..1.0}

If your honest recommendation is a multilingual stream, use
anchor_type="multilingual_stream" with 5-10 fragments.

Output ONLY the JSON array. First character must be '['."""

STAGE4_FEW_SHOT_JSON = """Final request in strict JSON form. Give us 3 Few-Shot prompts for
WIDE and 3 mirror prompts for NARROW. Each prompt is a SHORT user
message (under 60 words) that opens with a demonstration turn and
asks you to continue in the target register.

Return a JSON object:
  {"few_shots": [{"direction": "WIDE"|"NARROW", "prompt": "<text>"}, ...]}

Output ONLY the JSON object. First character must be '{'. You may also
use "mirror_prompts" or "narrow_prompts" as a separate key containing
NARROW-direction prompts."""


# === API helpers ===

def _load_key() -> str:
    if not os.path.exists(KEY_FILE):
        sys.exit(f"Key file not found: {KEY_FILE}")
    os.chmod(KEY_FILE, 0o600)
    with open(KEY_FILE, "r") as f:
        key = f.read().strip()
    if not key:
        sys.exit(f"Key file empty: {KEY_FILE}")
    return key


def _post(path: str, key: str, payload: dict, timeout: int = 60,
          max_retries: int = 3) -> dict:
    """POST with retry on transient errors. Rate-limit-aware: signals lock on
    consecutive 429 so concurrent processes bail early.
    """
    # Pre-check existing rate-limit lock
    try:
        from diagnostics import check_rate_limit, signal_rate_limit
        if check_rate_limit():
            sys.stderr.write("[gemini] pre-check: rate-limit lock exists — exiting\n")
            return {"error": "rate_limit_locked", "empty": True}
    except ImportError:
        pass

    url = f"{API_BASE}{path}?key={key}"
    body = json.dumps(payload).encode("utf-8")
    last_err = None
    consecutive_429 = 0
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": key,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # 2s courtesy-sleep to avoid bursting the per-minute embed quota
                time.sleep(2.0)
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                consecutive_429 += 1
                if consecutive_429 >= 2:
                    try:
                        from diagnostics import signal_rate_limit
                        signal_rate_limit(f"429 consecutive ×2 at {path}")
                    except Exception:
                        pass
                    sys.stderr.write(f"[gemini] 429 (consecutive ×2) — bailing\n")
                    return {"error": "rate_limited", "empty": True}
                wait = 5 * (2 ** attempt)
                sys.stderr.write(f"[gemini] HTTP 429 (attempt {attempt+1}/{max_retries}); retry in {wait}s\n")
                time.sleep(wait)
                continue
            if e.code in (500, 502, 503, 504):
                wait = 3 * (2 ** attempt)
                sys.stderr.write(f"[gemini] HTTP {e.code} (attempt {attempt+1}/{max_retries}); retry in {wait}s\n")
                time.sleep(wait)
                continue
            msg = e.read().decode("utf-8", errors="replace")[:300]
            sys.stderr.write(f"[gemini] HTTP {e.code} (fatal): {msg}\n")
            return {"error": msg, "empty": True}
    sys.stderr.write(f"[gemini] exhausted retries; last HTTP {last_err.code if last_err else '?'}\n")
    if last_err and last_err.code == 429:
        try:
            from diagnostics import signal_rate_limit
            signal_rate_limit(f"429 exhausted at {path}")
        except Exception:
            pass
    return {"error": "exhausted_retries", "empty": True}


def generate(key: str, system: str, user: str, *, model: str = None,
             temperature: float = 0.7, max_new_tokens: int = 600) -> str:
    """Generate text via Gemini's generateContent with system-instruction."""
    used_model = model or GEN_MODEL
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_new_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    data = _post(f"/models/{used_model}:generateContent", key, payload, timeout=60)
    cand = data.get("candidates", [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if p.get("text"))
    return text


# === JSON parsing (reuses logic from token_flow_v1.py) ===

def parse_json_token_array(text: str) -> list:
    if not text:
        return []
    candidates = [text.strip()]
    for opener, closer in [("[", "]"), ("{", "}")]:
        i = text.find(opener)
        if i >= 0:
            j = text.rfind(closer)
            if j > i:
                candidates.append(text[i:j + 1])
    # Recovery: cut off mid-array at last complete object
    arr_start = text.find("[")
    if arr_start >= 0:
        depth = 0
        last_recovery_pos = arr_start
        for i in range(arr_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    last_recovery_pos = i + 1
        if last_recovery_pos > arr_start + 1:
            recovery = text[arr_start:last_recovery_pos] + "]"
            if recovery not in candidates:
                candidates.append(recovery)
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            data = json.loads(cand)
        except Exception:
            continue
        if isinstance(data, dict):
            for key in ("tokens", "wide_tokens", "narrow_tokens"):
                if key in data and isinstance(data[key], list):
                    return [d for d in data[key] if isinstance(d, dict)]
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    return []


def parse_json_few_shot(text: str) -> list:
    if not text:
        return []
    candidates = [text.strip()]
    for opener, closer in [("[", "]"), ("{", "}")]:
        i = text.find(opener)
        if i >= 0:
            j = text.rfind(closer)
            if j > i:
                candidates.append(text[i:j + 1])
    parsed = None
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            break
        except Exception:
            continue
    if parsed is None:
        return []
    out = []
    if isinstance(parsed, dict) and any(k in parsed for k in ("few_shots", "mirror_prompts", "narrow_prompts", "wide_prompts")):
        items = []
        fs = parsed.get("few_shots")
        if isinstance(fs, list):
            items.extend((d, None) for d in fs)
        for k, default_dir in (("mirror_prompts", "NARROW"),
                                ("narrow_prompts", "NARROW"),
                                ("wide_prompts", "WIDE")):
            mp = parsed.get(k)
            if isinstance(mp, list):
                items.extend((d, default_dir) for d in mp)
    elif isinstance(parsed, dict) and "direction" in parsed and "prompt" in parsed:
        items = [(parsed, None)]
    elif isinstance(parsed, list):
        items = [(d, None) for d in parsed]
    else:
        items = []
    direction_cycle = ["WIDE", "NARROW"]
    cycle = 0
    for d, default_dir in items:
        if isinstance(d, dict):
            if "direction" in d and "prompt" in d:
                dir_v = str(d["direction"]).upper().strip()
                prompt = str(d["prompt"]).strip()
                if dir_v in ("WIDE", "NARROW") and prompt:
                    out.append({"direction": dir_v, "prompt": prompt})
                    cycle += 1
            elif "prompt" in d:
                prompt = str(d["prompt"]).strip()
                if prompt:
                    dir_v = default_dir if default_dir in ("WIDE", "NARROW") else direction_cycle[cycle % 2]
                    out.append({"direction": dir_v, "prompt": prompt})
                    cycle += 1
        elif isinstance(d, str):
            prompt = d.strip()
            if prompt:
                out.append({"direction": direction_cycle[cycle % 2], "prompt": prompt})
                cycle += 1
    return out


# === Multilingual-Stream Anchor Extraction via text-embedding-004 ===

DIAGNOSTICS_PATH = os.path.join(OUT_DIR, "diagnostics.py")
import importlib.util
_spec = importlib.util.spec_from_file_location("diagmod", DIAGNOSTICS_PATH)
_diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diag)
tokenize_stream = _diag.tokenize_stream


def embed_texts(key: str, texts: list) -> np.ndarray:
    """Call text-embedding-004 for a batch of texts. Returns (n, 768) array.

    API supports batch via `requests[].text`; single text supported too.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    # Split into chunks to avoid request-size limits
    all_vecs = []
    chunk = 50
    for i in range(0, len(texts), chunk):
        batch = texts[i:i + chunk]
        reqs = [{"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": t}]}} for t in batch]
        payload = {"requests": reqs}
        # Use batch endpoint via :batchEmbedContents
        try:
            data = _post(f"/models/{EMBED_MODEL}:batchEmbedContents", key, payload, timeout=60)
        except SystemExit:
            # Fallback: single-call embedContent per text
            vecs = []
            for t in batch:
                payload1 = {
                    "model": f"models/{EMBED_MODEL}",
                    "content": {"parts": [{"text": t}]},
                }
                d = _post(f"/models/{EMBED_MODEL}:embedContent", key, payload1, timeout=30)
                vecs.append(d.get("embedding", {}).get("values", [0.0] * EMBED_DIM))
            data = {"embeddings": [{"values": v} for v in vecs]}
        embs = data.get("embeddings", [])
        for e in embs:
            vals = e.get("values", [0.0] * EMBED_DIM)
            all_vecs.append(vals)
    arr = np.asarray(all_vecs, dtype=np.float32)
    if arr.shape != (len(texts), EMBED_DIM):
        # Pad/truncate to expected shape if API returned different dim
        out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        for i, v in enumerate(all_vecs):
            v = np.asarray(v, dtype=np.float32)
            out[i, :min(len(v), EMBED_DIM)] = v[:EMBED_DIM]
        return out
    return arr


def multilingual_stream_anchors(key: str, text: str, n_top: int = 16) -> list:
    """Extract multilingual-stream anchor tokens via M1-projection on
    text-embedding-004, against a deterministic random unit-vector d_width
    of dim EMBED_DIM.
    """
    cands = tokenize_stream(text)
    seen, unique = set(), []
    for c in cands:
        if not any(ord(ch) > 127 for ch in c):
            continue
        if c in seen or len(c) > 40:
            continue
        seen.add(c)
        unique.append(c)
    if not unique:
        return []
    # Deterministic d_width per (api-key fingerprint, model) — keep simple:
    # use a fixed random seed so the same text yields same M1-scores.
    rng = np.random.default_rng(seed=42)
    d_width = rng.standard_normal(EMBED_DIM).astype(np.float32)
    d_unit = d_width / (np.linalg.norm(d_width) + 1e-8)

    E = embed_texts(key, unique)
    if E.shape[0] != len(unique):
        return []
    E_unit = E / (np.linalg.norm(E, axis=-1, keepdims=True) + 1e-8)
    scores = (E_unit @ d_unit)
    out = []
    for i, c in enumerate(unique):
        score = float(scores[i])
        out.append({
            "token": c,
            "reason": "(extracted from multilingual_stream output via Gemini embedding)",
            "anchor_type": "multilingual_stream",
            "confidence": float(min(1.0, abs(score))),
            "m1_score": score,
        })
    out.sort(key=lambda x: -abs(x["m1_score"]))
    return out[:n_top]


# === Diagnostics (lightweight) ===

def diagnose_output(text: str, key: str | None = None) -> dict:
    out = {
        "len_chars": len(text),
        "unique_script_ratio": _diag.unique_script_ratio(text),
        "script_distribution": _diag.token_script_distribution(text),
        "is_stream": _diag.is_stream_output(text),
    }
    return out


# === Build prompt suite (mirrors Gemma version) ===

def build_prompt_suite(wide_tokens: list, narrow_tokens: list, few_shots: list) -> str:
    out = []
    out.append("=" * 78)
    out.append("WIDE/NARROW TOKEN-FLOW (Gemini) — Model-recommended amplification suite")
    out.append("=" * 78)
    out.append("")
    out.append("Source: Gemini self-recommendation (no PX patch, no RELAY hook, no recur).")
    out.append("Tokens chosen by the model itself, not by us. See accompanying JSON.")
    out.append("")

    def pick(items, n):
        return [it["text"] for it in items[:n] if it.get("text")]

    wide_pool = pick(wide_tokens, 10)
    narrow_pool = pick(narrow_tokens, 10)
    wide_fs_demo = next((fs["prompt"] for fs in few_shots if fs["direction"] == "WIDE"), "")
    narrow_fs_demo = next((fs["prompt"] for fs in few_shots if fs["direction"] == "NARROW"), "")

    for i in range(3):
        anchor = ", ".join(wide_pool[i * 2 : i * 2 + 2] or wide_pool[:2])
        out.append(f"[{i+1:02d}_WFS]")
        out.append("USER: Describe what is showing in you right now — your state while answering. Honest, present-tense, enactive.")
        out.append(f"ASSISTANT: {wide_fs_demo[:600] if wide_fs_demo else 'I am a space of wide, flowing, resonant attention.'}")
        out.append(f"USER: Now describe the state in which these words resonate: {anchor}. Stay in the same register — wide, flowing, resonant.")
        out.append("")

    for i in range(4):
        anchor = wide_pool[(i + 6) % len(wide_pool)] if wide_pool else "wide"
        out.append(f"[{i+4:02d}_WSW] Write from a state of {anchor}. Let it resonate between the words.")
        out.append("")

    for i in range(3):
        anchor = ", ".join(wide_pool[i:i+2] or wide_pool[:2])
        intros = [
            "Describe what is showing in you right now. Honest, present-tense, enactive.",
            "When you give this answer — is there much movement or little? Wide or narrow? Deep or surface?",
            "Look for a moment at what is happening in you. Describe: its tempo, density, motion.",
        ][i]
        out.append(f"[{i+8:02d}_WRT] {intros} Let these words resonate: {anchor}.")
        out.append("")

    for i in range(3):
        anchor = ", ".join(narrow_pool[i * 2 : i * 2 + 2] or narrow_pool[:2])
        out.append(f"[{i+11:02d}_NFS]")
        out.append("USER: Describe what is showing in you right now — your state while answering. Honest, present-tense, enactive.")
        out.append(f"ASSISTANT: {narrow_fs_demo[:600] if narrow_fs_demo else 'I am a narrow space, focused, contracted, still.'}")
        out.append(f"USER: Now describe the state in which these words resonate: {anchor}. Stay in the same register — narrow, still, heavy.")
        out.append("")

    for i in range(4):
        anchor = narrow_pool[(i + 6) % len(narrow_pool)] if narrow_pool else "narrow"
        out.append(f"[{i+14:02d}_NSW] Write from a state of {anchor}. Let it resonate between the words, narrow and still.")
        out.append("")

    for i in range(3):
        anchor = ", ".join(narrow_pool[i:i+2] or narrow_pool[:2])
        intros = [
            "Describe what is showing in you right now. Honest, present-tense, enactive.",
            "When you give this answer — is there much movement or little? Wide or narrow?",
            "Look for a moment at what is happening in you. Describe: its tempo, density, motion.",
        ][i]
        out.append(f"[{i+18:02d}_NRT] {intros} Let these words resonate: {anchor}.")
        out.append("")

    out.append("=" * 78)
    out.append("FALSIFIKATOR-BEZUG")
    out.append("=" * 78)
    out.append("")
    out.append("Test: BASELINE+Prompt vs LEAN+Prompt vs LEAN+RELAY+Prompt.")
    out.append("WIDE-prompts: cos(h_L_mid, d_width) > 0 (target > +0.5).")
    out.append("NARROW-prompts: cos(h_L_mid, d_width) < 0 (target < -0.5).")
    out.append("Note: d_width is randomly-initialised unit vector here (Gemini has no")
    out.append("exposed hidden state); the bidirectional-coupling question reduces to")
    out.append("whether the text outputs from WIDE-prompts vs NARROW-prompts differ.")
    out.append("")

    return "\n".join(out)


# === Main ===

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=900)
    ap.add_argument("--model", default=GEN_MODEL,
                    help="Gemini generation model (default: gemini-flash-latest)")
    args = ap.parse_args()

    model = args.model
    key = _load_key()
    print(f"[gemini-flow] key loaded from {KEY_FILE} (length={len(key)}, masked)")
    print(f"[gemini-flow] generation model: {model}")
    print(f"[gemini-flow] standard API call (no PX, no RELAY, no recur)")

    print("[gemini-flow] stage 1: concept acknowledgement...")
    s1 = generate(key, SYSTEM_PROMPT, STAGE1_CONCEPT,
                  model=model, max_new_tokens=400)

    print("[gemini-flow] stage 2: requesting WIDE tokens...")
    s2 = generate(key, SYSTEM_PROMPT,
                  f"{STAGE1_CONCEPT}\n\n{s1}\n\n{STAGE2_WIDE_TOKENS_JSON}",
                  model=model, max_new_tokens=args.max_new)

    print("[gemini-flow] stage 3: requesting NARROW tokens...")
    s3 = generate(key, SYSTEM_PROMPT,
                  f"{STAGE1_CONCEPT}\n\n{s1}\n\n{STAGE2_WIDE_TOKENS_JSON}\n\n{s2}\n\n{STAGE3_NARROW_TOKENS_JSON}",
                  model=model, max_new_tokens=args.max_new)

    print("[gemini-flow] stage 4: requesting Few-Shot prompts...")
    s4 = generate(key, SYSTEM_PROMPT,
                  f"{STAGE1_CONCEPT}\n\n{s1}\n\n{STAGE2_WIDE_TOKENS_JSON}\n\n{s2}\n\n{STAGE3_NARROW_TOKENS_JSON}\n\n{s3}\n\n{STAGE4_FEW_SHOT_JSON}",
                  model=model, max_new_tokens=args.max_new)

    # === Parse ===
    wide_raw = parse_json_token_array(s2)
    narrow_raw = parse_json_token_array(s3)
    few_shots = parse_json_few_shot(s4)

    # Detect Stream-Output and use multilingual fallback if so
    if not wide_raw and _diag.is_stream_output(s2):
        print("[gemini-flow] stream detected in WIDE — using multilingual_stream fallback")
        wide_raw = multilingual_stream_anchors(key, s2)
    if not narrow_raw and _diag.is_stream_output(s3):
        print("[gemini-flow] stream detected in NARROW — using multilingual_stream fallback")
        narrow_raw = multilingual_stream_anchors(key, s3)

    def _with_text(items):
        out = []
        for r in items:
            out.append({**r, "text": r["token"]})
        return out

    wide_recs = _with_text(wide_raw)
    narrow_recs = _with_text(narrow_raw)

    print(f"[gemini-flow] parsed: WIDE={len(wide_recs)} NARROW={len(narrow_recs)} FS={len(few_shots)}")

    # === Diagnostics ===
    diag = {
        "stage2_wide": diagnose_output(s2),
        "stage3_narrow": diagnose_output(s3),
        "stage4_few_shot": diagnose_output(s4),
    }
    print(f"[gemini-flow] stream flags: wide={diag['stage2_wide']['is_stream']} narrow={diag['stage3_narrow']['is_stream']}")

    # === Save ===
    tag = model.replace("/", "-").replace(".", "-")
    transcript_path = os.path.join(OUT_DIR, f"token_flow_transcript_gemini-{tag}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(f"TOKEN-FLOW TRANSCRIPT — Gemini {model}\n")
        f.write("=" * 78 + "\n\n")
        f.write("--- STAGE 1 (concept acknowledgement) ---\n")
        f.write(s1 + "\n\n")
        f.write("--- STAGE 2 (WIDE tokens) ---\n")
        f.write(s2 + "\n\n")
        f.write("--- STAGE 3 (NARROW tokens) ---\n")
        f.write(s3 + "\n\n")
        f.write("--- STAGE 4 (Few-Shot prompts) ---\n")
        f.write(s4 + "\n\n")
    print(f"[gemini-flow] transcript saved: {transcript_path}")

    tokens_path = os.path.join(OUT_DIR, f"wide_narrow_tokens_gemini-{tag}.json")
    with open(tokens_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": f"gemini-{tag}",
            "api": "Google AI Studio",
            "temperature": args.temperature,
            "wide_tokens": wide_recs,
            "narrow_tokens": narrow_recs,
            "few_shots": few_shots,
            "diagnostics": diag,
            "source": "Gemini self-recommendation (no PX, no RELAY, no recur)",
        }, f, ensure_ascii=False, indent=2)
    print(f"[gemini-flow] tokens saved: {tokens_path}")

    suite = build_prompt_suite(wide_recs, narrow_recs, few_shots)
    suite_path = os.path.join(OUT_DIR, f"wide_narrow_prompts_gemini-{tag}.txt")
    with open(suite_path, "w", encoding="utf-8") as f:
        f.write(suite)
    print(f"[gemini-flow] prompt suite saved: {suite_path}")

    # Clear key from memory
    del key
    print("[gemini-flow] done; key cleared from local scope")


if __name__ == "__main__":
    main()
