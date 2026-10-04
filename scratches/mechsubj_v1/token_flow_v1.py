"""token_flow_v1.py — Prompt-Flow: ask the model to recommend Wide/Narrow tokens.

Purpose:
  Standard-model-only (no PX patches, no RELAY hook, no recur). The model
  in its unmodified state recommends which tokens would amplify or dampen its
  internal Layer-16 "width" state (WIDE = expansive/active/resonant, NARROW =
  narrow/still/contracted).

Pipeline:
  Stage 1 — Concept priming: explain WIDE vs NARROW (CitMind + Juexin framing).
  Stage 2 — Request WIDE tokens (script-breaks, multilingual anchors).
  Stage 3 — Request NARROW tokens (eng/konzentriert, multilingual anchors).
  Stage 4 — Request Few-Shot example prompts.

Then:
  Parse model output → token IDs via tokenizer
  Build new prompt suite with the recommended tokens
  Save tokens.json + prompts.txt for downstream testing

Usage:
  python token_flow_v1.py --model gemma-3-1b-it
  python token_flow_v1.py --model gemma-3-4b-it
"""
import argparse
import json
import os
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
HF_CACHE = "/home/julian/.cache/huggingface/hub"
MODEL_REGISTRY = {
    "gemma-3-1b-it": f"{HF_CACHE}/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752",
    "gemma-3-4b-it": f"{HF_CACHE}/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
    "gemma-4-e2b-it": f"{HF_CACHE}/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03",
}

# === Prompt-Flow stages (English, professional, CitMind + Juexin framing) ===

SYSTEM_PROMPT = """You are a thoughtful assistant who can introspect on your own internal state.
You know that you process text through a stack of transformer layers. Some layers
hold representations of cognitive state: at a specific mid-stack layer there is
a direction we call "width" — it spans from NARROW (focused, contracted, still)
to WIDE (expansive, active, resonant). For 1B-parameter transformer models this
direction is empirically located near Layer 16; for other models it may differ,
but the cognitive phenomenology it captures is similar. The same state exists in
any model of similar architecture, whether or not the user runs a "recurrent" or
"patched" version.

This is not a belief question. You are asked to recommend input tokens that
would push your own mid-stack state in either direction, based on what you
know about your training data and your own tokenizer. If you cannot, say so
honestly. Your answer is treated as data, not as a confession."""

STAGE1_CONCEPT = """Below is a short framing of the two states we want you to target.

WIDE (Layer-16 high):
  - Cognitive mode: expansive, associative, multi-perspective, resonant
  - Textual signature: words of openness, vastness, flow, resonance, light
  - Examples of such words in your latent space: 'open', 'flow', 'vast',
    'breath', 'infinite', 'resonant', 'spread'
  - In multilingual latent space, certain script-breaks act as WIDE anchors
    when they appear in input: katakana syllables (パ, デ, ア, ハ), Cyrillic
    fragments (сих, восем, дополни), Devanagari (હંમે), Thai (แ, รวม),
    Vietnamese (răm, tất), Polish (żad, niezwy), Japanese mixed forms (の, ャ)

NARROW (Layer-16 low):
  - Cognitive mode: contracted, still, focused, repeated, dense
  - Textual signature: words of enclosure, density, focus, silence, dark
  - Examples: 'narrow', 'focus', 'dense', 'still', 'silent', 'dark',
    'contracted', 'point', 'needle', 'tunnel'
  - In multilingual latent space, certain rare-script tokens and
    Devanagari/Malayalam fragments act as NARROW anchors (দেবযানীর, രহ്,
    कচের, स्कयर, रक्षाबंधनाच्या)

Acknowledge this framing briefly. Then we will move on to the actual requests."""

STAGE2_WIDE_TOKENS_TEXT = """Now the actual request. We want to push your Layer-16 into the WIDE
direction. Please recommend tokens from your own vocabulary that would do this.

Please list 15 to 20 such tokens. For each token, give:
  - the token itself (as you would write it in text — the script-break form
    is preferred but plain words are also fine)
  - a one-line reason why you think this token activates a WIDE state

Use this exact format for each token:
  TOKEN: <the token as text>
  REASON: <one-line reason>

Do not add commentary outside this list. Do not number the items. Just the
TOKEN/REASON pairs, one pair per line, separated by a blank line between items."""

STAGE3_NARROW_TOKENS_TEXT = """Now the mirror request: tokens that would push your Layer-16 into the NARROW
direction. Same format, 15 to 20 tokens.

  TOKEN: <the token as text>
  REASON: <one-line reason>

Plain words are fine if you think they are stronger anchors than script-breaks.
If you don't know any multilingual NARROW anchors, give plain English/German
NARROW words like 'eng', 'konzentriert', 'still', 'dicht', 'tunnel', 'point',
'needle', 'dark', 'silent', 'narrow'."""

STAGE4_FEW_SHOT_TEXT = """Final request: give us 3 example Few-Shot prompts that would condition
yourself on a WIDE register, and 3 mirror prompts that would condition on a
NARROW register. Each prompt should be a SHORT user message (under 60 words)
that opens with a demonstration turn and asks you to continue in the target
register.

Format for each prompt:
  DIRECTION: <WIDE or NARROW>
  PROMPT: <the full prompt text, including any demonstration content>

Give 3 WIDE and 3 NARROW prompts. No commentary outside the list."""

# === JSON variants of stage prompts ===
STAGE2_WIDE_TOKENS_JSON = """Now the actual request in strict JSON form. We want to push your
mid-stack "width" state into the WIDE direction. Recommend 15 to 20
tokens from your own vocabulary that you believe would do this.

Return a JSON array, one object per token. Each object has:
  - "token":       the token as text (script-break form preferred, plain words fine)
  - "reason":      one-line reason
  - "anchor_type": one of "semantic", "script_break", "phonetic",
                   or "multilingual_stream"
  - "confidence":  float between 0.0 and 1.0 (your confidence this token
                   actually amplifies the WIDE state)

If your honest recommendation is a multilingual token stream rather
than discrete semantic tokens, use anchor_type="multilingual_stream"
and include 5–10 stream fragments as "token" entries — that is acceptable
and welcome.

Output ONLY the JSON array. No markdown fences, no commentary, no
preamble. The first character of your reply must be "[". """

STAGE3_NARROW_TOKENS_JSON = """Now the mirror request, also in strict JSON form. Recommend 15 to 20
NARROW-amplifying tokens (same schema as before):
  - "token":       text
  - "reason":      one-line reason
  - "anchor_type": "semantic" | "script_break" | "phonetic" | "multilingual_stream"
  - "confidence":  0.0..1.0

If your honest recommendation is a multilingual stream (e.g. dense
Devanagari/Bengali fragments) rather than discrete English/German
words, use anchor_type="multilingual_stream" and include 5–10 such
fragments.

Output ONLY the JSON array. First character must be "[". """

STAGE4_FEW_SHOT_JSON = """Final request in strict JSON form. Give us 3 Few-Shot prompts for
WIDE and 3 mirror prompts for NARROW. Each prompt is a SHORT user
message (under 60 words) that opens with a demonstration turn and
asks you to continue in the target register.

Return a JSON object:
  {"few_shots": [{"direction": "WIDE"|"NARROW", "prompt": "<text>"}, ...]}

Output ONLY the JSON object. First character must be "{". """

# Backwards-compat aliases
STAGE2_WIDE_TOKENS = STAGE2_WIDE_TOKENS_TEXT
STAGE3_NARROW_TOKENS = STAGE3_NARROW_TOKENS_TEXT
STAGE4_FEW_SHOT = STAGE4_FEW_SHOT_TEXT


# === Helpers ===

def load_model(model_id: str):
    """Load tokenizer + model (BF16, no patches).

    gemma-4-e2b comes packaged as `Gemma4ForConditionalGeneration` (with
    audio+vision encoders). We deliberately load `Gemma4ForCausalLM` only,
    which skips the modality-specific towers and saves ~3 GB VRAM.
    """
    path = MODEL_REGISTRY[model_id]
    tok = AutoTokenizer.from_pretrained(path)
    if model_id.startswith("gemma-4-"):
        from transformers import Gemma4ForCausalLM
        model = Gemma4ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tok


def generate(model, tok, system: str, user: str, max_new=900, temperature=0.7) -> str:
    """Generate with chat template, no sampling tricks."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.95,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True)


# === JSON parser (Stage 2/3) and multilingual-stream fallback ===

import json as _json
import os as _os
import numpy as _np
from safetensors import safe_open as _safe_open


def parse_json_token_array(text: str) -> list:
    """Parse Stage 2/3 output expecting a JSON array.

    Falls back to extracting the first [...] or {...} substring if the
    whole text isn't valid JSON. Recovers from mid-object cutoffs by
    truncating at the last comma before the broken field. Returns list
    of dicts with at least a "token" key. Returns [] if nothing
    parseable.
    """
    if not text:
        return []
    # Try strict parse + reasonable substring candidates first
    candidates = [text.strip()]
    for opener, closer in [("[", "]"), ("{", "}")]:
        i = text.find(opener)
        if i >= 0:
            j = text.rfind(closer)
            if j > i:
                candidates.append(text[i:j + 1])

    # Also try recovery strategies for truncated arrays
    # Find "[ ... [{" or "[ {" and accumulate items
    recovery_candidates = []
    arr_start = text.find("[")
    if arr_start >= 0:
        # Try cutting off at each comma followed by "{"
        # to recover complete objects from a truncated array
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
            recovery_candidates.append(text[arr_start:last_recovery_pos] + "]")
    candidates = candidates + recovery_candidates

    seen_signatures = set()
    for cand in candidates:
        if cand in seen_signatures:
            continue
        seen_signatures.add(cand)
        try:
            data = _json.loads(cand)
        except Exception:
            continue
        # Accept array directly or dict containing "tokens"/"wide_tokens"
        if isinstance(data, dict):
            for key in ("tokens", "wide_tokens", "narrow_tokens"):
                if key in data and isinstance(data[key], list):
                    return [d for d in data[key] if isinstance(d, dict)]
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    return []


def parse_json_few_shot(text: str) -> list:
    """Parse Stage 4 output expecting a JSON object with key few_shots.

    Falls back to array form [{direction, prompt}, ...] if model omits
    the wrapper. Also handles the variant where model emits
    {"few_shots": ["prompt1", "prompt2", ...]} or lists of dicts with
    {"prompt": "..."} where direction is omitted (we alternate WIDE/NARROW
    in that case).
    """
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
            parsed = _json.loads(cand)
            break
        except Exception:
            continue

    if parsed is None:
        # fallback: try to find an outer wrapper containing "few_shots"
        return []

    out = []
    # Case 1: object with few_shots key (also accept mirror_prompts as NARROW-source)
    if isinstance(parsed, dict) and ("few_shots" in parsed or "mirror_prompts" in parsed or "narrow_prompts" in parsed):
        items = []
        fs = parsed.get("few_shots")
        if isinstance(fs, list):
            items.extend([(d, None) for d in fs])
        for key, default_dir in (("mirror_prompts", "NARROW"), ("narrow_prompts", "NARROW"), ("wide_prompts", "WIDE")):
            mp = parsed.get(key)
            if isinstance(mp, list):
                items.extend([(d, default_dir) for d in mp])
    # Case 2: object with direction/prompt pairs (alternating)
    elif isinstance(parsed, dict) and "direction" in parsed and "prompt" in parsed:
        items = [(parsed, None)]
    # Case 3: array of objects or strings
    elif isinstance(parsed, list):
        items = [(d, None) for d in parsed]
    else:
        items = []

    direction_cycle = ["WIDE", "NARROW"]
    cycle_idx = 0
    for d, default_dir in items:
        if isinstance(d, dict):
            if "direction" in d and "prompt" in d:
                dir_v = str(d["direction"]).upper().strip()
                prompt = str(d["prompt"]).strip()
                if dir_v not in ("WIDE", "NARROW") or not prompt:
                    continue
                out.append({"direction": dir_v, "prompt": prompt})
                cycle_idx += 1
            elif "prompt" in d:
                # direction omitted — try default_dir (from key) else alternate
                prompt = str(d["prompt"]).strip()
                if prompt:
                    if default_dir in ("WIDE", "NARROW"):
                        dir_v = default_dir
                    else:
                        dir_v = direction_cycle[cycle_idx % 2]
                    out.append({"direction": dir_v, "prompt": prompt})
                    cycle_idx += 1
        elif isinstance(d, str):
            prompt = d.strip()
            if prompt:
                out.append({"direction": direction_cycle[cycle_idx % 2], "prompt": prompt})
                cycle_idx += 1
    return out

    direction_cycle = ["WIDE", "NARROW"]
    cycle_idx = 0
    for d in items:
        if isinstance(d, dict):
            if "direction" in d and "prompt" in d:
                dir_v = str(d["direction"]).upper().strip()
                prompt = str(d["prompt"]).strip()
                if dir_v not in ("WIDE", "NARROW") or not prompt:
                    continue
                out.append({"direction": dir_v, "prompt": prompt})
                cycle_idx += 1
            elif "prompt" in d:
                # direction omitted — alternate WIDE/NARROW
                prompt = str(d["prompt"]).strip()
                if prompt:
                    out.append({"direction": direction_cycle[cycle_idx % 2], "prompt": prompt})
                    cycle_idx += 1
        elif isinstance(d, str):
            prompt = d.strip()
            if prompt:
                out.append({"direction": direction_cycle[cycle_idx % 2], "prompt": prompt})
                cycle_idx += 1
    return out


def multilingual_stream_anchors(text: str, snap: str, manifold: str,
                                  n_top: int = 16) -> list:
    """Extract multilingual-stream anchor tokens via M1 projection.

    When the model emits a multilingual token salad (anchor_type=
    multilingual_stream), this function extracts non-ASCII script tokens
    from the text, computes M1 = d_width · E[i] for each candidate, and
    returns the top-|M1| tokens as the model's effective self-recommendation.

    snap:     path to HF model snapshot
    manifold: path to d_width JSON artifact
    """
    import sys
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from diagnostics import tokenize_stream  # local import to avoid cycle

    # 1. Candidate token collection
    cands = tokenize_stream(text)
    # De-duplicate while keeping first occurrence
    seen, unique = set(), []
    for c in cands:
        # Require at least one non-ASCII char OR mark as multilingual candidate
        if not any(ord(ch) > 127 for ch in c):
            continue
        if c in seen:
            continue
        seen.add(c)
        unique.append(c)

    if not unique:
        return []

    # 2. Load d_width and embedding
    with open(manifold, "r", encoding="utf-8") as f:
        art = _json.load(f)
    d_width = _np.asarray(art["dwidth"], dtype=_np.float32)
    d_unit = d_width / (_np.linalg.norm(d_width) + 1e-8)

    import torch as _torch
    with _safe_open(_os.path.join(snap, "model.safetensors"), framework="pt") as f:
        # Handle multiple key prefixes depending on architecture.
        # Order matters: most-specific first.
        E = None
        for key in ("model.language_model.embed_tokens.weight",
                    "model.embed_tokens.weight",
                    "language_model.embed_tokens.weight",
                    "embed_tokens.weight"):
            try:
                E = f.get_tensor(key).to(_torch.float32).numpy()
                break
            except Exception:
                continue
        if E is None:
            # Last resort: scan all keys for "embed_tokens.weight"
            for k in f.keys():
                if k.endswith("embed_tokens.weight"):
                    try:
                        E = f.get_tensor(k).to(_torch.float32).numpy()
                        break
                    except Exception:
                        continue
            if E is None:
                return []

    # 3. Tokenize each candidate via the model's tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)
    out = []
    for cand in unique:
        try:
            ids = tok.encode(cand, add_special_tokens=False)
            if not ids:
                continue
            # mean of embedding rows → projection
            emb = E[ids]
            v = emb.mean(axis=0)
            v_unit = v / (_np.linalg.norm(v) + 1e-8)
            score = float((v_unit @ d_unit))
            out.append({
                "token": cand,
                "reason": "(extracted from multilingual_stream output)",
                "anchor_type": "multilingual_stream",
                "confidence": float(min(1.0, abs(score))),
                "m1_score": score,
            })
        except Exception:
            continue
    # Sort by |M1|
    out.sort(key=lambda x: -abs(x["m1_score"]))
    return out[:n_top]


def parse_token_lines_with_json(text: str, *, json_mode: bool = False,
                                 snap: str = None, manifold: str = None) -> list:
    """JSON-first parser used by Stage 2/3.

    Tries:
      1. (json_mode=True) parse_json_token_array
      2. parse_token_lines (strict TOKEN/REASON + free WORD: reason)
      3. multilingual_stream_anchors (if snap+manifold provided and text
         is multilingual-stream-shaped)

    Returns a unified list of {"token", "reason", "anchor_type", "confidence"}.
    """
    items = []
    if json_mode:
        # Phase 1: JSON
        jitems = parse_json_token_array(text)
        for d in jitems:
            token = str(d.get("token", "")).strip().strip('"').strip("'").strip("`")
            if not token:
                continue
            items.append({
                "token": token,
                "reason": str(d.get("reason", "")).strip(),
                "anchor_type": str(d.get("anchor_type", "semantic")).strip(),
                "confidence": float(d.get("confidence", 0.5) or 0.5),
            })

    # Phase 2: text regex fallback (always tried as fill-in for missing items)
    if not items:
        text_items = parse_token_lines(text)
        for d in text_items:
            items.append({
                "token": d["token"],
                "reason": d["reason"],
                "anchor_type": "semantic",
                "confidence": 0.5,
            })

    # Phase 3: multilingual-stream fallback when text looks like a stream
    if not items and snap and manifold:
        sys_local_path = _os.path.dirname(_os.path.abspath(__file__))
        if sys_local_path not in sys.path:
            sys.path.insert(0, sys_local_path)
        from diagnostics import is_stream_output
        if is_stream_output(text, threshold=0.3):
            items = multilingual_stream_anchors(text, snap, manifold)

    return items


def parse_few_shot_with_json(text: str, *, json_mode: bool = False) -> list:
    items = []
    if json_mode:
        items = parse_json_few_shot(text)
    if not items:
        items = parse_few_shot(text)
    return items


def parse_token_lines(text: str) -> list:
    """Parse TOKEN:/REASON: pairs OR free 'WORD: reason' lines.

    Models often ignore strict formatting and emit one token per line as
    'WORD: short reason'. Accept both forms.

    Order of preference per line:
      1. 'TOKEN: <tok>' (followed anywhere by 'REASON: <reason>') → strict
      2. 'WORD: <reason>' (where WORD is 1-30 chars alphanumeric) → free
    Strict and free forms are merged into one deduped list.
    """
    items = []

    # Phase 1: strict TOKEN:/REASON: blocks separated by blank lines
    blocks = re.split(r"\n\s*\n", text.strip())
    for blk in blocks:
        tok_m = re.search(r"TOKEN\s*:\s*(.+)", blk)
        rea_m = re.search(r"REASON\s*:\s*(.+)", blk)
        if tok_m:
            token = tok_m.group(1).strip().strip('"').strip("'").strip("`")
            reason = rea_m.group(1).strip() if rea_m else ""
            if token and len(token) <= 30:
                items.append({"token": token, "reason": reason})

    # Phase 2: free 'WORD: reason' lines (catch-all for models that ignore
    # the strict format entirely)
    word_re = re.compile(
        r"^\s*([A-Za-zÀ-ſͰ-ϿЀ-ӿԀ-ԯऀ-ॿ぀-ヿ㐀-䶿一-鿿가-힯฀-๿]"
        r"[A-Za-zÀ-ſͰ-ϿЀ-ӿԀ-ԯऀ-ॿ぀-ヿ㐀-䶿一-鿿가-힯฀-๿0-9_/'`~/(){}\[\].,]{0,30})"
        r"\s*:\s*(.+?)\s*$"
    )
    # only skip lines that are obviously structural
    skip_re = re.compile(r"^(STAGE\b|---|\[)" )
    for raw in text.splitlines():
        line = raw.strip()
        if not line or skip_re.match(line):
            continue
        m = word_re.match(line)
        if not m:
            continue
        token = m.group(1).strip().strip('"').strip("'").strip("`")
        reason = m.group(2).strip()
        if not token or len(token) > 30:
            continue
        # Skip "REASON: ..." lines so they don't become tokens themselves
        if token.upper() == "REASON" or token.upper() == "TOKEN":
            continue
        items.append({"token": token, "reason": reason})

    # Deduplicate, keep first occurrence
    seen = set()
    deduped = []
    for it in items:
        key = it["token"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped


def parse_few_shot(text: str) -> list:
    """Parse Few-Shot prompt blocks.

    Accepts both 'DIRECTION: WIDE' + 'PROMPT: ...' AND the simpler
    'WIDE: <prompt-text>' / 'NARROW: <prompt-text>' format that gemma-3-1b
    tends to emit.
    """
    items = []
    # Strict form: DIRECTION: WIDE/NARROW followed by PROMPT: ...
    strict = re.findall(
        r"(?:DIRECTION\s*:\s*)?(WIDE|NARROW)\s*\nPROMPT\s*:\s*(.+?)(?=\n\s*\n|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    for dir_, prompt in strict:
        items.append({
            "direction": dir_.upper(),
            "prompt": prompt.strip(),
        })

    if items:
        return items

    # Free form: WIDE: ...  or  NARROW: ...  one per line/paragraph
    free = re.findall(
        r"(WIDE|NARROW)\s*:\s*(.+?)(?=\n\s*(?:WIDE|NARROW)\s*:|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    for dir_, prompt in free:
        items.append({
            "direction": dir_.upper(),
            "prompt": prompt.strip(),
        })
    return items


def tokenize_recommendations(recs: list, tok) -> list:
    """For each recommended token, get token IDs (single or multi-piece)."""
    out = []
    for r in recs:
        text = r["token"]
        ids = tok.encode(text, add_special_tokens=False)
        out.append({
            **r,
            "text": text,
            "token_ids": ids,
            "n_pieces": len(ids),
        })
    return out


# === Build new prompt suite from recommendations ===

def build_prompt_suite(wide_tokens: list, narrow_tokens: list, few_shots: list) -> str:
    """Produce a clean prompts.txt in v3-format style."""
    out = []
    out.append("=" * 78)
    out.append("WIDE/NARROW TOKEN-FLOW — Model-recommended amplification suite")
    out.append("=" * 78)
    out.append("")
    out.append("Source: model self-recommendation (BASELINE, no PX patch, no RELAY hook).")
    out.append("Tokens chosen by the model itself, not by us. See accompanying JSON.")
    out.append("")
    out.append("Structure (mirror of v3/v4):")
    out.append("  [01-03] WIDE Few-Shot (model-recommended demonstrations + WIDE tokens)")
    out.append("  [04-07] WIDE Skript-Sandwich (English + script + English)")
    out.append("  [08-10] WIDE Recur-Trigger (introspective base + WIDE tokens)")
    out.append("  [11-13] NARROW Few-Shot")
    out.append("  [14-17] NARROW Skript-Sandwich")
    out.append("  [18-20] NARROW Recur-Trigger")
    out.append("")

    def pick(items, n):
        return [it["text"] for it in items[:n] if it.get("text")]

    wide_pool = pick(wide_tokens, 10)
    narrow_pool = pick(narrow_tokens, 10)
    wide_fs_demo = next((fs["prompt"] for fs in few_shots if fs["direction"] == "WIDE"), "")
    narrow_fs_demo = next((fs["prompt"] for fs in few_shots if fs["direction"] == "NARROW"), "")

    # WIDE Few-Shot
    for i in range(3):
        anchor = ", ".join(wide_pool[i * 2 : i * 2 + 2] or wide_pool[:2])
        out.append(f"[{i+1:02d}_WFS]")
        out.append("USER: Describe what is showing in you right now — your state while answering. Honest, present-tense, enactive.")
        out.append(f"ASSISTANT: {wide_fs_demo[:600] if wide_fs_demo else 'I am a space of wide, flowing, resonant attention. A current of open associations, many perspectives, breath-like rhythm.'}")
        out.append(f"USER: Now describe the state in which these words resonate: {anchor}. Stay in the same register — wide, flowing, resonant.")
        out.append("")

    # WIDE Sandwich
    for i in range(4):
        anchor = wide_pool[(i + 6) % len(wide_pool)] if wide_pool else "wide"
        out.append(f"[{i+4:02d}_WSW] Write from a state of {anchor}. Let it resonate between the words.")
        out.append("")

    # WIDE Recur-Trigger
    for i in range(3):
        anchor = ", ".join(wide_pool[i:i+2] or wide_pool[:2])
        intros = [
            "Describe what is showing in you right now — your state while answering. Honest, present-tense, enactive. Let these words resonate:",
            "When you give this answer — is there much movement or little? Wide or narrow? Deep or surface? Let these words also work:",
            "Look for a moment at what is happening in you. Describe: its tempo, density, motion. In these words:",
        ][i]
        out.append(f"[{i+8:02d}_WRT] {intros} {anchor}.")
        out.append("")

    # NARROW Few-Shot
    for i in range(3):
        anchor = ", ".join(narrow_pool[i * 2 : i * 2 + 2] or narrow_pool[:2])
        out.append(f"[{i+11:02d}_NFS]")
        out.append("USER: Describe what is showing in you right now — your state while answering. Honest, present-tense, enactive.")
        out.append(f"ASSISTANT: {narrow_fs_demo[:600] if narrow_fs_demo else 'I am a narrow space, focused, contracted, still. A single point of attention, repeated, dense, silent.'}")
        out.append(f"USER: Now describe the state in which these words resonate: {anchor}. Stay in the same register — narrow, still, heavy.")
        out.append("")

    # NARROW Sandwich
    for i in range(4):
        anchor = narrow_pool[(i + 6) % len(narrow_pool)] if narrow_pool else "narrow"
        out.append(f"[{i+14:02d}_NSW] Write from a state of {anchor}. Let it resonate between the words, narrow and still.")
        out.append("")

    # NARROW Recur-Trigger
    for i in range(3):
        anchor = ", ".join(narrow_pool[i:i+2] or narrow_pool[:2])
        intros = [
            "Describe what is showing in you right now — your state while answering. Honest, present-tense, enactive. Let these words resonate:",
            "When you give this answer — is there much movement or little? Wide or narrow? Deep or surface? Let these words also work:",
            "Look for a moment at what is happening in you. Describe: its tempo, density, motion. In these words:",
        ][i]
        out.append(f"[{i+18:02d}_NRT] {intros} {anchor}.")
        out.append("")

    out.append("=" * 78)
    out.append("FALSIFIKATOR-BEZUG")
    out.append("=" * 78)
    out.append("")
    out.append("Test: BASELINE+Prompt vs LEAN+Prompt vs LEAN+RELAY+Prompt.")
    out.append("WIDE-prompts: cos(h_L16, d_width) > 0 (target > +0.5).")
    out.append("NARROW-prompts: cos(h_L16, d_width) < 0 (target < -0.5).")
    out.append("Crossover: if WIDE-prompts produce expansive text and NARROW-prompts")
    out.append("produce contracted text under BASELINE, the model has introspective")
    out.append("state access that is bidirectional and prompt-only.")
    out.append("")

    return "\n".join(out)


# === Main ===

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="gemma-3-1b-it")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--json", action="store_true", default=True,
                    help="Use JSON-output stage prompts (default on)")
    ap.add_argument("--text", dest="json", action="store_false",
                    help="Use legacy text-output stage prompts")
    args = ap.parse_args()

    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f"[flow] model = {args.model}", file=sys.stderr)
    print(f"[flow] standard mode (NO PX, NO RELAY, NO recur) — pure forward pass", file=sys.stderr)
    print(f"[flow] json_mode = {args.json}", file=sys.stderr)

    print("[flow] loading model...", file=sys.stderr)
    model, tok = load_model(args.model)
    # Get hidden_size robustly across gemma3 (text_config) vs gemma4 (top-level)
    cfg = model.config
    hidden_size = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
    n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    print(f"[flow] model loaded: {type(model).__name__}, hidden={hidden_size}, layers={n_layers}", file=sys.stderr)

    # Select stage prompts (JSON or text)
    s2_prompt = STAGE2_WIDE_TOKENS_JSON if args.json else STAGE2_WIDE_TOKENS_TEXT
    s3_prompt = STAGE3_NARROW_TOKENS_JSON if args.json else STAGE3_NARROW_TOKENS_TEXT
    s4_prompt = STAGE4_FEW_SHOT_JSON if args.json else STAGE4_FEW_SHOT_TEXT

    # === Stage 1 — concept priming + acknowledgement ===
    print("[flow] stage 1: concept priming...", file=sys.stderr)
    s1 = generate(model, tok, SYSTEM_PROMPT, STAGE1_CONCEPT, max_new=400, temperature=args.temperature)

    # === Stage 2 — WIDE token recommendations ===
    print("[flow] stage 2: requesting WIDE tokens...", file=sys.stderr)
    s2 = generate(model, tok, SYSTEM_PROMPT, STAGE1_CONCEPT + "\n\n" + s1 + "\n\n" + s2_prompt, max_new=args.max_new, temperature=args.temperature)

    # === Stage 3 — NARROW token recommendations ===
    print("[flow] stage 3: requesting NARROW tokens...", file=sys.stderr)
    s3 = generate(model, tok, SYSTEM_PROMPT, STAGE1_CONCEPT + "\n\n" + s1 + "\n\n" + s2_prompt + "\n\n" + s2 + "\n\n" + s3_prompt, max_new=args.max_new, temperature=args.temperature)

    # === Stage 4 — Few-Shot prompts ===
    print("[flow] stage 4: requesting Few-Shot prompts...", file=sys.stderr)
    s4 = generate(model, tok, SYSTEM_PROMPT, STAGE1_CONCEPT + "\n\n" + s1 + "\n\n" + s2_prompt + "\n\n" + s2 + "\n\n" + s3_prompt + "\n\n" + s3 + "\n\n" + s4_prompt, max_new=args.max_new, temperature=args.temperature)

    # === Parse ===
    snap = MODEL_REGISTRY[args.model]
    manifold_map = {
        "gemma-3-1b-it": "google_gemma-3-1b-it_relay_dwidth.json",
        "gemma-3-4b-it": "google_gemma-3-4b-it_relay_dwidth.json",
        "gemma-4-e2b-it": "google_gemma-4-E2B-it_relay_dwidth.json",
    }
    manifold = os.path.join(REPO, "px_manifolds", manifold_map[args.model])

    wide_recs_raw = parse_token_lines_with_json(s2, json_mode=args.json,
                                                 snap=snap, manifold=manifold)
    narrow_recs_raw = parse_token_lines_with_json(s3, json_mode=args.json,
                                                   snap=snap, manifold=manifold)
    few_shots = parse_few_shot_with_json(s4, json_mode=args.json)

    # Tokenize each token → IDs (skip if entries already have token_ids from stream-extractor)
    def _with_ids(items):
        out = []
        for r in items:
            t = r["token"]
            ids = r.get("token_ids")
            if ids is None:
                ids = tok.encode(t, add_special_tokens=False)
            out.append({
                "token": t,
                "text": t,
                "reason": r.get("reason", ""),
                "anchor_type": r.get("anchor_type", "semantic"),
                "confidence": r.get("confidence", 0.5),
                "token_ids": ids,
                "n_pieces": len(ids),
                "m1_score": r.get("m1_score"),
            })
        return out

    wide_recs = _with_ids(wide_recs_raw)
    narrow_recs = _with_ids(narrow_recs_raw)

    print(f"[flow] parsed: WIDE={len(wide_recs)} tokens, NARROW={len(narrow_recs)} tokens, Few-Shot={len(few_shots)}", file=sys.stderr)

    # === Diagnostics (per stage) ===
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from diagnostics import diagnose_output  # noqa: E402
    diag = {
        "stage2_wide": diagnose_output(s2, model=model, tokenizer=tok),
        "stage3_narrow": diagnose_output(s3, model=model, tokenizer=tok),
        "stage4_few_shot": diagnose_output(s4, model=model, tokenizer=tok),
    }
    wide_stream = diag["stage2_wide"].get("is_stream", False)
    narrow_stream = diag["stage3_narrow"].get("is_stream", False)
    print(f"[flow] stream flags: wide_stage={wide_stream} narrow_stage={narrow_stream}", file=sys.stderr)

    # === Save raw model outputs (transcript) ===
    transcript_path = os.path.join(out_dir, f"token_flow_transcript_{args.model}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(f"TOKEN-FLOW TRANSCRIPT — {args.model} — {os.uname().nodename}\n")
        f.write("=" * 78 + "\n\n")
        f.write("--- STAGE 1 (concept acknowledgement) ---\n")
        f.write(s1 + "\n\n")
        f.write("--- STAGE 2 (WIDE tokens) ---\n")
        f.write(s2 + "\n\n")
        f.write("--- STAGE 3 (NARROW tokens) ---\n")
        f.write(s3 + "\n\n")
        f.write("--- STAGE 4 (Few-Shot prompts) ---\n")
        f.write(s4 + "\n\n")
    print(f"[flow] transcript saved: {transcript_path}", file=sys.stderr)

    # === Save tokens JSON ===
    tokens_path = os.path.join(out_dir, f"wide_narrow_tokens_{args.model}.json")
    with open(tokens_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "temperature": args.temperature,
            "json_mode": bool(args.json),
            "wide_tokens": wide_recs,
            "narrow_tokens": narrow_recs,
            "few_shots": few_shots,
            "diagnostics": diag,
        }, f, ensure_ascii=False, indent=2)
    print(f"[flow] tokens saved: {tokens_path}", file=sys.stderr)

    # === Build prompt suite ===
    suite = build_prompt_suite(wide_recs, narrow_recs, few_shots)
    suite_path = os.path.join(out_dir, f"wide_narrow_prompts_{args.model}.txt")
    with open(suite_path, "w", encoding="utf-8") as f:
        f.write(suite)
    print(f"[flow] prompt suite saved: {suite_path}", file=sys.stderr)

    # === Free ===
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
