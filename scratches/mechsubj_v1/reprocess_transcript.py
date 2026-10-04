"""reprocess_transcript.py — Wendet den Parser auf ein gespeichertes
token_flow-Transcript an und baut die Prompt-Suite neu. JSON-Modus optional.

Mit --json (default an) wird der JSON-Parser mit multilingual-stream-Fallback
verwendet. Mit --text wird der alte Text-Parser benutzt.
"""
import argparse
import json
import os
import re
import sys

REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")

# Import parsing helpers from the original module
sys.path.insert(0, OUT_DIR)
from token_flow_v1 import (  # type: ignore  # noqa: E402
    parse_token_lines_with_json,
    parse_few_shot_with_json,
    build_prompt_suite,
)
from transformers import AutoTokenizer  # noqa: E402

SNAP_1B = "/home/julian/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
SNAP_4B = "/home/julian/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
SNAP_E2B = "/home/julian/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03"

MANIFOLD = {
    "gemma-3-1b-it": os.path.join(REPO, "px_manifolds/google_gemma-3-1b-it_relay_dwidth.json"),
    "gemma-3-4b-it": os.path.join(REPO, "px_manifolds/google_gemma-3-4b-it_relay_dwidth.json"),
    "gemma-4-e2b-it": os.path.join(REPO, "px_manifolds/google_gemma-4-E2B-it_relay_dwidth.json"),
}


def split_sections(transcript: str):
    """Returns dict {stage_name: section_text}."""
    sections = {}
    current = None
    buf = []
    for line in transcript.splitlines():
        m = re.match(r"^---\s*STAGE\s*\d+\s*\((.+)\)\s*---", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        else:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def tokenize_recs(recs, tok):
    out = []
    for r in recs:
        text = r["token"]
        ids = r.get("token_ids")
        if ids is None:
            ids = tok.encode(text, add_special_tokens=False)
        out.append({
            **r,
            "text": text,
            "token_ids": ids,
            "n_pieces": len(ids),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MANIFOLD.keys()))
    ap.add_argument("--snap", default=None)
    ap.add_argument("--json", dest="use_json", action="store_true", default=True)
    ap.add_argument("--text", dest="use_json", action="store_false")
    args = ap.parse_args()

    snap = args.snap or (
        SNAP_1B if args.model == "gemma-3-1b-it" else
        SNAP_4B if args.model == "gemma-3-4b-it" else
        SNAP_E2B
    )
    manifold = MANIFOLD[args.model]

    transcript_path = os.path.join(OUT_DIR, f"token_flow_transcript_{args.model}.txt")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    sections = split_sections(transcript)

    s2 = sections.get("WIDE tokens", "")
    s3 = sections.get("NARROW tokens", "")
    s4 = sections.get("Few-Shot prompts", "")

    print(f"[reprocess] model={args.model} transcript={len(transcript)} bytes json_mode={args.use_json}", file=sys.stderr)
    print(f"[reprocess] STAGE2 length={len(s2)} STAGE3={len(s3)} STAGE4={len(s4)}", file=sys.stderr)

    wide_raw = parse_token_lines_with_json(s2, json_mode=args.use_json,
                                            snap=snap, manifold=manifold)
    narrow_raw = parse_token_lines_with_json(s3, json_mode=args.use_json,
                                              snap=snap, manifold=manifold)
    few_shots = parse_few_shot_with_json(s4, json_mode=args.use_json)

    print(f"[reprocess] parsed: WIDE={len(wide_raw)} NARROW={len(narrow_raw)} FS={len(few_shots)}", file=sys.stderr)
    for r in wide_raw[:5]:
        print(f"  WIDE  : {r.get('token','')!r:30s}  type={r.get('anchor_type','?')}", file=sys.stderr)
    for r in narrow_raw[:5]:
        print(f"  NARROW: {r.get('token','')!r:30s}  type={r.get('anchor_type','?')}", file=sys.stderr)
    for fs in few_shots[:6]:
        print(f"  FS    : {fs.get('direction')} {fs.get('prompt','')[:80]!r}", file=sys.stderr)

    print("[reprocess] loading tokenizer (small, no model)…", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(snap)

    wide_full = tokenize_recs(wide_raw, tok)
    narrow_full = tokenize_recs(narrow_raw, tok)

    # Diagnostics (lightweight, no model needed)
    sys.path.insert(0, OUT_DIR)
    from diagnostics import diagnose_output  # noqa: E402
    diag = {
        "stage2_wide": diagnose_output(s2),
        "stage3_narrow": diagnose_output(s3),
        "stage4_few_shot": diagnose_output(s4),
    }

    tokens_path = os.path.join(OUT_DIR, f"wide_narrow_tokens_{args.model}.json")
    with open(tokens_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "json_mode": bool(args.use_json),
            "wide_tokens": wide_full,
            "narrow_tokens": narrow_full,
            "few_shots": few_shots,
            "diagnostics": diag,
            "source": "model self-recommendation (BASELINE, no PX, no RELAY, no recur)",
        }, f, ensure_ascii=False, indent=2)
    print(f"[reprocess] tokens saved: {tokens_path}", file=sys.stderr)

    suite = build_prompt_suite(wide_full, narrow_full, few_shots)
    suite_path = os.path.join(OUT_DIR, f"wide_narrow_prompts_{args.model}.txt")
    with open(suite_path, "w", encoding="utf-8") as f:
        f.write(suite)
    print(f"[reprocess] prompt suite saved: {suite_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
