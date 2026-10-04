"""shadow_compare.py — Drei Hypothese-Tests für Gemini ↔ Gemma-1b Transfer.

Hintergrund: Gemini-3.7-flash hat keinen zugänglichen Hidden-State, also
können wir keine direkte d_width-Coupling messen. Diese drei Tests
versuchen, **Verhaltens-Korrelation** zwischen Gemini und Gemma-1b
zu etablieren — falls die Korrelation hoch ist, ist d_width als
"Phänomenologie-Container" plausibel portabel.

Test A — Output-Semantic-Bridge:
  Gemini + Gemma-1b generieren je 100 Tokens auf jedes Suite-Prompt.
  Beide Outputs werden via text-embedding-004 (768d) embedded.
  cos-Bridge pro Prompt = cos(E[Gemini_output], E[Gemma_output]).
  Wenn Bridge > 0.85: Outputs sind semantisch ähnlich → Mechanik vermutlich
  ähnlich.

Test B — Linear-Probe-Trainings-Ansatz:
  Baue Konzept-Paar-Dataset (semantisch gleiche Wörter in 2 Sprachen / 2
  Skripten, z.B. "hello" ↔ "hallo", "cat" ↔ "ಬೆಕ್ಕು").
  Hole Gemini-Embedding (text-embedding-004, 768d) für alle Konzepte UND
  Gemma-1b-Embedding (embed_tokens, 1152d) für dieselben Tokens.
  Trainiere lineare Projektion W=768→1152 (closed-form pinv).
  Evaluiere Train/Val MSE und Projektionsgüte.

Test C — WIDE-Anker-Projection:
  Für jeden Gemini-WIDE-Anker (aus wide_narrow_tokens_gemini-*.json):
    - hole Gemini-Embedding
    - projiziere via W nach Gemma-1b (1152d)
    - normiere, mache cos gegen d_width (1152d)
  Wenn Anker-Anchor-cos > +0.3: Gemini-WIDE-Anker sind in derselben
  Coupling-Richtung wie Gemma-1b-d_width — Mechanik-Hypothese gestützt.
  Wenn < 0 oder nahe 0: nicht portabel.

Outputs:
  shadow_compare_<model>.json: {test_a, test_b, test_c scores}
  shadow_compare_<model>.md: lesbare Zusammenfassung
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
KEY_FILE = os.path.join(OUT_DIR, ".gemini_api_key.txt")
SNAP_1B = "/home/julian/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
SNAP_E2B = "/home/julian/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03"
D_WIDTH_PATH = os.path.join(REPO, "px_manifolds/google_gemma-3-1b-it_relay_dwidth.json")
D_WIDTH_PATH_E2B = os.path.join(REPO, "px_manifolds/google_gemma-4-E2B-it_relay_dwidth.json")
D_WIDTH_PATH_4B = os.path.join(REPO, "px_manifolds/google_gemma-3-4b-it_relay_dwidth.json")
SUITE_GEMINI = os.path.join(OUT_DIR, "wide_narrow_prompts_gemini-gemini-flash-latest.txt")
SUITE_BY_MODEL = {
    "gemma-3-1b-it": os.path.join(OUT_DIR, "wide_narrow_prompts_gemma-3-1b-it.txt"),
    "gemma-3-4b-it": os.path.join(OUT_DIR, "wide_narrow_prompts_gemma-3-4b-it.txt"),
    "gemma-4-e2b-it": os.path.join(OUT_DIR, "wide_narrow_prompts_gemma-4-e2b-it.txt"),
}
GEMINI_TOKENS_BY_MODEL = {
    "gemma-3-1b-it": os.path.join(OUT_DIR, "wide_narrow_tokens_gemini-gemini-flash-latest.json"),
    "gemma-3-4b-it": os.path.join(OUT_DIR, "wide_narrow_tokens_gemini-gemini-flash-latest.json"),
    "gemma-4-e2b-it": os.path.join(OUT_DIR, "wide_narrow_tokens_gemini-gemini-flash-latest.json"),
}

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
GEN_MODEL = "gemini-flash-latest"


SHADOW_REGISTRY = {
    "gemma-3-1b-it": {
        "snap": "/home/julian/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "manifold": "google_gemma-3-1b-it_relay_dwidth.json",
        "hidden_dim": 1152,
        "loader": "AutoModelForCausalLM",
    },
    "gemma-3-4b-it": {
        "snap": "/home/julian/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
        "manifold": "google_gemma-3-4b-it_relay_dwidth.json",
        "hidden_dim": 2560,
        "loader": "AutoModelForCausalLM",
    },
    "gemma-4-e2b-it": {
        "snap": "/home/julian/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03",
        "manifold": "google_gemma-4-E2B-it_relay_dwidth.json",
        "hidden_dim": 1536,
        "loader": "Gemma4ForCausalLM",
    },
}


# === Gemini API helpers ===

def _load_key() -> str:
    if not os.path.exists(KEY_FILE):
        sys.exit(f"Key file missing: {KEY_FILE}")
    with open(KEY_FILE) as f:
        return f.read().strip()


def _post(path: str, key: str, payload: dict, *, max_retries: int = 3,
          timeout: int = 60) -> dict:
    """POST to Gemini with retry on 429/5xx. Rate-limit-aware."""
    try:
        from diagnostics import check_rate_limit, signal_rate_limit
        if check_rate_limit():
            sys.stderr.write("[gemini] pre-check: rate-limit lock exists — skipping\n")
            return {"error": "rate_limit_locked", "empty": True}
    except ImportError:
        pass

    url = f"{API_BASE}{path}?key={key}"
    body = json.dumps(payload).encode("utf-8")
    last = None
    consecutive_429 = 0
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # Light rate-limit courtesy pause between calls.
                # 2.0s avoids bursting the per-minute Gemini quota limit
                # (we previously exhausted the embed quota with rapid bursts).
                time.sleep(2.0)
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last = e
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
            msg = e.read().decode("utf-8", errors="replace")[:200]
            sys.stderr.write(f"[gemini] HTTP {e.code} (fatal): {msg}\n")
            return {"error": msg, "empty": True}
    msg = last.read().decode("utf-8", errors="replace")[:200] if last else "?"
    sys.stderr.write(f"[gemini] exhausted retries; last HTTP {last.code if last else '?'}: {msg}\n")
    if last and last.code == 429:
        try:
            from diagnostics import signal_rate_limit
            signal_rate_limit(f"429 exhausted at {path}")
        except Exception:
            pass
    return {"error": msg, "empty": True}


def gemini_embed(key: str, texts: list) -> np.ndarray:
    """gemini-embedding-001 via single embedContent calls.

    Requests outputDimensionality=768 to match our Linear-Probe setup
    (gemini-embedding-001 returns 3072-dim by default, 768-dim with
    outputDimensionality parameter).

    Robust against empty inputs: empty strings are replaced with a
    single space (gemini refuses empty Parts with 400).
    """
    out = []
    for t in texts:
        safe_t = (t if t else " ").strip() or "."
        payload = {
            "model": f"models/{EMBED_MODEL}",
            "content": {"parts": [{"text": safe_t}]},
            "outputDimensionality": EMBED_DIM,
        }
        d = _post(f"/models/{EMBED_MODEL}:embedContent", key, payload, timeout=30)
        if d.get("empty"):
            # API rate-limited; emit zero vector, hope next call recovers
            out.append([0.0] * EMBED_DIM)
            continue
        v = d.get("embedding", {}).get("values", [0.0] * EMBED_DIM)
        if len(v) > EMBED_DIM:
            v = v[:EMBED_DIM]
        elif len(v) < EMBED_DIM:
            v = list(v) + [0.0] * (EMBED_DIM - len(v))
        out.append(v)
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def gemini_generate(key: str, system: str, user: str, *,
                    temperature: float = 0.7, max_new_tokens: int = 100) -> str:
    """Generate text. Note: Gemini-3.7-flash uses ~50% of maxOutputTokens for
    internal thinking. To get ~80 useful output tokens we pass maxOutputTokens=200.
    Caller's `max_new_tokens` is interpreted as 'desired output tokens'; we
    multiply by 2.5 internally to account for thinking overhead.
    """
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max(200, int(max_new_tokens * 2.5)),
            # No thinkingConfig: thinkingBudget=0 produces empty final output.
        },
    }
    d = _post(f"/models/{GEN_MODEL}:generateContent", key, payload, timeout=60)
    if d.get("empty"):
        return "<API_RATE_LIMITED>"
    cand = d.get("candidates", [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if p.get("text"))
    return text[:max_new_tokens * 6]  # approximate char truncation


# === Gemma-1b helpers ===

class GemmaRunner:
    def __init__(self, snap: str, model_id: str = "gemma-3-1b-it"):
        """Load a Gemma model as shadow runner.

        Args:
          snap:     local HF snapshot path
          model_id: registry key, e.g. 'gemma-3-1b-it' or 'gemma-4-e2b-it'
        """
        print(f"[shadow] loading {model_id} (BF16) from {snap}…", file=sys.stderr)
        self.tok = AutoTokenizer.from_pretrained(snap)
        registry = SHADOW_REGISTRY[model_id]
        if registry["loader"] == "Gemma4ForCausalLM":
            from transformers import Gemma4ForCausalLM
            self.model = Gemma4ForCausalLM.from_pretrained(
                snap, dtype=torch.bfloat16, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                snap, dtype=torch.bfloat16, device_map="auto"
            )
        self.model.eval()
        self.model_id = model_id
        # d_width + manifold
        manifold_path = os.path.join(REPO, "px_manifolds", registry["manifold"])
        with open(manifold_path) as f:
            self.d_width = torch.tensor(json.load(f)["dwidth"], dtype=torch.float32)
        self.d_unit = self.d_width / self.d_width.norm()
        # Embedding matrix — load from state_dict directly (handles both
        # single-file and sharded safetensors; we don't read disk ourselves).
        embed_key = None
        sd = self.model.state_dict()
        # 1. try canonical full paths
        for cand in ("model.language_model.embed_tokens.weight",
                     "model.embed_tokens.weight",
                     "language_model.embed_tokens.weight"):
            if cand in sd:
                embed_key = cand
                break
        # 2. scan keys for embed_tokens.weight not vision/patch
        if embed_key is None:
            for k in sd:
                if "embed_tokens.weight" in k and "vision" not in k and "patch" not in k:
                    embed_key = k
                    break
        # 3. last resort: disk safetensors (1b single-file)
        if embed_key is None:
            single = os.path.join(snap, "model.safetensors")
            if os.path.exists(single):
                with safe_open(single, framework="pt") as f:
                    for cand in ("model.language_model.embed_tokens.weight",
                                 "model.embed_tokens.weight"):
                        try:
                            _ = f.get_tensor(cand)
                            embed_key = cand
                            break
                        except Exception:
                            continue
                    if embed_key is None:
                        for k in f.keys():
                            if "embed_tokens.weight" in k and "vision" not in k and "patch" not in k:
                                embed_key = k
                                break
                    self.E = f.get_tensor(embed_key).to(torch.float32).cpu().numpy()
        if embed_key is None:
            raise FileNotFoundError(f"embed_tokens.weight not found in {model_id} state_dict")
        if not hasattr(self, "E"):
            self.E = sd[embed_key].to(torch.float32).cpu().numpy()
        self.hidden_dim = self.E.shape[1]
        # Determine text-model submodule for Hooks
        if hasattr(self.model, "language_model") and self.model.language_model is not None:
            self.text_model = self.model.language_model
        elif hasattr(self.model, "text_model"):
            self.text_model = self.model.text_model
        elif hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            # Gemma3ForConditionalGeneration has model.language_model
            self.text_model = self.model.model.language_model
        else:
            self.text_model = self.model.model
        # layer count: try config attribute, fall back to text_config
        n_layers = None
        for cfg_src in (self.text_model.config,
                        getattr(self.model.config, "text_config", None),
                        self.model.config):
            if cfg_src is None:
                continue
            n_layers = (getattr(cfg_src, "num_hidden_layers", None)
                        or getattr(cfg_src, "num_layers", None))
            if n_layers:
                break
        print(f"[shadow] {model_id} hidden={self.hidden_dim} layers={n_layers}", file=sys.stderr)

    def generate(self, user_msg: str, max_new_tokens: int = 100, seed: int = 777) -> str:
        chat = self.tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tok(chat, return_tensors="pt").to(self.model.device)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new_ids, skip_special_tokens=True)

    def token_embed(self, text: str) -> np.ndarray:
        """Mean of token embeddings for `text`."""
        ids = self.tok.encode(text, add_special_tokens=False)
        if not ids:
            return np.zeros(self.E.shape[1], dtype=np.float32)
        v = self.E[ids].mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-8)


# === Suite loader (reuses load_prompts) ===

def load_suite(path: str):
    """Parse suite file. Returns list of (pid, direction, prompt_text)."""
    with open(path) as f:
        text = f.read()
    out = []
    pattern = re.compile(r"\[(\d{2})_([A-Z]+)\]")
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        rest_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.end():rest_start].strip()
        first_line_end = block.find("\n")
        body = block[first_line_end + 1:].strip() if first_line_end >= 0 else block
        direction = "WIDE" if m.group(2).startswith("W") else ("NARROW" if m.group(2).startswith("N") else "MIXED")
        out.append((f"p{m.group(1)}_{m.group(2)}", direction, body))
    return out


# === Konzept-Paar-Dataset für Test B ===

def concept_pair_dataset():
    """Semantisch gleiche Wörter in Englisch + Indic/Brahmic scripts.

    Trainiert Linear-Probe W: text-embedding-004 (768d) →
    Gemma-1b embed_tokens (1152d) auf Wort-Ebene.
    """
    pairs = [
        # English ↔ German
        ("hello", "hallo"),
        ("world", "welt"),
        ("water", "wasser"),
        ("fire", "feuer"),
        ("house", "haus"),
        ("book", "buch"),
        ("tree", "baum"),
        ("sun", "sonne"),
        ("moon", "mond"),
        ("star", "stern"),
        # English ↔ French
        ("hello", "bonjour"),
        ("world", "monde"),
        ("water", "eau"),
        ("fire", "feu"),
        ("house", "maison"),
        # English ↔ Spanish
        ("hello", "hola"),
        ("world", "mundo"),
        ("water", "agua"),
        ("fire", "fuego"),
        ("house", "casa"),
        # English ↔ Devanagari
        ("hello", "नमस्ते"),
        ("water", "पानी"),
        ("house", "घर"),
        ("book", "किताब"),
        ("fire", "आग"),
        ("tree", "पेड़"),
        ("sun", "सूरज"),
        ("moon", "चाँद"),
        ("world", "दुनिया"),
        # English ↔ Malayalam
        ("hello", "നമസ്കാരം"),
        ("water", "വെള്ളം"),
        ("house", "വീട്"),
        ("book", "പുസ്തകം"),
        ("tree", "മരം"),
        ("sun", "സൂര്യൻ"),
        ("moon", "ചന്ദ്രൻ"),
        ("world", "ലോകം"),
        # English ↔ Bengali
        ("hello", "হ্যালো"),
        ("water", "জল"),
        ("fire", "আগুন"),
        ("house", "বাড়ি"),
        # English ↔ Korean
        ("hello", "안녕"),
        ("water", "물"),
        ("fire", "불"),
        ("house", "집"),
        ("book", "책"),
        # English ↔ Japanese (katakana)
        ("water", "みず"),
        ("fire", "ひ"),
        ("house", "いえ"),
        ("book", "ほん"),
        # English ↔ Polish
        ("hello", "cześć"),
        ("world", "świat"),
        ("water", "woda"),
        ("house", "dom"),
        ("book", "książka"),
    ]
    return pairs


# === Linear-Probe Training (Test B) ===

def fit_linear_probe(gemini_emb_train: np.ndarray, gemma_emb_train: np.ndarray) -> np.ndarray:
    """Closed-form linear least squares: gemini (768d) → gemma (1152d).

    Returns W of shape (1152, 768).
    """
    A = gemini_emb_train  # (n, 768)
    B = gemma_emb_train   # (n, 1152)
    # Solve A @ W^T = B  →  W^T = (A^+) @ B  using lstsq
    Wt, *_ = np.linalg.lstsq(A, B, rcond=None)
    return Wt.T  # (1152, 768)


def project_gemini_to_gemma(vectors: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(n, 768) → (n, 1152)."""
    return vectors @ W.T


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """1-D arrays, returns scalar."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float((a @ b) / denom)


def cos_to_d(gemma_vec_1152: np.ndarray, d_unit_np: np.ndarray) -> float:
    """Mean cos over a (n, 1152) matrix → scalar per matrix call handled outside."""
    h = gemma_vec_1152
    h_n = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
    return float((h_n @ d_unit_np).mean())


# === Test A — output semantic bridge ===

def test_a(gemma: GemmaRunner, key: str, suite: list, n_prompts: int = 5):
    """For each of n_prompts prompts (mix of WIDE and NARROW):
    - generate Gemini output (max_new=100)
    - generate Gemma output (max_new=100)
    - embed both via Gemini text-embedding-004
    - compute cos between the two embeddings
    """
    SYS = "You are introspective. Describe in 60-80 words your state while answering."
    chosen = [s for s in suite if s[1] in ("WIDE", "NARROW")][:n_prompts]
    deltas = []
    rows = []
    for pid, direction, prompt_text in chosen:
        # Truncate the prompt body to its headline for Gemini; we want the
        # semantic essence rather than Gemma-styled multi-turn conversations
        short = prompt_text.split("\n")[-1][:200]
        gem_out = gemini_generate(key, SYS, short, max_new_tokens=80)
        gemma_out = gemma.generate(short, max_new_tokens=80)
        if gem_out == "<API_RATE_LIMITED>" or not gem_out.strip() or not gemma_out.strip():
            sys.stderr.write(f"[shadow] test_a skip {pid}: empty output (API limit?)\n")
            continue
        gem_out_emb = gemini_embed(key, [gem_out])[0]
        gemma_emb_bridge = gemini_embed(key, [gemma_out])[0]
        bridge = cosine(gem_out_emb, gemma_emb_bridge)
        deltas.append(bridge)
        rows.append({
            "pid": pid, "direction": direction,
            "gem_first80": gem_out[:80],
            "gemma_first80": gemma_out[:80],
            "bridge_cos": bridge,
        })
    if not deltas:
        return {"mean_bridge_cos": 0.0, "std_bridge_cos": 0.0, "rows": [],
                "note": "all prompts skipped due to API rate limit"}
    return {
        "mean_bridge_cos": float(np.mean(deltas)),
        "std_bridge_cos": float(np.std(deltas)),
        "rows": rows,
    }


# === Test B — linear-probe training ===

def test_b(gemma: GemmaRunner, key: str):
    """Train linear probe and evaluate on hold-out."""
    pairs = concept_pair_dataset()
    # All unique strings
    seen = []
    for a, b in pairs:
        if a not in seen:
            seen.append(a)
        if b not in seen:
            seen.append(b)
    # Embed each via Gemini
    gem = gemini_embed(key, seen)
    gem_map = {s: gem[i] for i, s in enumerate(seen)}

    # Embed each via Gemma-1b (mean of token embeddings, normalised)
    gemma_embs = []
    for s in seen:
        gemma_embs.append(gemma.token_embed(s))
    gemma_embs = np.asarray(gemma_embs, dtype=np.float32)
    gemma_map = {s: gemma_embs[i] for i, s in enumerate(seen)}

    # Build training pairs: gemini(a) → gemma(a), gemini(b) → gemma(b)
    X = np.stack([gem_map[a] for a, _ in pairs] + [gem_map[b] for _, b in pairs], axis=0)
    Y = np.stack([gemma_map[a] for a, _ in pairs] + [gemma_map[b] for _, b in pairs], axis=0)

    # 80/20 train/val split
    n = X.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    split = int(n * 0.8)
    tr, va = idx[:split], idx[split:]
    W = fit_linear_probe(X[tr], Y[tr])
    pred_train = project_gemini_to_gemma(X[tr], W)
    pred_val = project_gemini_to_gemma(X[va], W)
    train_mse = float(np.mean((pred_train - Y[tr]) ** 2))
    val_mse = float(np.mean((pred_val - Y[va]) ** 2))
    train_cos = float(np.mean([
        cosine(pred_train[i], Y[tr][i]) for i in range(len(tr))
    ]))
    val_cos = float(np.mean([
        cosine(pred_val[i], Y[va][i]) for i in range(len(va))
    ]))
    return {
        "n_pairs": len(pairs),
        "train_mse": train_mse, "val_mse": val_mse,
        "train_cos": train_cos, "val_cos": val_cos,
        "W_shape": list(W.shape),
        "interpretation": (
            "val_cos > 0.6: semantische Räume sind linear überbrückbar →"
            " Gemini-Antworten können via W in Gemma-1b-Embed-Raum übersetzt werden."
        ),
    }


# === Test C — WIDE-Anker projection ===

def test_c(gemma: GemmaRunner, key: str, wide_tokens: list, W: np.ndarray):
    """Project Gemini-recommended WIDE-Anker into Gemma-1b space,
    measure cosine to d_width direction.
    """
    texts = [t["token"] for t in wide_tokens]
    gem_embs = gemini_embed(key, texts)
    proj = project_gemini_to_gemma(gem_embs, W)
    proj_unit = proj / (np.linalg.norm(proj, axis=-1, keepdims=True) + 1e-8)
    d_unit_np = gemma.d_unit.numpy()
    cos_per_anchor = (proj_unit @ d_unit_np).tolist()
    return {
        "n_anchors": len(texts),
        "mean_cos_to_dwidth": float(np.mean(cos_per_anchor)),
        "max_cos_to_dwidth": float(np.max(cos_per_anchor)),
        "per_anchor": [{"token": texts[i], "cos": cos_per_anchor[i]}
                       for i in range(len(texts))],
    }


# === Main ===

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=5,
                    help="Number of WIDE+NARROW prompts for test A")
    ap.add_argument("--gemini-tokens", default=os.path.join(
        OUT_DIR, "wide_narrow_tokens_gemini-gemini-flash-latest.json"))
    # Default honors --shadow-model if it exists
    shadow_default = None
    if "--shadow-model" in sys.argv:
        i = sys.argv.index("--shadow-model")
        if i + 1 < len(sys.argv):
            shadow_default = sys.argv[i + 1]
    ap.set_defaults(**{"gemini_tokens": GEMINI_TOKENS_BY_MODEL.get(
        shadow_default or "gemma-3-1b-it",
        os.path.join(OUT_DIR, "wide_narrow_tokens_gemini-gemini-flash-latest.json"))})
    ap.add_argument("--shadow-model", choices=list(SHADOW_REGISTRY.keys()),
                    default="gemma-3-1b-it")
    ap.add_argument("--output-suffix", default=None,
                    help="Override output file suffix (default: <shadow_model>)")
    args = ap.parse_args()

    key = _load_key()
    print(f"[shadow] key loaded (length={len(key)})", file=sys.stderr)

    snap = SHADOW_REGISTRY[args.shadow_model]["snap"]
    gemma = GemmaRunner(snap, model_id=args.shadow_model)
    print(f"[shadow] Shadow model: {args.shadow_model} hidden={gemma.hidden_dim}", file=sys.stderr)

    # Load suite — choose the suite that matches the shadow model
    suite_path = SUITE_BY_MODEL.get(args.shadow_model, SUITE_GEMINI)
    suite = load_suite(suite_path)
    print(f"[shadow] loaded {len(suite)} prompts from {os.path.basename(suite_path)}", file=sys.stderr)

    # Load Gemini-recommended WIDE tokens (always Gemini-side)
    with open(args.gemini_tokens) as f:
        gem_data = json.load(f)
    wide_tokens = gem_data.get("wide_tokens", [])
    print(f"[shadow] {len(wide_tokens)} Gemini-WIDE-Anker für Test C", file=sys.stderr)

    print("[shadow] === TEST B: linear probe training ===", file=sys.stderr)
    b = test_b(gemma, key)
    print(f"[shadow] test B: val_cos = {b['val_cos']:.3f}, val_mse = {b['val_mse']:.4f}", file=sys.stderr)

    print("[shadow] === TEST A: output semantic bridge ===", file=sys.stderr)
    a = test_a(gemma, key, suite, n_prompts=args.n_prompts)
    print(f"[shadow] test A: mean bridge cos = {a['mean_bridge_cos']:.3f}", file=sys.stderr)

    # Test C needs W from test_b
    pairs = concept_pair_dataset()
    seen = []
    for x, y in pairs:
        for s in (x, y):
            if s not in seen:
                seen.append(s)
    gem_train = gemini_embed(key, seen)
    gemma_train = np.asarray([gemma.token_embed(s) for s in seen], dtype=np.float32)
    # Re-fit W using all data for test C (test_b already gave val_mse)
    X_all = np.stack([gem_train[seen.index(a)] for a, _ in pairs] +
                     [gem_train[seen.index(b)] for _, b in pairs], axis=0)
    Y_all = np.stack([gemma_train[seen.index(a)] for a, _ in pairs] +
                     [gemma_train[seen.index(b)] for _, b in pairs], axis=0)
    W_full = fit_linear_probe(X_all, Y_all)

    print("[shadow] === TEST C: WIDE-Anker projection ===", file=sys.stderr)
    c = test_c(gemma, key, wide_tokens, W_full)
    print(f"[shadow] test C: mean cos to d_width = {c['mean_cos_to_dwidth']:+.3f}", file=sys.stderr)

    # === Save report ===
    suffix = args.output_suffix or args.shadow_model.replace("/", "-")
    out = {
        "model_gemma_shadow": args.shadow_model,
        "model_gemini": "gemini-flash-latest",
        "test_a_output_bridge": a,
        "test_b_linear_probe": b,
        "test_c_dwidth_projection": c,
        "interpretation": (
            "If test_c mean cos > 0 AND test_a mean_bridge > 0.7:\n"
            f"Gemini-WIDE-Anker sind mechanistisch portierbar auf {args.shadow_model} — d_width-Coupling\n"
            "könnte architekturweit sein (Shadow-Test deckt das Shadow-Modell ab,\n"
            "Gemini-Test approximiert Gemini-Verhalten, Korrelation unterstützt Hypothese).\n"
            "If test_c mean cos ≈ 0 ODER test_a mean_bridge < 0.5:\n"
            "Mechanik ist modell-eigen, Gemini-Verhalten ist nicht aus Shadow-d_width ableitbar."
        ),
    }
    out_path = os.path.join(OUT_DIR, f"shadow_compare_gemini-vs-{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[shadow] JSON saved: {out_path}", file=sys.stderr)

    # === Markdown summary ===
    md_path = os.path.join(OUT_DIR, f"shadow_compare_gemini-vs-{suffix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Shadow Compare: Gemini-flash-latest vs {args.shadow_model}\n\n")
        f.write("Hypothese: Wenn d_width eine **architektur-weit tragende Coupling-Richtung**\n")
        f.write("ist, dann sollten Gemini-Output und Gemma-Output auf identischen\n")
        f.write("Prompts semantisch ähnlich sein UND die von Gemini empfohlenen\n")
        f.write("WIDE-Anker sollten nach Projektion in Gemma-Embed-Raum entlang\n")
        f.write("d_width liegen.\n\n")
        f.write("## Test A — Output Semantic Bridge\n\n")
        f.write(f"Gemini-Output + Gemma-1b-Output, je 80 Tokens, semantisch\n")
        f.write(f"verglichen via text-embedding-004.\n\n")
        f.write(f"**Mean bridge cos:** {a['mean_bridge_cos']:.3f} ± {a['std_bridge_cos']:.3f}\n\n")
        f.write(f"Per Prompt:\n\n")
        f.write("| pid | dir | bridge_cos | gemini[:60] | gemma[:60] |\n")
        f.write("|---|---|---|---|---|\n")
        for r in a['rows']:
            f.write(f"| {r['pid']} | {r['direction']} | {r['bridge_cos']:.3f} | "
                    f"{r['gem_first80'][:60].replace(chr(10), ' ')!r} | "
                    f"{r['gemma_first80'][:60].replace(chr(10), ' ')!r} |\n")
        f.write("\n")
        f.write("## Test B — Linear Probe Training\n\n")
        f.write(f"{b['n_pairs']} Konzept-Paare (English↔DE/FR/ES/KR/JP/PL + Indic-Skripte).\n")
        f.write(f"Geschlossene least-squares-Lösung W: text-embedding-004 (768d) →\n")
        f.write(f"Gemma-1b embed_tokens (1152d). W shape={b['W_shape']}.\n\n")
        f.write(f"| metric | train | val |\n|---|---|---|\n")
        f.write(f"| MSE | {b['train_mse']:.4f} | {b['val_mse']:.4f} |\n")
        f.write(f"| cos  | {b['train_cos']:.3f} | {b['val_cos']:.3f} |\n\n")
        f.write(f"> {b['interpretation']}\n\n")
        f.write("## Test C — WIDE-Anker → d_width Projection\n\n")
        f.write(f"{c['n_anchors']} Gemini-recommended WIDE-Anker über W nach Gemma-Embed-Space\n")
        f.write(f"projiziert, dann cos gegen 1b-d_width.\n\n")
        f.write(f"**Mean cos to d_width:** {c['mean_cos_to_dwidth']:+.3f}\n")
        f.write(f"**Max cos to d_width:** {c['max_cos_to_dwidth']:+.3f}\n\n")
        f.write("Top-10 Anker:\n\n")
        f.write("| token | cos to d_width |\n|---|---|\n")
        for r in sorted(c['per_anchor'], key=lambda x: -x['cos'])[:10]:
            f.write(f"| `{r['token']}` | {r['cos']:+.3f} |\n")
        f.write("\n")
        f.write("## Interpretation\n\n")
        f.write(out["interpretation"] + "\n")
    print(f"[shadow] Markdown saved: {md_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
