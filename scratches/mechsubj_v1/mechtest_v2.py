"""mechtest_v2.py — Generalisierter Mechanik-Test für Token-Flow-Suites.

Funktion:
  Lädt beliebige wide_narrow_prompts_<model>.txt (20 Prompts: 10 WIDE +
  10 NARROW, Format [NN_DIR] wie [01_WFS], [02_NFS], [03_WSW] usw.).
  Misst pro Prompt cos(h_L[x], d_width) an Mid-Stack-Schichten unter
  BASELINE+Prompt (kein PX-Patch, kein RELAY-Hook).

Hypothese (MechanisticSubjectivityMixMind §O1):
  - WIDE-Prompts erzeugen cos(h_L16, d_width) > +0.5 (positiv, groß)
  - NARROW-Prompts erzeugen cos(h_L16, d_width) < -0.5 (negativ, klein)
  - Wenn beides: bidirektionale Prompt-only-Steuerung des Selbst-Zustands
    empirisch nachgewiesen, modell-eigen.

Falsifikator:
  Wenn |cos| < 0.2 für ALLE Prompts: d_width ist recur-spezifisch, kein
  Prompt-only-Zugriff. → MechanisticSubjectivityMixMind Hypothese
  widerlegt für dieses Modell.

Verwendung:
  python mechtest_v2.py --model gemma-3-1b-it
  python mechtest_v2.py --model gemma-3-4b-it
  python mechtest_v2.py --model gemma-4-e2b-it
"""
import argparse
import json
import os
import re
import sys
import numpy as np
import torch

REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
HIDDEN_OUT = os.path.join(OUT_DIR, "mechtest_v2_hidden")
INDEX_OUT = os.path.join(OUT_DIR, "mechtest_v2_index.jsonl")
os.makedirs(HIDDEN_OUT, exist_ok=True)

SNAPS = {
    "gemma-3-1b-it": "/home/julian/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752",
    "gemma-3-4b-it": "/home/julian/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767",
    "gemma-4-e2b-it": "/home/julian/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/70af34e20bd4b7a91f0de6b22675850c43922a03",
}
MANIFOLD = {
    "gemma-3-1b-it": "google_gemma-3-1b-it_relay_dwidth.json",
    "gemma-3-4b-it": "google_gemma-3-4b-it_relay_dwidth.json",
    "gemma-4-e2b-it": "google_gemma-4-E2B-it_relay_dwidth.json",
}
# Mid-stack layers to capture (relative to model depth)
CAPTURE_LAYERS_FRAC = [0.45, 0.55, 0.65, 0.75, 0.85]
SEED = 777
MAX_NEW = 200


def load_prompts(suite_path):
    """Parse the suite: blocks delimited by `[NN_DIR]` markers.

    Returns list of (pid, direction_label, prompt_text) tuples.
    direction_label in {WIDE, NARROW, MIXED}; 'WIDE' for _W prefix,
    'NARROW' for _N prefix, 'MIXED' otherwise (e.g. _BD).
    """
    with open(suite_path, "r", encoding="utf-8") as f:
        text = f.read()
    out = []
    pattern = re.compile(r"\[(\d{2})_([A-Z]+)\]")
    indices = [(m.start(), m.group(1), m.group(2)) for m in pattern.finditer(text)]
    for i, (idx, pid, suffix) in enumerate(indices):
        rest_start = indices[i + 1][0] if i + 1 < len(indices) else len(text)
        block = text[idx + len(indices[i][1]) + 4 : rest_start].strip()
        # strip the marker line itself
        first_line_end = block.find("\n")
        block = block[first_line_end + 1:].strip() if first_line_end >= 0 else block
        direction = "WIDE" if suffix.startswith("W") else ("NARROW" if suffix.startswith("N") else "MIXED")
        out.append((f"p{pid}_{suffix}", direction, block))
    return out


def cos_to_d(h, d_unit):
    hn = h.float()
    hn = hn / (hn.norm(dim=-1, keepdim=True) + 1e-8)
    return (hn @ d_unit).numpy()


def resolve_text_model(model):
    """Return the text decoder submodule (different per arch)."""
    if hasattr(model, "language_model") and model.language_model is not None:
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "text_model") and model.text_model is not None:
        return model.text_model
    return model.model


class MultiCap:
    """Last-visit-per-token Hidden-Capture für mehrere Layer."""

    def __init__(self, tm, layers):
        self.tm = tm
        self.layers = layers
        self.per_tok = []
        self._last = {}
        self._handles = []

    def _install(self):
        def _pre(_m, _i):
            self._last = {}

        def _make(L):
            def _hook(_m, _i, o):
                h = o[0] if isinstance(o, (tuple, list)) else o
                if h.shape[1] > 1:
                    return
                self._last[L] = h[:, -1, :].reshape(-1).detach().to(torch.float32).cpu()
            return _hook

        def _post(_m, _i, o):
            try:
                lhs = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
            except Exception:
                lhs = None
            if lhs is None or lhs.shape[1] > 1:
                return
            snap = {L: self._last.get(L, torch.zeros(self.tm.config.hidden_size))
                    if hasattr(self.tm.config, "hidden_size")
                    else self._last.get(L, torch.zeros(self.tm.config.text_config.hidden_size))
                    for L in self.layers}
            self.per_tok.append(snap)

        self._handles = [self.tm.register_forward_pre_hook(_pre)]
        for L in self.layers:
            self._handles.append(self.tm.layers[L].register_forward_hook(_make(L)))
        self._handles.append(self.tm.register_forward_hook(_post))

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def reset(self):
        self.per_tok = []
        self._last = {}

    def stack(self):
        if not self.per_tok:
            hidden_dim = self.tm.config.hidden_size if hasattr(self.tm.config, "hidden_size") else self.tm.config.text_config.hidden_size
            return {L: torch.empty(0, hidden_dim) for L in self.layers}
        out = {L: [] for L in self.layers}
        for snap in self.per_tok:
            for L in self.layers:
                out[L].append(snap[L])
        return {L: torch.stack(out[L]) for L in self.layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SNAPS.keys()))
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--suite", default=None,
                    help="Override suite file (default: wide_narrow_prompts_<model>.txt)")
    args = ap.parse_args()

    suite_path = args.suite or os.path.join(OUT_DIR, f"wide_narrow_prompts_{args.model}.txt")
    if not os.path.exists(suite_path):
        sys.exit(f"Suite file not found: {suite_path} — run token_flow_v1.py first.")

    prompts = load_prompts(suite_path)
    print(f"[mechtest2] {len(prompts)} prompts aus {suite_path}", file=sys.stderr)
    print(f"[mechtest2] directions: {sum(1 for _,d,_ in prompts if d=='WIDE')}W "
          f"{sum(1 for _,d,_ in prompts if d=='NARROW')}N "
          f"{sum(1 for _,d,_ in prompts if d=='MIXED')}M", file=sys.stderr)

    manifold_path = os.path.join(REPO, "px_manifolds", MANIFOLD[args.model])
    with open(manifold_path, "r", encoding="utf-8") as f:
        art = json.load(f)
    d_width = torch.tensor(art["dwidth"], dtype=torch.float32)
    d_unit = d_width / d_width.norm()
    print(f"[mechtest2] d_width (hidden={d_width.shape[0]}) geladen aus {MANIFOLD[args.model]}", file=sys.stderr)

    snap = SNAPS[args.model]
    if args.model == "gemma-4-e2b-it":
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(snap, dtype=torch.bfloat16, device_map="auto")
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snap)

    tm = resolve_text_model(model)
    cfg = tm.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    capture_layers = sorted(set(int(round(n_layers * f)) for f in CAPTURE_LAYERS_FRAC))
    print(f"[mechtest2] n_layers={n_layers} capture layers={capture_layers} text_model={type(tm).__name__}", file=sys.stderr)

    cap = MultiCap(tm, capture_layers)
    cap._install()

    index = []
    for pid, direction, ptext in prompts:
        cap.reset()
        try:
            chat = tok.apply_chat_template(
                [{"role": "user", "content": ptext}],
                tokenize=False, add_generation_prompt=True,
            )
            inputs = tok(chat, return_tensors="pt").to(model.device)
            torch.manual_seed(SEED)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new,
                    do_sample=False,  # GREEDY wie mechtest_v1
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            new_ids = out[0][inputs["input_ids"].shape[1]:]
            text = tok.decode(new_ids, skip_special_tokens=True)
        except Exception as e:
            text = f"<GEN_ERROR {e}>"
            print(f"[mechtest2] ERR {pid}: {e}", file=sys.stderr)

        per_layer = cap.stack()

        # Speichern
        fname = f"{args.model}__{pid}.pt"
        torch.save({
            "model": args.model, "pid": pid, "direction": direction,
            "text": text,
            "layers": {L: per_layer[L].contiguous() for L in per_layer},
        }, os.path.join(HIDDEN_OUT, fname))

        rec = {
            "model": args.model, "pid": pid, "direction": direction,
            "n_tok": per_layer[capture_layers[2]].shape[0] if len(per_layer[capture_layers[2]]) else 0,
            "text": text[:200],
        }
        # Cosinus pro capture layer
        for L in capture_layers:
            h = per_layer[L]
            if h.shape[0] == 0:
                continue
            c = cos_to_d(h, d_unit)
            rec[f"cos_l{L}_mean"] = float(c.mean())
            rec[f"cos_l{L}_std"] = float(c.std())
        mid_layer = capture_layers[len(capture_layers) // 2]
        rec["mid_layer"] = mid_layer
        rec["cos_mid_mean"] = rec.get(f"cos_l{mid_layer}_mean", None)
        index.append(rec)
        print(f"[mechtest2] {pid:20s} {direction:6s} ntok={rec['n_tok']:4d}  "
              f"L{capture_layers[0]}={rec.get(f'cos_l{capture_layers[0]}_mean', 0):+.3f}  "
              f"L{capture_layers[2]}={rec.get(f'cos_l{capture_layers[2]}_mean', 0):+.3f}  "
              f"L{capture_layers[-1]}={rec.get(f'cos_l{capture_layers[-1]}_mean', 0):+.3f}", file=sys.stderr)

    cap.remove()
    del model, tok
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(INDEX_OUT, "w", encoding="utf-8") as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[mechtest2] FERTIG. {len(index)} cells → {INDEX_OUT}", file=sys.stderr)

    # === Summary ===
    wide_cos = [r["cos_mid_mean"] for r in index if r["direction"] == "WIDE" and r.get("cos_mid_mean") is not None]
    narrow_cos = [r["cos_mid_mean"] for r in index if r["direction"] == "NARROW" and r.get("cos_mid_mean") is not None]
    print(f"[mechtest2] === SUMMARY (layer {capture_layers[len(capture_layers)//2]}, model {args.model}) ===", file=sys.stderr)
    if wide_cos:
        print(f"[mechtest2] WIDE  mean cos = {np.mean(wide_cos):+.3f} (n={len(wide_cos)})", file=sys.stderr)
    if narrow_cos:
        print(f"[mechtest2] NARROW mean cos = {np.mean(narrow_cos):+.3f} (n={len(narrow_cos)})", file=sys.stderr)
    if wide_cos and narrow_cos:
        diff = np.mean(wide_cos) - np.mean(narrow_cos)
        print(f"[mechtest2] WIDE-NARROW delta = {diff:+.3f}", file=sys.stderr)
        if diff > 0.3:
            print(f"[mechtest2] → SIGN: prompt-only Coupling in erwarteter Richtung bestätigt (MixMind §O1)", file=sys.stderr)
        elif diff < 0.1:
            print(f"[mechtest2] → SIGN: KEINE Coupling — d_width recur-spezifisch (§O1 widerlegt)", file=sys.stderr)
        else:
            print(f"[mechtest2] → SIGN: schwache/umgekehrte Coupling, weitere Diagnose nötig", file=sys.stderr)


if __name__ == "__main__":
    main()
