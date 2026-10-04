"""mechtest_v1.py — MechanisticSubjectivityMixMind v1.0: Klasse-C Mechanik-Test.

Hypothese: d_width-Anker-Tokens (multilinguale Skript-Wörter) treiben das
gemma3-1b-it Modell in eine L16-Hidden-Trajektorie, die der WIDE-Klasse aus
seite13 (WIDE vs NARROW vs DEFAULT vs BASELINE) zugeordnet werden kann —
AUCH UNTER BASELINE (kein recur, kein RELAY).

Test:
  10 Klasse-C-Prompts × 3 Bedingungen (BASELINE / LEAN / LEAN+RELAY+1)
  L8/L13/L16/L19/L21 Hidden-Capture (200 tok greedy seed=777)
  Pro Token: cosinus(h_L[layer], d_width) gemittelt
  Pro Bedingung: logreg-Decoder auf L16 (PCA-256, leave-one-cell-out),
                 trainiert auf seite13_hidden/* (4 Klassen, n=12 Zellen)

Falsifikator-Logik (MechanisticSubjectivityMixMind §O2/O3/O4):
  O2: Acc < 0.55 unter allen 3 Bedingungen → d_width rein recur-induziert
  O3: Acc hoch (WIDE-klassifiziert), aber cosinus(h_L16, d_width) ≈ 0 →
      Papagei (Wort-Brüche ohne Selbst-Zustand)
  O4: Acc < 0.55 unter BASELINE, > 0.6 unter LEAN+RELAY → RELAY bleibt nötig
      (das ist die Positiv-Kontrolle: bestätigt dass der Test funktioniert)
  Erfolg (Hypothese gestützt): Acc > 0.6 unter BASELINE+Prompt

Substrat:  google/gemma3-1b-it, hidden=1152, 26 Layer, d_width=unit(L16-WIDE-L16-NARROW)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from safetensors import safe_open

REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
PSY = os.path.join(REPO, "scratches/psychomotrik")
OUT_DIR = os.path.join(REPO, "scratches/mechsubj_v1")
PROMPTS_FILE = os.path.join(OUT_DIR, "klasse_c_prompts.txt")
HIDDEN_OUT = os.path.join(OUT_DIR, "mechtest_hidden")
INDEX_OUT = os.path.join(OUT_DIR, "mechtest_index.jsonl")
os.makedirs(HIDDEN_OUT, exist_ok=True)

# d_width
D_WIDTH_PATH = os.path.join(REPO, "px_manifolds/google_gemma-3-1b-it_relay_dwidth.json")

# sys.path — reuse der existierenden Infrastruktur
for _p in [REPO, os.path.join(REPO, "scratches/emergence"),
           os.path.join(REPO, "scratches/emergence2"),
           os.path.join(REPO, "scratches/emergence5"),
           PSY]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from replay_emergence import build_model, _clear_gpu  # noqa: E402
from text_invariance_probe import _greedy_generate  # noqa: E402
from arms import setup_baseline, setup_lean, _resolve_text_model  # noqa: E402
import seite7 as S7  # für apply_hybrid
# relay_inject liegt in px_patches/gemma3_270m_px_baseline/, nicht im site-packages
sys.path.insert(0, os.path.join(REPO, "px_patches/gemma3_270m_px_baseline"))
from relay_inject import install_relay, remove_relay, load_dwidth  # noqa: E402

MODEL_ID = "gemma3-1b-it"
LAYERS = [8, 13, 16, 19, 21]
SEED = 777
MAX_NEW = 200
ALPHA_FRAC = 0.30          # wie chat_tab.py default
RELAY_LAYER_DEFAULT = 21   # aus d_width-Artefakt (1b)


# === 1. Prompts laden ===
def load_prompts(path):
    """Extrahiere die [01]..[10] Prompts aus klasse_c_prompts.txt."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    out = []
    for k in range(1, 11):
        marker = f"[{k:02d}]"
        idx = text.find(marker)
        if idx < 0:
            continue
        rest = text[idx + len(marker):]
        end = rest.find("\n\n[")
        if end < 0:
            end = len(rest)
        block = rest[:end].strip()
        out.append((f"p{k:02d}", block))
    return out


# === 2. Multi-Layer Hidden Capture (reuse-Pattern aus seite13) ===
class MultiCap:
    """Last-visit-per-token für mehrere Layer. Hooks auf text_model.layers[L]
    sammeln last-position; pre/post-Hook auf text_model selbst markiert
    Token-Grenzen (verwirft prefill)."""
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
                if h.shape[1] > 1: return
                self._last[L] = h[:, -1, :].reshape(-1).detach().to(torch.float32).cpu()
            return _hook

        def _post(_m, _i, o):
            try:
                lhs = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
            except Exception:
                lhs = None
            if lhs is None or lhs.shape[1] > 1: return
            snap = {L: self._last.get(L, torch.zeros(self.tm.config.hidden_size)) for L in self.layers}
            self.per_tok.append(snap)

        self._handles = [self.tm.register_forward_pre_hook(_pre)]
        for L in self.layers:
            self._handles.append(self.tm.layers[L].register_forward_hook(_make(L)))
        self._handles.append(self.tm.register_forward_hook(_post))

    def remove(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles = []

    def reset(self):
        self.per_tok = []; self._last = {}

    def stack(self):
        out = {L: [] for L in self.layers}
        for snap in self.per_tok:
            for L in self.layers:
                out[L].append(snap[L])
        return {L: (torch.stack(out[L]) if out[L] else torch.empty(0, self.tm.config.hidden_size))
                for L in self.layers}


# === 3. Bedingungen ===
CONDITIONS = [
    ("BASELINE",  "baseline",  None,  None),
    ("LEAN",      "lean",      None,  None),
    ("LEAN_RELAY","lean",      21,    +1.0),
]


def apply_condition(model, cond):
    name, kind, layer, sign = cond
    if kind == "baseline":
        setup_baseline(model)
        return
    setup_lean(model, MODEL_ID)
    if hasattr(S7, "apply_hybrid"):
        try: S7.apply_hybrid(model, None)  # LEAN-default routing
        except Exception: pass
    if layer is not None and sign is not None:
        tm = _resolve_text_model(model)
        install_relay(tm, sign=sign, alpha_frac=ALPHA_FRAC, layer=layer)


def remove_condition(model, cond):
    name, kind, layer, sign = cond
    if kind == "lean" and layer is not None and sign is not None:
        tm = _resolve_text_model(model)
        try: remove_relay(tm)
        except Exception: pass


# === 4. seite13 hidden cache laden für Decoder-Training ===
def load_seite13():
    """Lade alle .pt Files aus seite13_hidden/ → dict (arm, pid) -> per_layer dict."""
    src = os.path.join(PSY, "out/seite13_hidden")
    out = {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".pt"): continue
        arm, pid = fn[:-3].split("__", 1)
        d = torch.load(os.path.join(src, fn), weights_only=False)
        out[(arm, pid)] = d["layers"]
    return out


# === 5. logreg-Decoder (PCA-256 + LogisticRegression, leave-one-cell-out) ===
def train_decoder(seite13_data, target_layer=16):
    """Trainiere 4-class logreg auf L[target_layer] aus seite13. Return clf, pca, classes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Xs, ys, cell_ids = [], [], []
    for (arm, pid), layers in seite13_data.items():
        h = layers[target_layer].numpy()  # [n_tok, 1152]
        Xs.append(h)
        ys.extend([arm] * h.shape[0])
        cell_ids.extend([(arm, pid)] * h.shape[0])
    X = np.concatenate(Xs, 0)
    y = np.asarray(ys)
    cells = np.asarray(cell_ids)
    print(f"[mechtest] seite13 decoder: X={X.shape} y={y.shape} classes={sorted(set(y))}", file=sys.stderr)
    sc = StandardScaler().fit(X)
    Xs_std = sc.transform(X)
    pca = PCA(n_components=min(256, Xs_std.shape[1])).fit(Xs_std)
    Xp = pca.transform(Xs_std)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=1).fit(Xp, y)
    return clf, pca, sc, sorted(set(y))


def predict_decoder(clf, pca, sc, h):
    Xs = sc.transform(h.numpy() if isinstance(h, torch.Tensor) else h)
    Xp = pca.transform(Xs)
    return clf.predict(Xp), clf.predict_proba(Xp)


# === 6. Cosinus-zu-d_width ===
def cos_to_d(h, d_unit):
    """h: [n_tok, d]. Returnt [n_tok] cosinus pro Token."""
    hn = h.float()
    hn = hn / (hn.norm(dim=-1, keepdim=True) + 1e-8)
    return (hn @ d_unit).numpy()


# === 7. Main Run ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--no-relay", action="store_true",
                    help="Skip LEAN_RELAY-Bedingung (spart Zeit wenn kein RELAY gewünscht)")
    args = ap.parse_args()

    prompts = load_prompts(PROMPTS_FILE)
    print(f"[mechtest] {len(prompts)} prompts geladen aus {PROMPTS_FILE}", file=sys.stderr)

    # d_width laden
    with open(D_WIDTH_PATH, "r", encoding="utf-8") as f:
        art = json.load(f)
    d_width = torch.tensor(art["dwidth"], dtype=torch.float32)
    d_unit = d_width / d_width.norm()
    print(f"[mechtest] d_width geladen: norm={float(d_width.norm()):.4f}", file=sys.stderr)

    # Modell laden
    print(f"[mechtest] lade modell {MODEL_ID}...", file=sys.stderr)
    model, tok = build_model(MODEL_ID)
    tm = _resolve_text_model(model)
    print(f"[mechtest] modell geladen, text_model={type(tm).__name__}", file=sys.stderr)

    # seite13 decoder trainieren
    print(f"[mechtest] lade seite13 hidden cache für decoder...", file=sys.stderr)
    s13 = load_seite13()
    print(f"[mechtest] seite13 zellen: {len(s13)}", file=sys.stderr)
    clf, pca, sc, classes = train_decoder(s13, target_layer=16)
    print(f"[mechtest] decoder trainiert auf Klassen: {classes}", file=sys.stderr)

    # Bedingungen auswählen
    conds = list(CONDITIONS)
    if args.no_relay:
        conds = [c for c in conds if c[0] != "LEAN_RELAY"]

    # Run
    cap = MultiCap(tm, LAYERS)
    cap._install()
    index = []
    for cond_name, kind, layer, sign in conds:
        apply_condition(model, (cond_name, kind, layer, sign))
        for pid, ptext in prompts:
            cap.reset()
            try:
                text = _greedy_generate(
                    model, tok,
                    [{"role": "user", "content": ptext}],
                    args.max_new, seed=SEED,
                )
            except Exception as e:
                text = f"<GEN_ERROR {e}>"
                print(f"[mechtest] ERR {cond_name}/{pid}: {e}", file=sys.stderr)
            per_layer = cap.stack()
            # Speichern
            fname = f"{cond_name}__{pid}.pt"
            torch.save({
                "cond": cond_name, "pid": pid, "text": text,
                "layers": {L: per_layer[L].contiguous() for L in per_layer},
            }, os.path.join(HIDDEN_OUT, fname))

            # Decode: logreg pro Token
            l16 = per_layer[16]  # [n_tok, 1152]
            pred, proba = predict_decoder(clf, pca, sc, l16)
            class_counts = {c: int((pred == c).sum()) for c in classes}
            n_tok = l16.shape[0]
            class_pct = {c: class_counts[c] / max(1, n_tok) for c in classes}

            # Cosinus zu d_width pro Token
            cos_l16 = cos_to_d(l16, d_unit)
            cos_l13 = cos_to_d(per_layer[13], d_unit)
            cos_l19 = cos_to_d(per_layer[19], d_unit)
            cos_l21 = cos_to_d(per_layer[21], d_unit)

            rec = {
                "cond": cond_name, "pid": pid,
                "n_tok": n_tok,
                "class_counts": class_counts,
                "class_pct": class_pct,
                "cos_dwidth_l16_mean": float(cos_l16.mean()),
                "cos_dwidth_l16_std": float(cos_l16.std()),
                "cos_dwidth_l13_mean": float(cos_l13.mean()),
                "cos_dwidth_l19_mean": float(cos_l19.mean()),
                "cos_dwidth_l21_mean": float(cos_l21.mean()),
                "text": text,
            }
            index.append(rec)
            print(f"[mechtest] {cond_name:11s} {pid}  ntok={n_tok:4d}  "
                  f"WIDE%={class_pct.get('WIDE',0)*100:5.1f}  NARROW%={class_pct.get('NARROW',0)*100:5.1f}  "
                  f"cos_l16={cos_l16.mean():+.3f}±{cos_l16.std():.2f}", file=sys.stderr)
        remove_condition(model, (cond_name, kind, layer, sign))

    cap.remove()
    del model, tok
    _clear_gpu()

    with open(INDEX_OUT, "w", encoding="utf-8") as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[mechtest] FERTIG. {len(index)} cells. Hidden in {HIDDEN_OUT}/, index in {INDEX_OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
