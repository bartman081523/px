"""Kalibrierung für GF(5)-AWQ: Aktivierungsstatistik je Linear-Eingang.

Sammelt für jede lineare Schicht das mittlere |x| pro Eingangskanal über einen
kleinen, diversen Korpus (Englisch, Deutsch, Code). Diese Statistik gewichtet
den Quantisierungsfehler so, wie er tatsächlich durch das Netz wirkt:
Kanäle, die große Aktivierungen tragen, bekommen feinere Schritte.

Ausgabe: `calib.npz` mit einem Vektor je Layername (+ Metadaten).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

CORPUS = [
    "Write a Python function that returns the sum of two integers. Include a docstring.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n",
    "Explain how attention works in a transformer model, step by step, for a beginner.",
    "Die Hauptstadt von Frankreich ist Paris. Sie liegt an der Seine und hat etwa zwei Millionen Einwohner.",
    "class Cache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.store = {}\n",
    "A binary search runs in O(log n) time on a sorted array because each step halves the search space.",
    "Im Maschinenbau beschreibt die Zugfestigkeit die maximale Spannung, die ein Werkstoff vor dem Bruch erträgt.",
    "SELECT user_id, COUNT(*) AS n FROM orders WHERE created_at > '2026-01-01' GROUP BY user_id ORDER BY n DESC;",
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.",
    "Berechne die Ableitung von f(x) = 3x^2 + 2x - 7 und erkläre die Rechenschritte.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/julian/.cache/huggingface/hub/"
                                       "models--google--gemma-4-E2B-it/snapshots/"
                                       "4adf15748de9e1a0097b6274cee9c15836c41c37")
    ap.add_argument("--out", default="/home/julian/.cache/huggingface/calib_gf5.npz")
    ap.add_argument("--max-length", type=int, default=256)
    a = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()

    acc: dict[str, torch.Tensor] = {}
    count = {"n": 0}

    def make_hook(name):
        def hook(_mod, inputs):
            if not inputs:
                return
            x = inputs[0]
            if x.dim() < 2:
                return
            flat = x.detach().reshape(-1, x.shape[-1]).float()
            v = flat.abs().mean(dim=0)
            if name in acc:
                acc[name] += v
            else:
                acc[name] = v.clone()
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            handles.append(mod.register_forward_pre_hook(make_hook(name)))
    print(f"[calib] hooks: {len(handles)}", flush=True)

    with torch.no_grad():
        for i, text in enumerate(CORPUS):
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=a.max_length).to("cuda")
            model(**ids)
            count["n"] += 1
            print(f"[calib] sample {i+1}/{len(CORPUS)}", flush=True)
    for h in handles:
        h.remove()

    out = {}
    for name, v in acc.items():
        v = (v / count["n"]).cpu().numpy().astype(np.float32)
        v = np.maximum(v, 1e-6)
        out[name] = v
    np.savez_compressed(a.out, **out)
    meta = {"model": a.model, "samples": count["n"], "max_length": a.max_length,
            "layers": len(out)}
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[calib] wrote {a.out} ({len(out)} layers, {count['n']} samples)")
    # kurzer Sanity-Check
    k = next(iter(out))
    print(f"[calib] {k}: mean|a|={out[k].mean():.4f} max={out[k].max():.2f} "
          f"ratio max/median={out[k].max()/np.median(out[k]):.1f}")


if __name__ == "__main__":
    main()
