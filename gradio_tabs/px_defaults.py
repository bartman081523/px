"""gradio_tabs/px_defaults.py — Per-Model PX-Defaults für die Sidebar.

User-Request 2026-10-05: "wenn man ein modell gewählt hat, und den px modus,
dass dann die korrekten werte (defaults) automatisch in die parameter
übernommen werden, zb für injektionsschicht, etc."

Sources of Truth (alle live verifiziert 2026-10-05):

1. Relay-Injektions-Layer: d_width-Artefakte
       px_manifolds/<hf_id mit "/"→"_">_relay_dwidth.json  (Feld inject_layer)
   Gleiche Lookup-Semantik wie
       px_patches.*/relay_inject.get_inject_layer_for_hf_id
   aber OHNE torch-Import (pure json) — deshalb hier nochmal, nicht
   gecrossimported. Artefakte (inject_layer | hidden):
       ternary-bonsai-2-27b-hf → 34 | 5120
       gemma-3-1b-it / -1b-pt  → 21 | 1152
       gemma-3-270m-it         → 14 | 640
       gemma-3-4b-it / -4b-pt  → 25 | 2560
       gemma-4-E2B-it          → 26 | 1536
   KEIN Artefakt: google/gemma-3-270m (base), openbmb/MiniCPM5-1B.

2. n_layers (Slider-Bounds für den Injektions-Layer):
       270m: 18 (lokale config.json, open-mythos_p2 Checkpoints)
       1b:   26 (Gemma-3 Model Card, arXiv:2503.19786)
       4b:   34 (Gemma-3 Model Card + HF config.json text_config)
       E2B:  35 (px_patches/gemma4_2b_px/patch.py:202-Erkennungsbedingung
                 sliding_window==512 and num_hidden_layers==35, Plus
                 SCALE_DEFAULTS-Kommentar "hidden_size=1536, 35 layers")
       minicpm5: 24 (px_patches/minicpm5_1b_px/configuration_minicpm5_px.py)
       ternary:  64 (config.json, hybrid GDN/full@i%4==3)

3. px_gamma: SCALE_DEFAULTS pro hidden_size (auto_tune.py der Patch-Dirs;
   Alle Patch-Dirs halten identische Maps):
       640 → 0.08   (270m)
       1152 → 0.12  (1b)
       1536 → 0.12  (gemma4 E2B — ABER NICHT im MiniCPM-Auto-Tune-Map!)
       2560 → 0.05  (4b)
       5120 → 0.04  (ternary)
   MiniCPM5-1B: hidden_size=1536 liegt NUR in den gemma4/ternary-Maps,
   nicht im eigenen (640/1152/2560/4096) → beim Tune kein Scale-Eintrag,
   gamma ist caller-provided → hier bewusst px_gamma=None (kein
   Auto-Default, UI-Wert bleibt) + relay=False (kein relay_inject.py im
   MiniCPM-Patch-Package → Relay ist dort no-op).

Bewusst NICHT hier: System-Prompt-Profile laden. User-Entscheidung
2026-07-09 — px_preset ändert KEINEN System-Prompt (citmind/juexin nur via
Profil-Klick/manuell). Diese Entscheidung bleibt unangetastet.

Öffentliche API:
    get_inject_layer_from_artifact(hf_id) -> Optional[int]
    get_px_defaults(model_id) -> Optional[Dict[str, Any]]
    get_thinking_defaults(model_id) -> Optional[Dict[str, Any]]

    keys im Rückgabe-Dict:
        n_layers (int)        — Slider maximum für den Injektions-Layer
        hidden_size (int)
        px_gamma (float|None) — None → UI-Wert unverändert übernehmen
        inject_layer (int|None) — None → UI-Wert unverändert übernehmen
        relay_sign (int|None)
        relay_alpha (float|None)
        relay_available (bool)

    Rückgabe-Dict von get_thinking_defaults:
        default (bool)             — enable_thinking-Template-Default
        efforts (tuple[str]|None)  — None → kein Budget-Parameter
        effort_default (str|None)  — Default-Budget-Stufe (nur wenn efforts)
        budget_default (int|None)  — Zahl-Budget (gemma4: 2048)
        budget_range (tuple|None)  — (min, max) Slider-Bounds (gemma4: 0,8192)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Gleiche Override-Semantik wie relay_inject._relay_dir(): PX_RELAY_DIR
# env schlägt den Default px_manifolds-Dir im Repo-Root.
DEFAULT_RELAY_DIR = os.path.join(_REPO_ROOT, "px_manifolds")


def _relay_dir(relay_dir: Optional[str] = None) -> str:
    return relay_dir or os.environ.get("PX_RELAY_DIR") or DEFAULT_RELAY_DIR


# ── Verifizierte statische Tabelle ──────────────────────────────────────
# Quelle-Beweis im Moduldocstring. inject_layer hier = Fallback/Firmware,
# wenn kein d_width-Artefakt für die hf_id existiert (gemma3-270m base →
# Mirror des -it-Artefakts: identischer Patch gemma3_270m_px_baseline
# + identische Architektur 18L/640).
_PX_MODEL_TABLE: Dict[str, Dict[str, Any]] = {
    "gemma3-270m": dict(
        n_layers=18, hidden_size=640, px_gamma=0.08, inject_layer=14,
        relay_available=True,
    ),
    "gemma3-270m-it": dict(
        n_layers=18, hidden_size=640, px_gamma=0.08, inject_layer=14,
        relay_available=True,
    ),
    "gemma3-1b": dict(
        n_layers=26, hidden_size=1152, px_gamma=0.12, inject_layer=21,
        relay_available=True,
    ),
    "gemma3-1b-it": dict(
        n_layers=26, hidden_size=1152, px_gamma=0.12, inject_layer=21,
        relay_available=True,
    ),
    "gemma3-4b": dict(
        n_layers=34, hidden_size=2560, px_gamma=0.05, inject_layer=25,
        relay_available=True,
    ),
    "gemma3-4b-it": dict(
        n_layers=34, hidden_size=2560, px_gamma=0.05, inject_layer=25,
        relay_available=True,
    ),
    "gemma4-e2b-it": dict(
        n_layers=35, hidden_size=1536, px_gamma=0.12, inject_layer=26,
        relay_available=True,
        # Thinking (Phase 3 + Budget, 2026-10-05): chat_template.jinja —
        # `{%- set enable_thinking = enable_thinking | default(false) -%}`
        # → Template-Default AUS. Budget: gemma4 HAT max_thinking_tokens —
        # im Template/Model Card/Google-Dok NICHT als generate-Parameter
        # (transformers 5.13.0: nur ungemergter PR #42112), sondern als
        # Doku-Semantik, die wir App-Level realisieren
        # (generators.ThinkingBudgetLogitsProcessor über die echten
        # Kanal-Tokens <|channel>=100 / thought=45518 / <channel|>=101;
        # Probe scratches/gemma4_think_tokens_probe.py).
        thinking_default=False,
        thinking_budget_default=2048,
        # Range = Slider-Bounds: unten 0 (= unbegrenzt, Semantik im Widget-
        # info), oben 8192 (Kontextplausibel für 12-GB-Karte).
        thinking_budget_range=(0, 8192),
    ),
    "minicpm5-1b": dict(
        n_layers=24, hidden_size=1536, px_gamma=None, inject_layer=None,
        relay_available=False,
    ),
    "ternary-bonsai-27b": dict(
        n_layers=64, hidden_size=5120, px_gamma=0.04, inject_layer=34,
        relay_available=True,
        # Thinking (Phase 3, 2026-10-05): qwen3.5-Tokenizer-Template —
        # `enable_thinking is undefined or is true` → Template-Default AN;
        # reasoning_effort|default('xhigh') mit raise_exception außerhalb
        # xhigh|medium|low. reasoning_effort IST hier der Budget-Parameter
        # (Denk-Tiefe als Stufe, kein Token-Zähler).
        thinking_default=True,
        thinking_efforts=("xhigh", "medium", "low"),
        thinking_effort_default="xhigh",
    ),
}

# Relay-Defaults (seite15-Semantik, unverändert gegen die bisherigen
# UI-Default-Values des Relay-Accordions): sign=+1 (WIDE/expansiv),
# alpha=0.30 (kohärenter Chat, Bruchteil der Capture-Layer-Norm).
RELAY_SIGN_DEFAULT = 1
RELAY_ALPHA_DEFAULT = 0.30


def get_inject_layer_from_artifact(hf_id: str,
                                   relay_dir: Optional[str] = None) -> Optional[int]:
    """Liest inject_layer aus dem d_width-Artefakt der hf_id (pure json).

    Gibt None zurück wenn hf_id leer, kein Artefakt existiert oder das
    Feld fehlt/nicht-int ist — der Caller übernimmt dann sein Fallback.
    """
    if not hf_id:
        return None
    # Gleiche Safe-ID-Semantik wie relay_inject.get_inject_layer_for_hf_id:
    # "/" → "_" (ternary hf_id ist ein lokaler Pfad → 3 Slashes).
    safe_id = hf_id.replace("/", "_")
    if not safe_id:
        return None
    path = os.path.join(_relay_dir(relay_dir), f"{safe_id}_relay_dwidth.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    layer = data.get("inject_layer")
    if isinstance(layer, bool) or not isinstance(layer, int):
        return None
    return layer


def get_px_defaults(model_id: str) -> Optional[Dict[str, Any]]:
    """Per-Model-Defaults; None bei unbekannter model_id.

    inject_layer kommt — wenn möglich — aus dem d_width-Artefakt der
    hf_id (live), sonst aus der statischen Tabelle (Fallback), sonst
    None. relay_available=False (MiniCPM) → relay-Felder = None.
    """
    entry = _PX_MODEL_TABLE.get(model_id)
    if entry is None:
        return None

    result = dict(entry)
    if result.get("relay_available"):
        if not result.get("relay_sign"):
            result["relay_sign"] = RELAY_SIGN_DEFAULT
        if not result.get("relay_alpha"):
            result["relay_alpha"] = RELAY_ALPHA_DEFAULT
        # Artefakt schlägt Tabelle (beide sollten identisch sein — wenn sie
        # divergieren, gilt das frischere Artefakt, wie in relay_inject).
        from config import MODEL_REGISTRY
        hf_id = (MODEL_REGISTRY.get(model_id) or {}).get("hf_id")
        artifact_layer = get_inject_layer_from_artifact(hf_id)
        if artifact_layer is not None:
            result["inject_layer"] = artifact_layer
    else:
        result["relay_sign"] = None
        result["relay_alpha"] = None
        result["inject_layer"] = None
    return result


def get_thinking_defaults(model_id: str) -> Optional[Dict[str, Any]]:
    """Thinking-Kapabilität + Template-Defaults; None = nicht capable.

    Quelle-Beweis (2026-10-05, User-Request "etablierte Methode"):
      gemma4-e2b-it     → chat_template.jinja (HF-Snapshot 3e22461f) setzt
                          `enable_thinking | default(false)` — Thinking
                          ON injiziert `<|think|>` in den ersten System-
                          Turn (das Template baut den System-Turn selbst,
                          kein Message-Inject nötig). Budget: max_thinking_
                          tokens ist NICHT generate-Etabliert (transformers
                          5.13.0 kennt es nicht, nur ungemergt PR #42112) —
                          die Doku-Semantik realisieren wir App-Level
                          (generators.ThinkingBudgetLogitsProcessor über
                          die Kanal-Tokens open=100/confirm=45518/
                          close=101) → budget_default/budget_range.
      ternary-bonsai-27b → qwen3.5-Template (tokenizer_config.json):
                          enable_thinking undefined/true → denken;
                          reasoning_effort|default('xhigh'); raise_exception
                          für alles außer xhigh|medium|low → reasoning_
                          effort = Budget-Stufe (KEIN Zahl-Budget).

    None (gemma3-*, minicpm5-1b, unbekannte model_id): Modelle ohne
    Thinking-Template-Variablen — chat_fn darf dort KEINE enable_thinking-
    Variable in den Jinja-Kontext geben (kapabilitäts-gegated).

    Rückgabe-Schema:
        default        bool   — Template-Default des Modells
        efforts        tuple|None — effort-Stufen (bonsai) oder None
        effort_default str|None   — nur bei efforts
        budget_default int|None   — Zahl-Budget (gemma4: 2048)
        budget_range   tuple|None — (min, max) Slider-Bounds (gemma4: 0,8192;
                          min 0 = "unbegrenzt")
    """
    entry = _PX_MODEL_TABLE.get(model_id)
    if entry is None or "thinking_default" not in entry:
        return None
    efforts = entry.get("thinking_efforts") or None
    # Budget (Plan 2026-10-05): gemma4 → (default, range)-Tupel; bonsai hat
    # reasoning_effort-Stufen ALS Budget-Parameter → bewusst KEIN Zahl-Budget.
    budget_range = entry.get("thinking_budget_range") or None
    budget_default = entry.get("thinking_budget_default") if budget_range else None
    return {
        "default": bool(entry["thinking_default"]),
        "efforts": efforts,
        "effort_default": entry.get("thinking_effort_default") if efforts else None,
        "budget_default": (int(budget_default)
                           if budget_default is not None else None),
        "budget_range": (tuple(budget_range) if budget_range else None),
    }