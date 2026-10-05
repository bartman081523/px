"""tests/test_px_defaults.py — Tests für gradio_tabs/px_defaults.py.

Plan 2026-10-05 (User-Request): "wenn man ein modell gewählt hat, und den
px modus, dass dann die korrekten werte (defaults) automatisch in die
parameter übernommen werden, zb für injektionsschicht, etc."

Pinnt die per-Model-Defaults-Tabelle + die d_width-Artefakt-Übersteuerung.
Refactor-Detector: Wenn die Registry-Keys oder die Tabellenwerte (n_layers,
inject_layer, px_gamma, relay_available) abweichen, fallen diese Tests.
Phase 3 (2026-10-05): get_thinking_defaults — gemma4 (enable_thinking,
kein Budget im installierten Stack) vs. bonsai (reasoning_effort-Stufen).

Run:
    /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python \
      tests/test_px_defaults.py
"""
import os
import sys
import json
import tempfile
import shutil
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradio_tabs.px_defaults import (
    get_inject_layer_from_artifact,
    get_px_defaults,
    get_thinking_defaults,
)


def test_p1_table_covers_exactly_the_registry():
    """T-P1: _PX_MODEL_TABLE kennt GENAU die 9 MODEL_REGISTRY-Keys."""
    from config import MODEL_REGISTRY
    registry_keys = set(MODEL_REGISTRY.keys())
    defaults = set()
    for key in registry_keys:
        entry = get_px_defaults(key)
        assert entry is not None, f"registry key ohne defaults: {key}"
        defaults.add(key)
    # und umgekehrt: keine Tabelle-Komplettbelegung, die die Registry nicht hergibt
    assert defaults == registry_keys


def test_p2_gemma3_1b_it_defaults():
    """T-P2: gemma3-1b-it — 26 Layer, Artifact-Layer 21, gamma 0.12, Relay."""
    d = get_px_defaults("gemma3-1b-it")
    assert d["n_layers"] == 26
    assert d["inject_layer"] == 21  # == d_width-Artefakt google_gemma-3-1b-it
    assert d["hidden_size"] == 1152
    assert d["px_gamma"] == 0.12
    assert d["relay_available"] is True
    assert d["relay_sign"] == 1
    assert d["relay_alpha"] == 0.30


def test_p3_ternary_defaults():
    """T-P3: ternary-bonsai-27b — 64 Layer, Artifact-Layer 34, gamma 0.04."""
    d = get_px_defaults("ternary-bonsai-27b")
    assert d["n_layers"] == 64
    assert d["inject_layer"] == 34  # == d_width-Artefakt ternary-bonsai-2-27b-hf
    assert d["hidden_size"] == 5120
    assert d["px_gamma"] == 0.04
    assert d["relay_available"] is True


def test_p4_gemma4_e2b_defaults():
    """T-P4: gemma4-e2b-it — 35 Layer, Artifact-Layer 26, gamma 0.12."""
    d = get_px_defaults("gemma4-e2b-it")
    assert d["n_layers"] == 35
    assert d["inject_layer"] == 26  # == d_width-Artefakt google_gemma-4-E2B-it
    assert d["hidden_size"] == 1536
    assert d["px_gamma"] == 0.12


def test_p5_minicpm_no_relay_no_gamma():
    """T-P5: minicpm5-1b — kein Relay-Gerüst (kein relay_inject.py im
    Patch-Package), kein 1536-Eintrag im eigenen SCALE_DEFAULTS →
    px_gamma=None; Relay-Felder None (UI-Werte bleiben)."""
    d = get_px_defaults("minicpm5-1b")
    assert d["n_layers"] == 24
    assert d["hidden_size"] == 1536
    assert d["relay_available"] is False
    assert d["relay_sign"] is None
    assert d["relay_alpha"] is None
    assert d["inject_layer"] is None
    assert d["px_gamma"] is None


def test_p6_gemma3_270m_base_mirrors_it_artifact():
    """T-P6: gemma3-270m (base, KEIN Artifact) spiegelt -it: L14/18."""
    for key in ("gemma3-270m", "gemma3-270m-it"):
        d = get_px_defaults(key)
        assert d["n_layers"] == 18
        assert d["inject_layer"] == 14  # live-Artefakt google_gemma-3-270m-it
        assert d["hidden_size"] == 640
        assert d["px_gamma"] == 0.08


def test_p7_artifact_reader_live_values():
    """T-P7: get_inject_layer_from_artifact mit ECHTEN Artefakten."""
    live = [
        ("/home/julian/.cache/huggingface/ternary-bonsai-2-27b-hf", 34),
        ("google/gemma-3-1b-it", 21),
        ("google/gemma-3-1b-pt", 21),
        ("google/gemma-3-270m-it", 14),
        ("google/gemma-3-4b-it", 25),
        ("google/gemma-3-4b-pt", 25),
        ("google/gemma-4-E2B-it", 26),
    ]
    for hf_id, layer in live:
        assert get_inject_layer_from_artifact(hf_id) == layer, hf_id


def test_p8_artifact_reader_edge_cases():
    """T-P8: fehlendes Artefakt / leere hf_id / nicht-int → None."""
    assert get_inject_layer_from_artifact("openbmb/MiniCPM5-1B") is None
    assert get_inject_layer_from_artifact("") is None
    assert get_inject_layer_from_artifact(None) is None
    assert get_inject_layer_from_artifact("gibts/nicht") is None


def test_p9_artifact_reader_px_relay_dir_env(tmp_path):
    """T-P9: PX_RELAY_DIR-Override == relay_inject-Semantik (live gelesen)."""
    d = tmp_path / "relays"
    d.mkdir()
    (d / "custom_artifact_relay_dwidth.json").write_text(
        json.dumps({"inject_layer": 9, "hidden_size": 640}), encoding="utf-8"
    )
    old = os.environ.get("PX_RELAY_DIR")
    try:
        os.environ["PX_RELAY_DIR"] = str(d)
        assert get_inject_layer_from_artifact("custom/artifact") == 9
    finally:
        if old is None:
            os.environ.pop("PX_RELAY_DIR", None)
        else:
            os.environ["PX_RELAY_DIR"] = old


def test_p10_artifact_layer_wins_over_table():
    """T-P10: frisches Artefakt schlägt die statische Tabelle (gleiche
    Regel wie relay_inject) — hier via PX_RELAY_DIR-Override."""
    d = tempfile.mkdtemp()
    try:
        with open(os.path.join(d, "google_gemma-3-1b-it_relay_dwidth.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"inject_layer": 11, "hidden_size": 1152}, f)
        old = os.environ.get("PX_RELAY_DIR")
        try:
            os.environ["PX_RELAY_DIR"] = d
            d2 = get_px_defaults("gemma3-1b-it")
            assert d2["inject_layer"] == 11      # Artifact-Wert
            assert d2["n_layers"] == 26          # Tabellen-Bounds unverändert
        finally:
            if old is None:
                os.environ.pop("PX_RELAY_DIR", None)
            else:
                os.environ["PX_RELAY_DIR"] = old
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_p11_unknown_model_returns_none():
    """T-P11: unbekannte model_id → None (Handler no-op't dann sauber)."""
    assert get_px_defaults("unbekannt-modell") is None
    assert get_px_defaults("") is None
    assert get_px_defaults(None) is None


# --- T-P12ff: get_thinking_defaults (Phase 3, 2026-10-05) ------------------

def test_p12_gemma4_thinking_no_budget():
    """T-P12: gemma4-e2b-it — Template default(false), KEIN Budget-Parameter
    (efforts=None → kein Budget-Widget; max_thinking_tokens existiert nur im
    ungemergten transformers-PR #42112, nicht in 5.13.0/Template/Model Card)."""
    d = get_thinking_defaults("gemma4-e2b-it")
    assert d == {
        "default": False,
        "efforts": None,
        "effort_default": None,
    }


def test_p13_bonsai_thinking_with_effort_stages():
    """T-P13: ternary-bonsai-27b — Template-Default AN, reasoning_effort ist
    der Budget-Parameter (Stufen xhigh|medium|low, Default xhigh)."""
    d = get_thinking_defaults("ternary-bonsai-27b")
    assert d == {
        "default": True,
        "efforts": ("xhigh", "medium", "low"),
        "effort_default": "xhigh",
    }


def test_p14_incapable_models_return_none():
    """T-P14: nicht-thinking-capable Modelle → None (gemma3-Skalen, MiniCPM,
    Unbekannt). chat_fn darf dort KEINE Template-Extras hingeschicken."""
    for model_id in ("gemma3-270m", "gemma3-270m-it", "gemma3-1b",
                     "gemma3-1b-it", "gemma3-4b", "gemma3-4b-it",
                     "minicpm5-1b", "unbekannt-modell"):
        assert get_thinking_defaults(model_id) is None, model_id
    assert get_thinking_defaults("") is None
    assert get_thinking_defaults(None) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))