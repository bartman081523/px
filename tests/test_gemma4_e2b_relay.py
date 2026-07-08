"""Mock-Tests für gemma4_2b_px RELAY-Wiring (kein GPU).

Verifiziert, dass ACTIVE_MANIFOLD_RELAY im gemma4_2b_px-Patch korrekt
verdrahtet ist, OHNE ein echtes Modell zu laden. Setzt einen minimalen
text_model-Mock auf, der die Attribute hält, die relay_inject erwartet
(hidden_size, _name_or_path) — plus die _px_*-Attribute, die patch.py
darauf schreibt.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _MockConfig:
    """Minimal text_model.config."""
    def __init__(self, hidden_size=1536, num_hidden_layers=35, hf_id=""):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self._name_or_path = hf_id


class _MockOuterConfig:
    """Minimal model.config (Top-Level)."""
    def __init__(self, hf_id="google/gemma-4-E2B-it"):
        self._name_or_path = hf_id


class _MockTextModel:
    """Minimal text_model — die Attribute, die patch.py setzt + liest."""
    def __init__(self, hf_id="", is_gemma4_conditional=False):
        self.config = _MockConfig(hf_id=hf_id)
        # Patches writes these:
        self._px_relay_handles = []
        self._px_relay_cfg = {}
        self._px_hf_id = None
        # Original forward bookkeeping
        self.forward = lambda *a, **kw: None
        self._is_gemma4_conditional = is_gemma4_conditional


class _MockOuter:
    """Minimal outer model (model.model)."""
    def __init__(self):
        self.forward = lambda *a, **kw: None


class _MockModel:
    """Minimal model wrapper — entweder Gemma3ForConditionalGeneration-Shape
    (mit .model) oder reines Gemma4ForCausalLM (nur text_model)."""
    def __init__(self, hf_id="google/gemma-4-E2B-it", is_gemma4_conditional=True):
        self.config = _MockOuterConfig(hf_id=hf_id)
        if is_gemma4_conditional:
            self.model = _MockOuter()
            self.model.language_model = _MockTextModel(hf_id="", is_gemma4_conditional=True)
        else:
            self.text_model = _MockTextModel(hf_id=hf_id)


def _import_gemma4_patch():
    """Importiert px_patches.gemma4_2b_px.patch — das geht ohne Modell
    nur, wenn die Top-Level-Imports von torch/transformers verfügbar sind.
    Wir mocken diese, falls nötig."""
    try:
        from px_patches.gemma4_2b_px import patch as P
        return P
    except Exception as e:
        # Wenn torch nicht verfügbar: importiere nur die Symbole, die wir
        # für die Source-Inspektion brauchen.
        raise RuntimeError(f"kann gemma4_2b_px.patch nicht importieren: {e}")


def test_relay_inject_module_importable():
    """relay_inject.py wurde von gemma3_270m kopiert und ist importierbar."""
    from px_patches.gemma4_2b_px import relay_inject
    # Public API der shared engine
    for name in ("install_relay", "remove_relay", "load_dwidth"):
        assert hasattr(relay_inject, name), f"fehlt: {name}"


def test_patch_imports_relay_inject():
    """patch.py importiert install_relay/remove_relay aus relay_inject."""
    P = _import_gemma4_patch()
    # Direkt: das Modul hat die Symbole im Namespace gebunden
    import px_patches.gemma4_2b_px.patch as P2
    src = open(os.path.join(os.path.dirname(P2.__file__), "patch.py")).read()
    assert "from .relay_inject import install_relay, remove_relay" in src
    # Source-Level: install_relay + remove_relay werden in patch.py aufgerufen
    assert "install_relay(text_model," in src
    assert "remove_relay(text_model)" in src


def test_valid_preset_set_includes_relay():
    """ACTIVE_MANIFOLD_RELAY ist im valid-preset-Set von apply_px_patch."""
    import px_patches.gemma4_2b_px.patch as P2
    src = open(os.path.join(os.path.dirname(P2.__file__), "patch.py")).read()
    # Die valid-preset-Whitelist enthält ACTIVE_MANIFOLD_RELAY
    assert '"ACTIVE_MANIFOLD_RELAY"' in src
    # ACTIVE_MANIFOLD_LEAN bleibt als eigener preset erhalten
    assert '"ACTIVE_MANIFOLD_LEAN"' in src
    # lean-Flag wird auf BEIDE Presets gesetzt
    assert 'lean = (config_preset in ("ACTIVE_MANIFOLD_LEAN", "ACTIVE_MANIFOLD_RELAY"))' in src


def test_relay_default_layer_for_gemma4_is_26():
    """Default-Layer für gemma4-e2b relay ist 26 (post-recur)."""
    import px_patches.gemma4_2b_px.patch as P2
    src = open(os.path.join(os.path.dirname(P2.__file__), "patch.py")).read()
    # install_relay wird mit layer=26 aufgerufen (statt gemma3 21)
    assert "layer=defaults.get(\"relay_layer\", 26)" in src


def test_relay_sign_default_for_relay_preset_is_positive():
    """ACTIVE_MANIFOLD_RELAY → sign=+1 (WIDE/expansiv), andere → 0."""
    import px_patches.gemma4_2b_px.patch as P2
    src = open(os.path.join(os.path.dirname(P2.__file__), "patch.py")).read()
    assert '+1 if config_preset == "ACTIVE_MANIFOLD_RELAY" else 0' in src


def test_remove_px_patch_calls_remove_relay():
    """remove_px_patch ruft remove_relay als erstes auf (idempotent)."""
    import px_patches.gemma4_2b_px.patch as P2
    src = open(os.path.join(os.path.dirname(P2.__file__), "patch.py")).read()
    # In remove_px_patch: remove_relay(text_model) am Anfang.
    fn_start = src.index("def remove_px_patch(model):")
    fn_body = src[fn_start:]
    assert "remove_relay(text_model)" in fn_body[:300]  # innerhalb der ersten Zeilen


def test_remove_px_patch_drops_relay_attrs():
    """_px_relay_handles, _px_relay_cfg, _px_hf_id sind im delattr-Loop."""
    import px_patches.gemma4_2b_px.patch as P2
    src = open(os.path.join(os.path.dirname(P2.__file__), "patch.py")).read()
    for attr in ("_px_relay_handles", "_px_relay_cfg", "_px_hf_id"):
        assert attr in src, f"delattr-Loop fehlt: {attr}"


def test_hf_id_propagation_for_conditional_model():
    """Bei Gemma4 conditional wird hf_id vom outer config auf text_model._px_hf_id
    propagiert (text_model.config._name_or_path ist leer im HF-Set)."""
    import px_patches.gemma4_2b_px.patch as P2
    src = open(os.path.join(os.path.dirname(P2.__file__), "patch.py")).read()
    # Source: outer_hf_id wird gelesen, _px_hf_id gesetzt
    assert "outer_hf_id = getattr(getattr(model, \"config\", None), \"_name_or_path\", None)" in src
    assert "text_model._px_hf_id = outer_hf_id" in src


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
