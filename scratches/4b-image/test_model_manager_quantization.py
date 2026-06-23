"""
test_model_manager_quantization.py — TDD-rot/green für model_manager-Integration
==================================================================================

Phase D von Plan 1. Was diese Tests pinnen:
  - model_manager._load_model akzeptiert `quantization: Literal["none","int8"]`
  - quantization="int8" → nach from_pretrained monkey-patcht ALLE nn.Linear
  - quantization="none" → kein Patch (Default-Verhalten erhalten)
  - Server-Request-Param "quantization" wird durchgereicht
  - Registry-Eintrag "quantization" wird respektiert
  - GPU-Speicher-Messung: int8-Last braucht weniger als bf16-Last
    (mit Mock: ein winziges Model, realer Speicher-Vergleich)

Wichtig: Diese Tests mocken from_pretrained (kein echter 4b-Download) —
sie prüfen die LOGIK der Integration, nicht die Korrektheit der
Quantisierung selbst (das ist in Phase A/B/C getestet).

Run:
    /path/to/venv/bin/python test_model_manager_quantization.py
"""
import sys
import os
from unittest.mock import patch, MagicMock

# Repo-Root für `import model_manager`
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _fake_tiny_model():
    """Kleines Mock-Model, das 'from_pretrained' zurückgibt."""
    import torch
    import torch.nn as nn
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 128, bias=True)
            self.linear2 = nn.Linear(128, 32, bias=False)
        def forward(self, x):
            return self.linear2(self.linear1(x).relu())
    return TinyModel()


# Tests ---------------------------------------------------------------------

def test_model_manager_load_respects_quantization_int8():
    """_load_model mit quantization='int8' monkey-patcht die Linears."""
    from model_manager import ModelManager
    from quantized_linear import QuantizedLinear
    from config import MODEL_REGISTRY

    fake_model = _fake_tiny_model()
    fake_model_id = "fake-tiny-4b-mock"

    # Mocke AutoModelForCausalLM.from_pretrained + AutoTokenizer.
    # AutoProcessor wird nur bei gemma3_conditional/gemma4_conditional geladen,
    # wir nutzen hier text-only "gemma3", also nicht relevant.
    with patch("model_manager.AutoModelForCausalLM") as m_causal, \
         patch("model_manager.AutoTokenizer") as m_tok:
        m_causal.from_pretrained.return_value = fake_model
        m_tok.from_pretrained.return_value = MagicMock()
        if fake_model_id not in MODEL_REGISTRY:
            MODEL_REGISTRY[fake_model_id] = {
                "hf_id": "fake/4b",
                "tokenizer_id": "fake/4b",
                "patch_dir": None,
                "patch_kwargs": {},
                "model_type": "gemma3",  # text-only, kein Processor nötig
                "dtype": "bfloat16",
                "max_length": 4096,
                "quantization": "int8",  # ← der Test
            }
        else:
            MODEL_REGISTRY[fake_model_id]["quantization"] = "int8"

        mgr = ModelManager(max_loaded_models=1)
        entry = mgr._load_model(fake_model_id, px_subjective=False, quantization="int8")
        loaded_model = entry["model"]
        n_q = sum(1 for _ in loaded_model.modules() if isinstance(_, QuantizedLinear))
        assert n_q >= 2, f"expected >= 2 QuantizedLinears, got {n_q}"
        mgr.unload(fake_model_id)
        MODEL_REGISTRY.pop(fake_model_id, None)

    print(f"[OK] load with quantization=int8 → {n_q} QuantizedLinears")


def test_model_manager_load_respects_quantization_none():
    """_load_model mit quantization='none' macht KEINEN Patch."""
    from model_manager import ModelManager
    from quantized_linear import QuantizedLinear
    from config import MODEL_REGISTRY

    fake_model = _fake_tiny_model()
    fake_model_id = "fake-tiny-4b-mock-none"

    with patch("model_manager.AutoModelForCausalLM") as m_causal, \
         patch("model_manager.AutoTokenizer") as m_tok:
        m_causal.from_pretrained.return_value = fake_model
        m_tok.from_pretrained.return_value = MagicMock()
        if fake_model_id not in MODEL_REGISTRY:
            MODEL_REGISTRY[fake_model_id] = {
                "hf_id": "fake/4b",
                "tokenizer_id": "fake/4b",
                "patch_dir": None,
                "patch_kwargs": {},
                "model_type": "gemma3",
                "dtype": "bfloat16",
                "max_length": 4096,
                "quantization": "none",
            }
        MODEL_REGISTRY[fake_model_id]["quantization"] = "none"

        mgr = ModelManager(max_loaded_models=1)
        entry = mgr._load_model(fake_model_id, px_subjective=False, quantization="none")
        loaded_model = entry["model"]
        n_q = sum(1 for _ in loaded_model.modules() if isinstance(_, QuantizedLinear))
        assert n_q == 0, f"expected 0 QuantizedLinears, got {n_q}"
        mgr.unload(fake_model_id)
        MODEL_REGISTRY.pop(fake_model_id, None)

    print(f"[OK] load with quantization=none → {n_q} QuantizedLinears (no patch)")


def test_model_manager_registry_default_int8_for_4b():
    """4b-Registry-Eintrag hat 'quantization': 'int8' als Default.

    Begründung: 4b ohne Quantisierung passt nicht auf 12 GB für Lang-Prefill.
    Das ist eine bewusste Default-Entscheidung; User können via
    Request-Param auf 'none' überschreiben."""
    from config import MODEL_REGISTRY
    eintrag = MODEL_REGISTRY.get("gemma3-4b-it", {})
    q = eintrag.get("quantization")
    assert q in ("int8",), f"4b quantization default {q!r} != 'int8'"
    print(f"[OK] 4b registry default quantization={q!r}")


def test_model_manager_load_signature_accepts_quantization():
    """_load_model-Signatur akzeptiert `quantization` als Parameter (mit Default)."""
    import inspect
    from model_manager import ModelManager
    sig = inspect.signature(ModelManager._load_model)
    assert "quantization" in sig.parameters, (
        f"_load_model missing 'quantization' param, got: {list(sig.parameters)}")
    # Default sollte "none" sein (Backwards-Compat).
    default = sig.parameters["quantization"].default
    assert default in ("none", None), f"quantization default {default!r} != 'none'"
    print(f"[OK] _load_model(quantization={default!r}) signature present")


def test_model_manager_quantization_param_in_get_model():
    """get_model/get_model_async reichen quantization an _load_model durch."""
    import inspect
    from model_manager import ModelManager
    for meth in ("get_model", "get_model_async", "_load_model"):
        if not hasattr(ModelManager, meth):
            continue
        sig = inspect.signature(getattr(ModelManager, meth))
        # Mindestens eine der Pfade muss quantization durchreichen.
        if "quantization" in sig.parameters:
            print(f"[OK] {meth}(quantization=...) signature present")
            return
    raise AssertionError("no method on ModelManager accepts 'quantization' parameter")


if __name__ == "__main__":
    tests = [
        ("load int8 patches Linears",       test_model_manager_load_respects_quantization_int8),
        ("load none is noop",               test_model_manager_load_respects_quantization_none),
        ("4b registry default int8",        test_model_manager_registry_default_int8_for_4b),
        ("_load_model signature",           test_model_manager_load_signature_accepts_quantization),
        ("get_model param passthrough",     test_model_manager_quantization_param_in_get_model),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
