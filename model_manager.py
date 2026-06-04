"""
model_manager.py — Lazy Model Loading, Re-Patching, and PX Metrics
===================================================================
Manages model lifecycle on GPU. Lazy loads on first request.
Re-patches instead of reloading when switching Peak ↔ Subjective.
"""

import torch
import sys
import os
import time
import importlib
from typing import Dict, Optional

from config import MODEL_REGISTRY
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelManager:
    def __init__(self):
        self._models: Dict[str, dict] = {}        # model_id -> loaded entry
        self._loading: Dict[str, bool] = {}        # model_id -> loading flag
        self._last_used: Dict[str, float] = {}     # model_id -> timestamp
        self._px_metrics: Dict[str, dict] = {}     # model_id -> latest metrics

    async def get_model(self, model_id: str, px_subjective: bool = False) -> dict:
        """Get a loaded model, loading lazily if needed.

        If model is loaded with different subjective mode,
        re-apply the patch (no weight reload needed).
        """
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_id}")

        if model_id in self._models:
            entry = self._models[model_id]
            current_subjective = entry.get("px_subjective", False)
            if current_subjective != px_subjective:
                self._reapply_patch(model_id, px_subjective)
            self._last_used[model_id] = time.time()
            return entry

        # Lazy load
        if self._loading.get(model_id):
            raise RuntimeError(f"Model {model_id} is currently loading")

        self._loading[model_id] = True
        try:
            entry = self._load_model(model_id, px_subjective)
            self._models[model_id] = entry
            self._last_used[model_id] = time.time()
            return entry
        finally:
            self._loading[model_id] = False

    def _load_model(self, model_id: str, px_subjective: bool) -> dict:
        """Load model weights + tokenizer + apply PX patch."""
        registry = MODEL_REGISTRY[model_id]
        hf_id = registry["hf_id"]
        tok_id = registry["tokenizer_id"]

        print(f"[ModelManager] Loading {model_id} from {hf_id} (subjective={px_subjective})...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_id)
        # Apply manual chat template if needed (Gemma3 base model)
        if registry.get("chat_template_manual"):
            tokenizer.chat_template = registry["chat_template_manual"]

        # Load model
        dtype = getattr(torch, registry["dtype"])
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map="auto",
        )

        # Apply PX patch (skip for unpatched baseline models)
        if registry.get("patch_dir") is not None:
            patch_kwargs = dict(registry["patch_kwargs"])
            if px_subjective and registry.get("subjective_kwargs"):
                patch_kwargs.update(registry["subjective_kwargs"])

            apply_fn = self._get_patch_function(model_id, "apply_px_patch")
            apply_fn(model, **patch_kwargs)
            print(f"[ModelManager] {model_id} loaded and patched successfully.")
        else:
            print(f"[ModelManager] {model_id} loaded WITHOUT patch (baseline).")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "registry": registry,
            "px_subjective": px_subjective,
            "model_type": registry["model_type"],
        }

    def _reapply_patch(self, model_id: str, px_subjective: bool):
        """Re-apply PX patch with different subjective mode (no weight reload)."""
        entry = self._models[model_id]
        registry = entry["registry"]

        print(f"[ModelManager] Re-patching {model_id} (subjective={px_subjective})...")

        # Remove existing patch
        remove_fn = self._get_patch_function(model_id, "remove_px_patch")
        if remove_fn:
            try:
                remove_fn(entry["model"])
            except Exception as e:
                print(f"[ModelManager] Warning: remove_px_patch failed: {e}")

        # Re-apply with new settings
        patch_kwargs = dict(registry["patch_kwargs"])
        if px_subjective and registry.get("subjective_kwargs"):
            patch_kwargs.update(registry["subjective_kwargs"])

        apply_fn = self._get_patch_function(model_id, "apply_px_patch")
        apply_fn(entry["model"], **patch_kwargs)
        entry["px_subjective"] = px_subjective

    def _get_patch_function(self, model_id: str, function_name: str):
        """Import patch module and get a function by name."""
        registry = MODEL_REGISTRY[model_id]
        patch_dir = registry["patch_dir"]

        # Ensure px_patches directory is on sys.path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        px_dir = os.path.join(base_dir, "px_patches")
        if px_dir not in sys.path:
            sys.path.insert(0, px_dir)

        # Import the patch module
        module_name = f"{patch_dir}.patch"
        if module_name not in sys.modules:
            mod = importlib.import_module(module_name)
        else:
            mod = sys.modules[module_name]

        fn = getattr(mod, function_name, None)
        if fn is None:
            raise AttributeError(f"Function {function_name} not found in {module_name}")
        return fn

    def get_px_metrics(self, model_id: str) -> dict:
        """Get latest PX cognitive metrics for a model."""
        registry = MODEL_REGISTRY.get(model_id, {})
        if registry.get("patch_dir") is None:
            return {"patched": False, "model_type": registry.get("model_type", "unknown")}

        if model_id not in self._models:
            return {}

        entry = self._models[model_id]
        model = entry["model"]

        # Try get_px_metrics function (MiniCPM5 version)
        try:
            get_fn = self._get_patch_function(model_id, "get_px_metrics")
            return get_fn(model)
        except (AttributeError, ImportError):
            pass

        # Fallback: read attributes directly
        text_model = self._resolve_text_model(model)
        if text_model is None:
            return {}

        return {
            "phi": getattr(text_model, "_px_phi", 1.0),
            "steps": getattr(text_model, "_px_loops_run", 0),
            "path": getattr(text_model, "_px_path", []),
            "zone": getattr(text_model, "_px_zone", "UNKNOWN"),
            "zone_weights": getattr(text_model, "_px_zone_weights", {}),
            "cognitive_signature": getattr(text_model, "_px_cognitive_signature", {}),
            "subjective": getattr(text_model, "_px_subjective_enabled", False),
        }

    def _resolve_text_model(self, model):
        """Find the transformer backbone in the model."""
        if hasattr(model, "layers") and hasattr(model, "rotary_emb"):
            return model
        for name, mod in model.named_modules():
            if hasattr(mod, "layers") and hasattr(mod, "rotary_emb"):
                return mod
        return model.model if hasattr(model, "model") else model

    def list_models(self) -> list:
        """Return list of available model IDs."""
        return list(MODEL_REGISTRY.keys())

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        return model_id in self._models

    def is_px_model(self, model_id: str) -> bool:
        """Check if a model has PX patch enabled."""
        registry = MODEL_REGISTRY.get(model_id, {})
        return registry.get("patch_dir") is not None

    def unload(self, model_id: str):
        """Unload a model from GPU memory."""
        if model_id in self._models:
            entry = self._models.pop(model_id)
            # Remove patch references before deleting model
            try:
                registry = entry.get("registry", {})
                if registry.get("patch_dir") is not None:
                    remove_fn = self._get_patch_function(model_id, "remove_px_patch")
                    if remove_fn:
                        remove_fn(entry["model"])
            except Exception:
                pass  # Best-effort cleanup
            del entry["model"]
            del entry["tokenizer"]
            if model_id in self._px_metrics:
                del self._px_metrics[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[ModelManager] {model_id} unloaded.")

    async def shutdown(self):
        """Cleanup all loaded models."""
        for model_id in list(self._models.keys()):
            entry = self._models[model_id]
            del entry["model"]
            del entry["tokenizer"]
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ModelManager] All models unloaded.")