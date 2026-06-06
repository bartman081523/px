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
import asyncio
from typing import Dict, Optional

from config import MODEL_REGISTRY
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelManager:
    def __init__(self, max_loaded_models: int = 1):
        self._models: Dict[str, dict] = {}        # model_id -> loaded entry
        self._loading: Dict[str, bool] = {}        # model_id -> loading flag
        self._last_used: Dict[str, float] = {}     # model_id -> timestamp
        self._px_metrics: Dict[str, dict] = {}     # model_id -> latest metrics
        self._busy: set = set()                    # model_id -> set of active usage
        self.max_loaded_models = max_loaded_models

    def lock_model(self, model_id: str):
        """Mark a model as busy (in use by a thread)."""
        self._busy.add(model_id)

    def unlock_model(self, model_id: str):
        """Mark a model as free."""
        if model_id in self._busy:
            self._busy.remove(model_id)

    def is_busy(self, model_id: str) -> bool:
        """Check if a model is currently in use."""
        return model_id in self._busy

    async def get_model(self, model_id: str, px_subjective: bool = False,
                         px_gamma: float = None, px_routing_mode: str = None,
                         px_config_preset: str = None) -> dict:
        """Get a loaded model, loading lazily if needed.

        If model is loaded with different subjective/gamma/routing mode,
        re-apply the patch (no weight reload needed).
        """
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_id}")

        if model_id in self._models:
            entry = self._models[model_id]
            current_subjective = entry.get("px_subjective", False)
            current_preset = entry.get("px_config_preset")
            
            needs_repatch = current_subjective != px_subjective
            if px_config_preset is not None and current_preset != px_config_preset:
                needs_repatch = True
            if px_gamma is not None and entry.get("px_gamma") != px_gamma:
                needs_repatch = True
            if px_routing_mode is not None and entry.get("px_routing_mode") != px_routing_mode:
                needs_repatch = True
                
            if needs_repatch:
                # Safety: wait if model is busy before re-patching
                wait_start = time.time()
                while self.is_busy(model_id):
                    if time.time() - wait_start > 10.0: # 10s timeout
                        print(f"[ModelManager] Warning: Timeout waiting for {model_id} to be free for re-patching.")
                        break
                    await asyncio.sleep(0.1)
                self._reapply_patch(model_id, px_subjective, px_gamma, px_routing_mode, px_config_preset)
            self._last_used[model_id] = time.time()
            return entry

        # Handle auto-unloading before loading new model
        while len(self._models) >= self.max_loaded_models:
            # Unload LRU model
            lru_model = min(self._last_used, key=self._last_used.get)
            
            # Safety: wait if model is busy
            wait_start = time.time()
            is_timed_out = False
            while self.is_busy(lru_model):
                if time.time() - wait_start > 30.0: # 30s timeout for unloading
                    print(f"[ModelManager] CRITICAL: Timeout waiting for {lru_model} to be free for unloading.")
                    is_timed_out = True
                    break
                await asyncio.sleep(0.1)
            
            if is_timed_out:
                # Try another model if capacity > 1, else we have to fail or force it
                if self.max_loaded_models > 1:
                    # Sort models by last_used and pick the first non-busy one
                    sorted_models = sorted(self._last_used.keys(), key=lambda k: self._last_used[k])
                    found_alternative = False
                    for m in sorted_models:
                        if not self.is_busy(m):
                            lru_model = m
                            found_alternative = True
                            break
                    if not found_alternative:
                        raise RuntimeError("All loaded models are busy and cannot be unloaded.")
                else:
                    print(f"[ModelManager] Force-unloading busy model {lru_model} due to timeout.")

            print(f"[ModelManager] Unloading model {lru_model} to free memory...")
            self.unload(lru_model)

        # Lazy load
        if self._loading.get(model_id):
            raise RuntimeError(f"Model {model_id} is currently loading")

        self._loading[model_id] = True
        try:
            entry = self._load_model(model_id, px_subjective, px_gamma, px_routing_mode, px_config_preset)
            self._models[model_id] = entry
            self._last_used[model_id] = time.time()
            return entry
        finally:
            self._loading[model_id] = False

    def _load_model(self, model_id: str, px_subjective: bool,
                     px_gamma: float = None, px_routing_mode: str = None,
                     px_config_preset: str = None) -> dict:
        """Load model weights + tokenizer + apply PX patch."""
        registry = MODEL_REGISTRY[model_id]
        hf_id = registry["hf_id"]
        tok_id = registry["tokenizer_id"]
        model_type = registry.get("model_type", "gemma3")

        print(f"[ModelManager] Loading {model_id} from {hf_id} (type={model_type}, subjective={px_subjective}, preset={px_config_preset})...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_id)
        # Apply manual chat template if needed (Gemma3 base model)
        if registry.get("chat_template_manual"):
            tokenizer.chat_template = registry["chat_template_manual"]

        # Load model
        dtype = getattr(torch, registry["dtype"])
        
        if model_type == "gemma3_conditional":
            from transformers import Gemma3ForConditionalGeneration
            model = Gemma3ForConditionalGeneration.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map="auto",
            )

        # Apply PX patch (skip for unpatched baseline models or if BASELINE preset selected)
        if registry.get("patch_dir") is not None and px_config_preset != "BASELINE":
            patch_kwargs = dict(registry["patch_kwargs"])
            if px_subjective and registry.get("subjective_kwargs"):
                patch_kwargs.update(registry["subjective_kwargs"])
            
            # Preset override
            if px_config_preset is not None:
                patch_kwargs["config_preset"] = px_config_preset
            
            if px_gamma is not None:
                patch_kwargs["gamma"] = px_gamma
            if px_routing_mode is not None:
                patch_kwargs["routing_mode"] = px_routing_mode

            apply_fn = self._get_patch_function(model_id, "apply_px_patch")
            apply_fn(model, **patch_kwargs)
            tm = self._resolve_text_model(model)
            model.tokenizer = tm.tokenizer = tokenizer 
            print(f"[ModelManager] {model_id} loaded and patched successfully.")
        else:
            print(f"[ModelManager] {model_id} loaded WITHOUT patch (baseline).")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "registry": registry,
            "px_subjective": px_subjective,
            "px_gamma": px_gamma,
            "px_routing_mode": px_routing_mode,
            "px_config_preset": px_config_preset,
            "model_type": model_type,
        }

    def _reapply_patch(self, model_id: str, px_subjective: bool,
                        px_gamma: float = None, px_routing_mode: str = None,
                        px_config_preset: str = None):
        """Re-apply PX patch with different settings (no weight reload)."""
        entry = self._models[model_id]
        registry = entry["registry"]

        print(f"[ModelManager] Re-patching {model_id} (subjective={px_subjective}, gamma={px_gamma}, routing={px_routing_mode}, preset={px_config_preset})...")

        # Remove existing patch
        remove_fn = self._get_patch_function(model_id, "remove_px_patch")
        if remove_fn:
            try:
                remove_fn(entry["model"])
            except Exception as e:
                print(f"[ModelManager] Warning: remove_px_patch failed: {e}")

        # Re-apply with new settings (if not BASELINE)
        if px_config_preset != "BASELINE":
            patch_kwargs = dict(registry["patch_kwargs"])
            if px_subjective and registry.get("subjective_kwargs"):
                patch_kwargs.update(registry["subjective_kwargs"])
            
            if px_config_preset is not None:
                patch_kwargs["config_preset"] = px_config_preset
                
            if px_gamma is not None:
                patch_kwargs["gamma"] = px_gamma
            if px_routing_mode is not None:
                patch_kwargs["routing_mode"] = px_routing_mode

            apply_fn = self._get_patch_function(model_id, "apply_px_patch")
            apply_fn(entry["model"], **patch_kwargs)
            tm = self._resolve_text_model(entry["model"])
            entry["model"].tokenizer = tm.tokenizer = entry["tokenizer"]
        else:
            print(f"[ModelManager] {model_id} returned to baseline state.")
        
        entry["px_subjective"] = px_subjective
        entry["px_gamma"] = px_gamma
        entry["px_routing_mode"] = px_routing_mode
        entry["px_config_preset"] = px_config_preset

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
            if model_id in self._last_used:
                del self._last_used[model_id]
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