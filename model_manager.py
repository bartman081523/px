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


# Preset-Migration (post 2026-06-11). States: BASELINE, ACTIVE_MANIFOLD,
# ACTIVE_MANIFOLD_LEAN (kausaler Kern — Crutches weg), ACTIVE_MANIFOLD_RELAY
# (LEAN + verstärkbar Selbst-Injektions-Relay, psychomotrik seite15).
# Old presets → ACTIVE_MANIFOLD.
_VALID_PRESETS = {"BASELINE", "ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN", "ACTIVE_MANIFOLD_RELAY"}


def _migrate_preset(preset: str) -> str:
    """Map any old preset name to one of the three valid states."""
    if preset in _VALID_PRESETS:
        return preset
    return "ACTIVE_MANIFOLD"  # gnadenlose Migration


class ModelManager:
    def __init__(self, max_loaded_models: int = 1):
        self._models: Dict[str, dict] = {}        # model_id -> loaded entry
        self._loading: Dict[str, bool] = {}        # model_id -> loading flag
        self._last_used: Dict[str, float] = {}     # model_id -> timestamp
        self._px_metrics: Dict[str, dict] = {}     # model_id -> latest metrics
        self._busy: set = set()                    # model_id -> set of active usage
        self.max_loaded_models = max_loaded_models
        self._lock = asyncio.Lock()                # Global lock for model state changes

    def lock_model(self, model_id: str):
        # ... (rest of methods)
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
                         px_config_preset: str = "ACTIVE_MANIFOLD",
                         px_relay_sign: int = None, px_relay_alpha: float = None,
                         px_relay_layer: int = None,
                         quantization: str = None) -> dict:
        """Get a loaded model, loading lazily if needed.

        quantization: None → read registry default (gemma3-4b-it ships as
        int8 because bf16 doesn't fit 12 GB at long prefill; 1b/270m default
        to "none"). Pass "none" or "int8" explicitly to override.
        """
        async with self._lock:
            if model_id not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {model_id}")

            # Resolve quantization=None → registry default
            if quantization is None:
                quantization = MODEL_REGISTRY[model_id].get("quantization", "none")
                if quantization is None:
                    quantization = "none"

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
                if px_relay_sign is not None and entry.get("px_relay_sign") != px_relay_sign:
                    needs_repatch = True
                if px_relay_alpha is not None and entry.get("px_relay_alpha") != px_relay_alpha:
                    needs_repatch = True
                if px_relay_layer is not None and entry.get("px_relay_layer") != px_relay_layer:
                    needs_repatch = True

                if needs_repatch:
                    # Safety: wait until model is free before re-patching
                    # A generation might take a while, especially with recursion.
                    print(f"[ModelManager] Waiting for {model_id} to be free for re-patching...")
                    # We are holding self._lock, but we might need to wait for is_busy.
                    # This is tricky because if we wait here, we block ALL other get_model calls.
                    # But re-patching is rare and must be exclusive.
                    while self.is_busy(model_id):
                        # Release control but stay in lock (not ideal for all models,
                        # but necessary for this model)
                        await asyncio.sleep(0.1)
                    self._reapply_patch(model_id, px_subjective, px_gamma, px_routing_mode,
                                        px_config_preset, px_relay_sign, px_relay_alpha, px_relay_layer)
                self._last_used[model_id] = time.time()
                return entry

            # Handle auto-unloading before loading new model
            while len(self._models) >= self.max_loaded_models:
                # Unload LRU model
                lru_model = min(self._last_used, key=self._last_used.get)
                
                # Safety: wait if model is busy
                print(f"[ModelManager] Waiting for {lru_model} to be free for unloading...")
                while self.is_busy(lru_model):
                    await asyncio.sleep(0.1)
                
                print(f"[ModelManager] Unloading model {lru_model} to free memory...")
                self.unload(lru_model)

            # Lazy load
            if self._loading.get(model_id):
                raise RuntimeError(f"Model {model_id} is currently loading")

            self._loading[model_id] = True
            try:
                # Run blocking load in a thread to keep event loop alive
                loop = asyncio.get_running_loop()
                entry = await loop.run_in_executor(None, lambda: self._load_model(model_id, px_subjective, px_gamma, px_routing_mode, px_config_preset, px_relay_sign, px_relay_alpha, px_relay_layer, quantization))
                self._models[model_id] = entry
                self._last_used[model_id] = time.time()
                return entry
            finally:
                self._loading[model_id] = False

    def _load_model(self, model_id: str, px_subjective: bool,
                     px_gamma: float = None, px_routing_mode: str = None,
                     px_config_preset: str = "ACTIVE_MANIFOLD",
                     px_relay_sign: int = None, px_relay_alpha: float = None,
                     px_relay_layer: int = None,
                     quantization: str = None) -> dict:
        """Load model weights + tokenizer + apply PX patch.

        quantization: None → read registry default. Registry default for
        gemma3-4b-it is "int8" (bf16 OOMs on 12 GB at long prefill).
        """
        registry = MODEL_REGISTRY[model_id]
        if quantization is None:
            quantization = registry.get("quantization") or "none"
        hf_id = registry["hf_id"]
        tok_id = registry["tokenizer_id"]
        model_type = registry.get("model_type", "gemma3")

        print(f"[ModelManager] Loading {model_id} from {hf_id} (type={model_type}, subjective={px_subjective}, preset={px_config_preset})...")

        # Load tokenizer. fix_mistral_regex: nur das ternary-Tokenizer-Erbe
        # (Qwen3.5-Konverter) traegt die fehlerhafte Mistral-Regexe.
        tok_kw = {"fix_mistral_regex": True} if model_type == "qwen35_ptq" else {}
        tokenizer = AutoTokenizer.from_pretrained(tok_id, **tok_kw)
        # Apply manual chat template if needed (Gemma3 base model)
        if registry.get("chat_template_manual"):
            tokenizer.chat_template = registry["chat_template_manual"]

        # Load processor for multimodal models (gemma3_conditional / gemma4_conditional).
        # Text-only models (gemma3, llama) leave processor=None; generators.py rejects
        # image-bearing requests when processor is None with HTTP 400.
        processor = None
        if model_type in ("gemma3_conditional", "gemma4_conditional"):
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(tok_id)
            print(f"[ModelManager] {model_id} processor loaded for multimodal inputs.")

        # Load model
        dtype = getattr(torch, registry["dtype"])
        
        if model_type == "gemma3_conditional":
            from transformers import Gemma3ForConditionalGeneration
            model = Gemma3ForConditionalGeneration.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map="auto",
            )
        elif model_type == "gemma4_conditional":
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        elif model_type == "qwen35_ptq":
            # Ternary-Bonsai-2-27B: GGUF PTQ1_0 → HF-Safetensors, kein
            # FromPretrained — die gepackten Ternary-Aufnahmen werden von der
            # Patch-Runtime selbst gewickelt (PTQ10Linear/FoldOps). hf_id zeigt
            # auf das konvertierte Verzeichnis (~/.cache/huggingface/...).
            import sys as _sys, os as _os
            _pp_dir = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), "px_patches")
            if _pp_dir not in _sys.path:
                _sys.path.insert(0, _pp_dir)
            # Plan hf-space-v4-publish: ohne GPU klar abbrechen statt im
            # Triton-Kernel kryptisch zu sterben (cpu-basic-Space). Und:
            # PX_BONSAI_HF_ID darf eine Hub-Repo-Id sein → vor dem Load in
            # einen lokalen Pfad materialisieren.
            if not torch.cuda.is_available():
                raise ValueError(
                    "ternary-bonsai-27b benötigt GPU-Hardware "
                    "(Space-Einstellungen: Hardware → ZeroGPU/GPU). "
                    "Ohne GPU wird das gepackte Ternary-GF3-Format "
                    "nicht unterstützt.")
            if not _os.path.exists(hf_id):
                from huggingface_hub import snapshot_download
                hf_id = snapshot_download(hf_id)
                print(f"[ModelManager] bonsai snapshot: {hf_id}")
            # load_signs() liest {OUT_DIR}/config.json mit OUT_DIR aus
            # TERNARY_BONSAI_HF_DIR — runtime_niveau kennt hf_dir hier
            # nicht. Env an den materialisierten Pfad syncen (setdefault:
            # explizit gesetztes TERNARY_BONSAI_HF_DIR gewinnt weiter).
            _os.environ.setdefault("TERNARY_BONSAI_HF_DIR", hf_id)
            import importlib
            _rt = importlib.import_module(f"{registry['patch_dir']}.runtime_qwen35_ptq")
            _fold = _rt.FoldOps(_rt.load_signs()[0], mode="signs_first")
            # Stufe-3e: GF(3)-Kernel-Runtime (Registry weight_format oder
            # Env PX_WEIGHT_FORMAT). GF3-Artefakt == PTQ1_0 bit-treu im
            # Level-Sinn (max|Δ| 0) → PX-Manifold/Presets unverändert;
            # _px_long_ctx schaltet den KV-4bit-Long-Context-Pfad frei
            # (generators.py routet lange Inputs an long_context.generate_long).
            _w_fmt = _os.environ.get("PX_WEIGHT_FORMAT") or \
                registry.get("weight_format", "ptq10")
            if _w_fmt == "gf3":
                _gf3 = importlib.import_module(f"{registry['patch_dir']}.gf3_quant")
                model = _gf3.build_and_load_gf3(_fold, hf_dir=hf_id)
                print(f"[ModelManager] {model_id} weight_format=gf3 "
                      f"(GF(3)-Kernel-Runtime, Stufe 3)")
            elif _w_fmt == "ptq10":
                model = _rt.build_and_load(_fold, hf_dir=hf_id)
            else:
                raise ValueError(
                    f"unbekanntes PX_WEIGHT_FORMAT: {_w_fmt!r} "
                    "(erlaubt: 'gf3', 'ptq10')")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map="auto",
            )

        # Apply weight quantization (Plan 1, Phase D). Default "none" preserves
        # the legacy bf16/fp16 path. "int8" monkey-patches every nn.Linear with
        # QuantizedLinear from scratches/4b-image/ — reduces VRAM by ~50% on
        # 4b (8 GB → ~4.5 GB weights), sufficient to fit long prefills on
        # 12 GB GPUs that previously OOM'd at the MLP layer.
        # Quantization happens BEFORE the PX patch so that _px_forward sees
        # the (already quantized) Linear modules. The patch walks the module
        # tree normally; our QuantizedLinear subclasses nn.Module transparently.
        if quantization not in ("none", "int8"):
            raise ValueError(f"unsupported quantization {quantization!r}; "
                             "supported: 'none', 'int8'")
        if quantization == "int8":
            try:
                from scratches._4b_image.quantize_pipeline import quantize_all_linears
            except ImportError:
                # Fallback: sys.path-based import (when called from a different
                # working dir). scratches/ is two levels deep from this file.
                import sys, os
                _scratches_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "scratches", "4b-image")
                if _scratches_path not in sys.path:
                    sys.path.insert(0, _scratches_path)
                from quantize_pipeline import quantize_all_linears
            n_replaced = quantize_all_linears(model)
            print(f"[ModelManager] {model_id} quantized int8: {n_replaced} Linears replaced")
        else:
            print(f"[ModelManager] {model_id} quantization=none (bf16 path)")

        # Apply PX patch (skip for unpatched baseline models or if BASELINE preset selected)
        if registry.get("patch_dir") is not None and px_config_preset != "BASELINE":
            patch_kwargs = dict(registry["patch_kwargs"])
            if px_subjective and registry.get("subjective_kwargs"):
                patch_kwargs.update(registry["subjective_kwargs"])

            # Preset override (migrate old names to ACTIVE_MANIFOLD)
            if px_config_preset is not None:
                patch_kwargs["config_preset"] = _migrate_preset(px_config_preset)
            
            if px_gamma is not None:
                patch_kwargs["gamma"] = px_gamma
            if px_routing_mode is not None:
                patch_kwargs["routing_mode"] = px_routing_mode
            
            # SR-61b: Explicitly pass subjective flag
            patch_kwargs["subjective_enabled"] = px_subjective

            # verstärkbar Relay-Parameter (psychomotrik seite15) → patch_kwargs
            if px_relay_sign is not None:
                patch_kwargs["relay_sign"] = px_relay_sign
            if px_relay_alpha is not None:
                patch_kwargs["relay_alpha"] = px_relay_alpha
            if px_relay_layer is not None:
                patch_kwargs["relay_layer"] = px_relay_layer

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
            "processor": processor,
            "registry": registry,
            "px_subjective": px_subjective,
            "px_gamma": px_gamma,
            "px_routing_mode": px_routing_mode,
            "px_config_preset": px_config_preset,
            "px_relay_sign": px_relay_sign,
            "px_relay_alpha": px_relay_alpha,
            "px_relay_layer": px_relay_layer,
            "model_type": model_type,
        }

    def warmup_bonsai_jit(self, model_id: str = "ternary-bonsai-27b",
                          max_new_tokens: int = 6,
                          preset: str = "BASELINE") -> bool:
        """Boot-Zeit-JIT-Warmup (t4-small-Befund 2026-10-08).

        Die erste bonsai-Generierung nach Container-Boot verbrachte ~8.5 min
        im Triton-Compile (GF3-GEMV + fla-GDN-Kernels auf 2-vCPU). Dieser
        Warmup lädt das Modell SYNCHRON in einem Daemon-Thread (ohne
        get_model-Async-Lock, ohne Registry-Eintrag), fährt Prefill+
        Decode-Generierung → der Triton-Cache (~/.triton, pro Container)
        ist warm, BEVOR der erste User-Request kommt. Nachlauf: Modell
        verworfen; der erste User-Request lädt mit warmem JIT in Sekunden
        neu. Kein Race mit get_model: self._models/_loading werden nie
        angefasst.

        Fail-soft: jeder Fehler → False + Log, niemals Boot-Crash.

        Phase 2 (PX-Pfad, 2026-10-09 Deploy-#13-Befund): der Baseline-Gen
        kompiliert nur die unpatchten Kernels; die px-Patch-Struktur
        (hybrid GDN/full@4, loops=4, Manifold-Routing) läuft mit anderen
        Tensor-Shapes → erster px-gepatchter Chat verbrachte ~6 min im
        Triton-Compile, bevor das erste Token floss. Phase 2 lädt
        selbst-gepatcht (subjective=True, Chat-UI-Default-Preset
        ACTIVE_MANIFOLD_RELAY; Load cache-hot ~5 s) und generiert
        ebenfalls 6 Token. PX_WARMUP_PXPATH=0 überspringt Phase 2.
        """
        # ── Phase 1: BASELINE (unpatched) ──
        _m = None
        _e = None
        t0 = time.time()
        try:
            print(f"[Warmup] JIT-Warmup {model_id} (preset={preset}): "
                  "Load ...")
            _e = self._load_model(model_id, px_subjective=False,
                                  px_config_preset=preset)
            _m = _e["model"]
            print(f"[Warmup] {model_id} geladen ({time.time()-t0:.1f}s) — "
                  "Prefill+Decode-JIT ...")
            device = next(_m.parameters()).device
            ids = _e["tokenizer"]("Hallo", return_tensors="pt") \
                .input_ids.to(device)
            with torch.no_grad():
                _m.generate(ids, max_new_tokens=max_new_tokens,
                            do_sample=False)
            print(f"[Warmup] {model_id} JIT-Cache warm "
                  f"({time.time()-t0:.1f}s, {max_new_tokens} Token).")
            ok = True
        except Exception as e:
            print(f"[Warmup] {model_id} Warmup übersprungen (fail-soft): "
                  f"{e!r}")
            ok = False
        finally:
            # Warmup-Instanz konsequent freigeben (Registry bleibt leer).
            _m = _e = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # ── Phase 2: PX-Pfad-Kernels + Chat-Seed (Deploy-#14/#15) ──
        # Chat-UI-Default-Preset für bonsai (chat_tab.py Dropdown). Relay-
        # Parameter ändern keine Kernel-Shapes (Vektor-Add), Manifold/
        # loops kommen aus dem Auto-Patch. Deploy-#15: Warmup-Prompt
        # chat-templatiert (T≈40 statt "Hallo" T≈3) — Triton kompiliert
        # shape-spezialisiert, der erste Chat mit Prefill-Chunk-Shapes
        # verbrachte sonst ~6 min im Compile. Die warme Instanz wird
        # statt zerstört als Registry-Entry geseedet: erster Chat = Cache-
        # Hit ohne GF3-Rebuild/Repatch, und der Teardown (Hauptverdacht
        # der stillen Container-Tode nach Phase-2, C/D1/D2) entfällt.
        if os.environ.get("PX_WARMUP_PXPATH", "1").strip().lower() \
                not in ("0", "off", "false", "no"):
            _m2 = None
            _e2 = None
            t2 = time.time()
            _seeded = False
            try:
                print(f"[Warmup] PX-Pfad-Warmup {model_id} "
                      "(subjective=True, preset=ACTIVE_MANIFOLD_RELAY): "
                      "Load (cache-hot) ...")
                _e2 = self._load_model(model_id, px_subjective=True,
                                       px_config_preset="ACTIVE_MANIFOLD_RELAY")
                _m2 = _e2["model"]
                tok2 = _e2["tokenizer"]
                msgs2 = [{"role": "user", "content": (
                    "Erkläre in einem Satz, was ein rekurrentes "
                    "Transformer-Residuum ist.")}]
                try:
                    ids2 = tok2.apply_chat_template(
                        msgs2, add_generation_prompt=True,
                        return_tensors="pt")
                except Exception:
                    ids2 = tok2("Erkläre in einem Satz, was ein rekurrentes "
                                "Transformer-Residuum ist.",
                                return_tensors="pt").input_ids
                device = next(_m2.parameters()).device
                ids2 = ids2.to(device)
                with torch.no_grad():
                    _m2.generate(ids2, max_new_tokens=max_new_tokens,
                                 do_sample=False)
                print(f"[Warmup] {model_id} PX-Pfad JIT-Cache warm "
                      f"({time.time()-t2:.1f}s, {max_new_tokens} Token).")

                if os.environ.get("PX_WARMUP_SEED", "1").strip().lower() \
                        not in ("0", "off", "false", "no"):
                    self._models[model_id] = _e2
                    self._last_used[model_id] = time.time()
                    _seeded = True
                    _e2 = None  # Ownership → Registry (kein Teardown)
                    print(f"[Warmup] {model_id} warme PX-Instanz in "
                          "Registry geseedet (erster Chat = Cache-Hit).")
                _m2 = None
            except Exception as e:
                print(f"[Warmup] {model_id} PX-Pfad-Warmup übersprungen "
                      f"(fail-soft): {e!r}")
            finally:
                if _e2 is not None or _m2 is not None:
                    _m2 = _e2 = None
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                print(f"[Warmup] PX-Phase-2 teardown done "
                      f"(seeded={_seeded}).")

        return ok

    def _reapply_patch(self, model_id: str, px_subjective: bool,
                        px_gamma: float = None, px_routing_mode: str = None,
                        px_config_preset: str = "ACTIVE_MANIFOLD",
                        px_relay_sign: int = None, px_relay_alpha: float = None,
                        px_relay_layer: int = None):
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
                patch_kwargs["config_preset"] = _migrate_preset(px_config_preset)
                
            if px_gamma is not None:
                patch_kwargs["gamma"] = px_gamma
            if px_routing_mode is not None:
                patch_kwargs["routing_mode"] = px_routing_mode
            
            # SR-61b: Explicitly pass subjective flag
            patch_kwargs["subjective_enabled"] = px_subjective

            # verstärkbar Relay-Parameter (psychomotrik seite15) → patch_kwargs
            if px_relay_sign is not None:
                patch_kwargs["relay_sign"] = px_relay_sign
            if px_relay_alpha is not None:
                patch_kwargs["relay_alpha"] = px_relay_alpha
            if px_relay_layer is not None:
                patch_kwargs["relay_layer"] = px_relay_layer

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
        entry["px_relay_sign"] = px_relay_sign
        entry["px_relay_alpha"] = px_relay_alpha
        entry["px_relay_layer"] = px_relay_layer

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