"""server_v35g.py — v3.5g OpenAI-Server (Standalone, in scratches/).

Service: gemma3-270m-it mit allen 9 PX-Preset-Arms aus v3.5g als /v1/models.
Default-Preset: ACTIVE_MANIFOLD_LEAN (v3.5f-bevorzugt, 5/32 HLE).
Optional: OFFICIAL_RIGOR als experimentelles Preset (degeneriert empirisch).

Hinweis: In scratches/ — kein Production-Code berührt.
"""
from __future__ import annotations

import time
import uuid
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import sys
import os

_VENV = "/run/media/julian/ML4/open-mythos_p2/venv_openmythos/lib/python3.10/site-packages"
_REPO = "/run/media/julian/ML4/ollama-work/all_space_6_16_stand"
sys.path.insert(0, _VENV)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v1"))
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v2"))
sys.path.insert(0, os.path.join(_REPO, "scratches/rigor_zone_v3"))

os.environ['HF_HUB_OFFLINE'] = '0'

from transformers import AutoTokenizer, AutoModelForCausalLM
from px_patches_v3.patch import apply_px_patch
from rigor_scales_v3 import ARM_CONFIGS


# ═══════════════════════════════════════════════════════════════════════════════
# 9 Modell-IDs (alle auf 270m, je 1 pro v3.5g-Arm)
# ═══════════════════════════════════════════════════════════════════════════════

HF_MODEL = "google/gemma-3-270m-it"

# Map: arm_name -> preset für apply_px_patch
# (arm.preset direkt, aber OFFICIAL_RIGOR muss via preset="OFFICIAL_RIGOR" laufen)
MODELS = {
    "gemma3-270m-px-baseline":         {"preset": "BASELINE",          "v3_5g_arm": "baseline"},
    "gemma3-270m-px-active-manifold":  {"preset": "ACTIVE_MANIFOLD",   "v3_5g_arm": "active_manifold"},
    "gemma3-270m-px-lean":             {"preset": "ACTIVE_MANIFOLD_LEAN", "v3_5g_arm": "active_manifold_lean", "default": True},
    "gemma3-270m-px-rigor-disabled":   {"preset": "ACTIVE_MANIFOLD",   "v3_5g_arm": "rigor_disabled",  "mephisto": False, "scale": 0.0},
    "gemma3-270m-px-rigor-least":      {"preset": "ACTIVE_MANIFOLD",   "v3_5g_arm": "rigor_least",     "mephisto": True, "scale": 0.1},
    "gemma3-270m-px-rigor-low":        {"preset": "ACTIVE_MANIFOLD",   "v3_5g_arm": "rigor_low",       "mephisto": True, "scale": 0.3},
    "gemma3-270m-px-rigor-mid":        {"preset": "ACTIVE_MANIFOLD",   "v3_5g_arm": "rigor_mid",       "mephisto": True, "scale": 0.5},
    "gemma3-270m-px-rigor-high":       {"preset": "ACTIVE_MANIFOLD",   "v3_5g_arm": "rigor_high",      "mephisto": True, "scale": 0.7},
    "gemma3-270m-px-rigor-full":       {"preset": "ACTIVE_MANIFOLD",   "v3_5g_arm": "rigor_full",      "mephisto": True, "scale": 1.0},
    "gemma3-270m-px-official-rigor":   {"preset": "OFFICIAL_RIGOR",    "v3_5g_arm": "official_rigor",  "mephisto": True, "scale": 0.3, "experimental": True},
}

DEFAULT_MODEL = "gemma3-270m-px-lean"  # v3.5f bevorzugt


# ═══════════════════════════════════════════════════════════════════════════════
# Model-Manager (Lazy-Loading, ein Modell zur Zeit)
# ═══════════════════════════════════════════════════════════════════════════════

class ModelManager:
    def __init__(self):
        # v3.5g-Multi-Model-Cache: Dict[model_id, (tokenizer, model)]
        # Vorher: single-slot _model/_current_model_id → Switch entlud + lud neu (3s waste)
        # Jetzt: alle geladenen Modelle bleiben im Speicher, Switch ohne Reload.
        self._models: Dict[str, tuple] = {}
        self._lock = False

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    def get_model(self, model_id: str):
        if model_id not in MODELS:
            raise HTTPException(404, f"Model {model_id} not in registry")
        # Cache-Hit: return ohne I/O
        if self.is_loaded(model_id):
            tok, mdl = self._models[model_id]
            return tok, mdl
        # Cache-Miss: lade + cache
        cfg = MODELS[model_id]
        print(f"[v3.5g-Server] Loading {model_id} (preset={cfg['preset']})...", flush=True)
        tok = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL, dtype=torch.bfloat16
        ).to("cuda").eval()
        if cfg["preset"] != "BASELINE":
            apply_px_patch(model.model, config_preset=cfg["preset"])
        self._models[model_id] = (tok, model)
        print(f"[v3.5g-Server] {model_id} loaded. (cache_size={len(self._models)})", flush=True)
        return tok, model

    def list_loaded(self) -> List[str]:
        """Welche Modelle sind gerade im Cache."""
        return list(self._models.keys())

    def _unload(self, model_id: Optional[str] = None):
        # Vorher: _unload() entlud das einzige Modell. Jetzt: optional per model_id.
        if model_id is None:
            # Unload all (für shutdown)
            for mid in list(self._models.keys()):
                self._unload(mid)
            return
        if model_id in self._models:
            _, mdl = self._models.pop(model_id)
            del mdl
            torch.cuda.empty_cache()

    async def shutdown(self):
        self._unload()


class CudaGraphRunnerCache:
    """Cache für CUDAGraphRunner pro Modell + Shape (B, max_seq).

    Motivation: model.generate() hat 197ms CUDA + 1100ms CPU-Overhead
    (57616 cudaLaunchKernel, 1083 cudaStreamSynchronize). CUDA-Graph captured
    die Decode-Steps in EINEM Graph-Op → 4.96× Speedup (vgl.
    test_cuda_graph_integration.py).

    Pro Modell: Dict[(B, max_seq), {runner, max_new, last_used}].
    LRU-Eviction bei max_runners_per_model.
    """

    def __init__(self, max_runners_per_model: int = 4):
        self._runners: Dict[str, List[Dict]] = {}  # model_id → List of {key, runner, max_new, last_used}
        self._max_per_model = max_runners_per_model

    def get_or_create(self, model_id: str, model, tokenizer,
                      input_ids, attention_mask, max_new_tokens: int):
        """Gibt gecachten Runner zurück oder erstellt neuen.

        B = batch_size (immer 1 für jetzt)
        max_seq = input_len + max_new + buffer (für jeden Request anders)
        Wir bucketing nach max_new (da sich das selten ändert).
        """
        # Lazy import (sonst kann der Test CUDAGraphRunner nicht mocken)
        from px_patches_v3 import cuda_graph_runner as _cgr
        CUDAGraphRunner = _cgr.CUDAGraphRunner
        CUDAGraphRunnerConfig = _cgr.CUDAGraphRunnerConfig
        B = input_ids.shape[0]
        # Bucket: max_new (vereinfacht — in Praxis könnte man input_len mit reinnehmen)
        key = (B, max_new_tokens)
        now = time.time()

        # Cache-Hit
        if model_id in self._runners:
            for entry in self._runners[model_id]:
                if entry["key"] == key:
                    entry["last_used"] = now
                    return entry["runner"]

        # Cache-Miss: erstelle neuen Runner
        T_in = input_ids.shape[1]
        max_seq = int(T_in * 2.5) + max_new_tokens + 100
        cfg = CUDAGraphRunnerConfig(batch_size=B, max_seq_len=max_seq)
        try:
            runner = CUDAGraphRunner(model, cfg)
            runner.setup(input_ids, attention_mask)
        except Exception as e:
            print(f"  [WARN] CUDA-Graph setup failed for {model_id}/{key}: "
                  f"{type(e).__name__}: {str(e)[:100]}", flush=True)
            return None

        # Cache hinzufügen
        if model_id not in self._runners:
            self._runners[model_id] = []
        self._runners[model_id].append({
            "key": key, "runner": runner,
            "max_new": max_new_tokens, "last_used": now,
        })
        # LRU-Eviction
        if len(self._runners[model_id]) > self._max_per_model:
            self._runners[model_id].sort(key=lambda e: e["last_used"])
            evicted = self._runners[model_id].pop(0)
            del evicted["runner"]
            torch.cuda.empty_cache()
        print(f"  [v3.5g-Server] CUDA-Graph cached for {model_id} "
              f"(key={key}, max_per_model={self._max_per_model}, "
              f"current={len(self._runners[model_id])})", flush=True)
        return runner

    def generate(self, runner, max_new_tokens: int) -> Any:
        """Decode max_new_tokens via CUDA-Graph runner. Returns token sequence (1, max_new)."""
        tokens = [runner.static_input_ids.clone()]
        for _ in range(max_new_tokens - 1):
            nxt = runner.step()
            runner.append(nxt)
            tokens.append(nxt)
        return torch.cat(tokens, dim=1)

    def clear_model(self, model_id: str):
        """Alle Runner für ein Modell löschen (z.B. wenn Modell entladen wird)."""
        if model_id in self._runners:
            for entry in self._runners[model_id]:
                del entry["runner"]
            del self._runners[model_id]
            torch.cuda.empty_cache()


manager = ModelManager()
runner_cache = CudaGraphRunnerCache(max_runners_per_model=4)


# ═══════════════════════════════════════════════════════════════════════════════
# Schemas (Pydantic, OpenAI-kompatibel)
# ═══════════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    temperature: float = 0.0  # PX → greedy
    top_p: float = 1.0
    max_tokens: int = 200
    stream: bool = False
    stop: Optional[List[str]] = None

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "px-v3.5g"
    v3_5g_arm: Optional[str] = None
    experimental: bool = False
    default: bool = False

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ═══════════════════════════════════════════════════════════════════════════════
# Anthropic Messages API (für cc-symbolic px-Mode)
# ═══════════════════════════════════════════════════════════════════════════════

class AnthropicMessage(BaseModel):
    role: str
    content: Any  # string oder list of blocks

class AnthropicRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[AnthropicMessage]
    max_tokens: int = 200
    system: Optional[Any] = None
    temperature: float = 0.0
    stream: bool = False
    stop_sequences: Optional[List[str]] = None

class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str

class AnthropicResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:24]}")
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicContentBlock]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: Optional[str] = None
    usage: Dict[str, int]


# ═══════════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app):
    print("[v3.5g-Server] Starting on port 7860 (gemma3-270m-it, 9 PX-Preset-Arms)")
    print(f"[v3.5g-Server] Default model: {DEFAULT_MODEL}")
    yield
    await manager.shutdown()
    print("[v3.5g-Server] Stopped.")


app = FastAPI(title="PX v3.5g Server", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Endpoints ──

@app.get("/")
async def root():
    return {
        "status": "ok",
        "server": "v3.5g",
        "model": HF_MODEL,
        "default_model": DEFAULT_MODEL,
        "available_models": list(MODELS.keys()),
        "loaded": manager.list_loaded(),
    }


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    models = []
    for mid, cfg in MODELS.items():
        models.append(ModelInfo(
            id=mid,
            v3_5g_arm=cfg.get("v3_5g_arm"),
            experimental=cfg.get("experimental", False),
            default=cfg.get("default", False),
        ))
    return ModelListResponse(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model_id = request.model or DEFAULT_MODEL
    tok, model = manager.get_model(model_id)

    # Chat-Format
    msgs = [{"role": m.role, "content": m.content} for m in request.messages]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")

    # Greedy (PX-Default, v3.5f-Befund)
    do_sample = request.temperature > 0.0

    def _gen():
        # CUDA-Graph-Pfad: nur für PX-Arms (baseline nutzt model.generate direkt).
        # Wenn Runner verfügbar: 4.96× speedup (197ms CUDA → 43ms CUDA, weniger CPU-Overhead).
        runner = None
        if model_id != "gemma3-270m-px-baseline":
            runner = runner_cache.get_or_create(
                model_id=model_id,
                model=model,
                tokenizer=tok,
                input_ids=ids["input_ids"],
                attention_mask=ids.get("attention_mask"),
                max_new_tokens=request.max_tokens,
            )
        if runner is not None:
            # CUDA-Graph-Decode
            with torch.inference_mode():
                tokens = runner_cache.generate(runner, request.max_tokens)
            new_tokens = tokens  # (1, max_new)
            gen_text = tok.decode(new_tokens[0], skip_special_tokens=True)
            return gen_text, new_tokens.shape[1], ids["input_ids"].shape[1]
        # Fallback: model.generate (für baseline oder CUDA-Graph-Setup-Fehler)
        with torch.inference_mode():
            out = model.generate(
                **ids,
                max_new_tokens=request.max_tokens,
                do_sample=do_sample,
                temperature=request.temperature if do_sample else 1.0,
                top_p=request.top_p,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = out[0, ids["input_ids"].shape[1]:]
        gen_text = tok.decode(new_tokens, skip_special_tokens=True)
        return gen_text, new_tokens.shape[0], ids["input_ids"].shape[1]

    if request.stream:
        # Streaming: wir geben das ganze Stück am Stück (kein Token-Streaming hier — 270m ist schnell)
        gen_text, n_new, n_prompt = _gen()
        async def _streamer():
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": gen_text},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_streamer(), media_type="text/event-stream")

    # Non-streaming
    gen_text, n_new, n_prompt = _gen()
    return ChatCompletionResponse(
        model=model_id,
        choices=[ChatChoice(
            index=0,
            message=ChatMessage(role="assistant", content=gen_text),
            finish_reason="stop",
        )],
        usage=Usage(prompt_tokens=n_prompt, completion_tokens=n_new, total_tokens=n_prompt + n_new),
    )


# ── Anthropic Messages API (für cc-symbolic px-Mode) ──

@app.post("/v1/messages")
async def messages(request: AnthropicRequest):
    """Anthropic Messages API → OpenAI-Pfad intern."""
    model_id = request.model or DEFAULT_MODEL
    tok, model = manager.get_model(model_id)

    # System-Prompt + Messages in OpenAI-Format
    oai_messages = []
    if request.system:
        sys_text = request.system if isinstance(request.system, str) else \
            "\n\n".join(b.get("text", "") for b in request.system if isinstance(b, dict))
        if sys_text:
            oai_messages.append({"role": "system", "content": sys_text})
    for m in request.messages:
        text = m.content if isinstance(m.content, str) else \
            "".join(b.get("text", "") for b in m.content if isinstance(b, dict) and b.get("type") == "text")
        oai_messages.append({"role": m.role, "content": text})

    # Chat-Format
    text = tok.apply_chat_template(oai_messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")

    do_sample = request.temperature > 0.0
    # CUDA-Graph-Pfad: gleiche Logik wie /v1/chat/completions
    runner = None
    if model_id != "gemma3-270m-px-baseline":
        runner = runner_cache.get_or_create(
            model_id=model_id,
            model=model,
            tokenizer=tok,
            input_ids=ids["input_ids"],
            attention_mask=ids.get("attention_mask"),
            max_new_tokens=request.max_tokens,
        )
    if runner is not None:
        with torch.inference_mode():
            tokens = runner_cache.generate(runner, request.max_tokens)
        new_tokens = tokens
        gen_text = tok.decode(new_tokens[0], skip_special_tokens=True)
        return AnthropicResponse(
            model=model_id,
            content=[AnthropicContentBlock(type="text", text=gen_text)],
            usage={"input_tokens": ids["input_ids"].shape[1], "output_tokens": new_tokens.shape[1]},
        )
    # Fallback
    with torch.inference_mode():
        out = model.generate(
            **ids,
            max_new_tokens=request.max_tokens,
            do_sample=do_sample,
            temperature=request.temperature if do_sample else 1.0,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, ids["input_ids"].shape[1]:]
    gen_text = tok.decode(new_tokens, skip_special_tokens=True)

    n_prompt = ids["input_ids"].shape[1]
    return AnthropicResponse(
        model=model_id,
        content=[AnthropicContentBlock(type="text", text=gen_text)],
        usage={"input_tokens": n_prompt, "output_tokens": new_tokens.shape[0]},
    )


# ── Main ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
