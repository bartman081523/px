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
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._current_model_id: Optional[str] = None
        self._lock = False

    def is_loaded(self, model_id: str) -> bool:
        return self._current_model_id == model_id and self._model is not None

    def get_model(self, model_id: str):
        if model_id not in MODELS:
            raise HTTPException(404, f"Model {model_id} not in registry")
        if self.is_loaded(model_id):
            return self._tokenizer, self._model
        # Lade neu
        self._unload()
        cfg = MODELS[model_id]
        print(f"[v3.5g-Server] Loading {model_id} (preset={cfg['preset']})...", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL, dtype=torch.bfloat16
        ).to("cuda").eval()
        if cfg["preset"] != "BASELINE":
            apply_px_patch(model.model, config_preset=cfg["preset"])
        self._model = model
        self._current_model_id = model_id
        print(f"[v3.5g-Server] {model_id} loaded.", flush=True)
        return self._tokenizer, self._model

    def _unload(self):
        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()
            self._model = None
            self._tokenizer = None
            self._current_model_id = None

    async def shutdown(self):
        self._unload()


manager = ModelManager()


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
        "loaded": [manager._current_model_id] if manager._current_model_id else [],
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
