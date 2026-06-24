"""
server.py — OpenAI-Compatible API Server for PX Models
======================================================
Serves gemma3-270m-px and minicpm5-1b-px through the OpenAI API format.
Compatible with LM Studio, Open WebUI, and any OpenAI API client.

Endpoints:
  GET  /v1/models              — List available models
  GET  /v1/models/{id}         — Model info
  POST /v1/chat/completions     — Chat (streaming + non-streaming)
  POST /v1/completions          — Text completion
  GET  /v1/px/metrics/{id}     — PX cognitive metrics (extension)
  GET  /                       — Health check
  GET  /v1/px/telemetry        — Server telemetry summary
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import MODEL_REGISTRY, SERVER_CONFIG
from schemas import (
    ChatCompletionRequest, CompletionRequest,
    ChatCompletionResponse, ChatChoice, ChatMessage, Usage, Role,
    CompletionResponse, CompletionChoice,
    ModelInfo, ModelListResponse,
)
from model_manager import ModelManager
from generators import (
    generate_chat_completion, generate_chat_completion_stream,
    generate_completion, generate_completion_stream,
)
from telemetry import telemetry


# ── App Setup ──

manager = ModelManager()


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan handler."""
    print("[PX API] Server starting...")
    yield
    # Shutdown: unload all models
    await manager.shutdown()
    print("[PX API] Server stopped.")


app = FastAPI(
    title="PX API Server",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for LM Studio, Open WebUI, browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health / Info ──

@app.get("/")
async def root():
    """Health check."""
    loaded = [mid for mid in manager.list_models() if manager.is_loaded(mid)]
    return {
        "status": "ok",
        "models_available": manager.list_models(),
        "models_loaded": loaded,
    }


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """OpenAI-compatible model listing."""
    models = []
    for mid in manager.list_models():
        models.append(ModelInfo(id=mid, owned_by="px-server"))
    return ModelListResponse(data=models)


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get info about a specific model."""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model {model_id} not found")
    registry = MODEL_REGISTRY[model_id]
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "px-server",
        "px_loaded": manager.is_loaded(model_id),
        "px_subjective": manager._models.get(model_id, {}).get("px_subjective", False),
        "hf_id": registry["hf_id"],
        "model_type": registry["model_type"],
    }


# ── Chat Completions ──

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint."""
    model_id = request.model
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model {model_id} not found")

    try:
        model_entry = await manager.get_model(
            model_id,
            px_subjective=request.px_subjective,
            px_gamma=request.px_gamma,
            px_routing_mode=request.px_routing_mode,
            px_config_preset=request.px_config_preset,
            px_relay_sign=request.px_relay_sign,
            px_relay_alpha=request.px_relay_alpha,
            px_relay_layer=request.px_relay_layer,
            quantization=request.quantization,
        )
    except Exception as e:
        raise HTTPException(503, f"Failed to load model: {e}")

    messages = [{"role": m.role.value, "content": m.content} for m in request.messages]

    if request.stream:
        return StreamingResponse(
            generate_chat_completion_stream(
                model_entry=model_entry,
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                model_id=model_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    try:
        result = await generate_chat_completion(
            model_entry=model_entry,
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    # Collect PX metrics
    px_metrics = manager.get_px_metrics(model_id)
    prompt_text = messages[-1]["content"] if messages else ""
    telemetry.record(
        model_id, 
        result["prompt_tokens"], 
        result["completion_tokens"], 
        px_metrics,
        prompt_text=prompt_text,
        completion_text=result["text"]
    )

    return ChatCompletionResponse(
        model=model_id,
        choices=[ChatChoice(
            index=0,
            message=ChatMessage(role=Role.assistant, content=result["text"]),
            finish_reason="stop",
        )],
        usage=Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
        ),
    )


# ── Text Completions ──

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible text completion endpoint."""
    model_id = request.model
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model {model_id} not found")

    try:
        model_entry = await manager.get_model(
            model_id,
            px_subjective=request.px_subjective,
            px_gamma=request.px_gamma,
            px_routing_mode=request.px_routing_mode,
            px_config_preset=request.px_config_preset,
            px_relay_sign=request.px_relay_sign,
            px_relay_alpha=request.px_relay_alpha,
            px_relay_layer=request.px_relay_layer,
            quantization=request.quantization,
        )
    except Exception as e:
        raise HTTPException(503, f"Failed to load model: {e}")

    if request.stream:
        return StreamingResponse(
            generate_completion_stream(
                model_entry=model_entry,
                prompt=request.prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stop=request.stop,
                model_id=model_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await generate_completion(
            model_entry=model_entry,
            prompt=request.prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    px_metrics = manager.get_px_metrics(model_id)
    telemetry.record(
        model_id, 
        result["prompt_tokens"], 
        result["completion_tokens"], 
        px_metrics,
        prompt_text=request.prompt,
        completion_text=result["text"]
    )

    return CompletionResponse(
        model=model_id,
        choices=[CompletionChoice(
            text=result["text"],
            index=0,
        )],
        usage=Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
        ),
    )


# ── PX Metrics (Extension) ──

@app.get("/v1/px/metrics/{model_id}")
async def px_metrics(model_id: str):
    """Get PX cognitive metrics for a model."""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(404, f"Model {model_id} not found")
    if not manager.is_loaded(model_id):
        raise HTTPException(404, f"Model {model_id} not loaded yet. Send a request first.")
    return manager.get_px_metrics(model_id)


@app.get("/v1/px/telemetry")
async def px_telemetry():
    """Get server telemetry summary."""
    return telemetry.get_summary()


# ── Main ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        log_level="info",
    )