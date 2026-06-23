"""
schemas.py — Pydantic Request/Response Models (OpenAI API Format)
=================================================================
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal
from enum import Enum
import time
import uuid


# ── Enums ──

class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


# ── PX Preset (post 2026-06-11) ──
# States: BASELINE (nackt), ACTIVE_MANIFOLD (full PX), ACTIVE_MANIFOLD_LEAN
# (kausaler Kern ohne die vier Crutches + AZS-Awareness-Injektion; validiert via
# scratches/consolidation), ACTIVE_MANIFOLD_RELAY (LEAN + verstärkbar Selbst-
# Injektions-Relay, psychomotrik seite15: Re-Injektion der modell-eigenen L16-
# Zustands-Richtung d_width am post-recur Layer; Motor unangetastet, forward_hook).
# Old presets (SUBJECTIVE, RIGOR, RESONANCE_CITY, DMT-FULL, UNCENSORED) migrate
# to ACTIVE_MANIFOLD at load time.
PXConfigPreset = Literal["BASELINE", "ACTIVE_MANIFOLD", "ACTIVE_MANIFOLD_LEAN", "ACTIVE_MANIFOLD_RELAY"]


# ── Request Models ──

class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    # PX extensions
    px_subjective: Optional[bool] = False
    px_gamma: Optional[float] = None           # Override gamma for LTI/ADC injection
    px_routing_mode: Optional[str] = None      # "adaptive" or "fixed"
    px_config_preset: Optional[PXConfigPreset] = None  # "BASELINE" | "ACTIVE_MANIFOLD" | ..._LEAN | ..._RELAY
    # verstärkbar Relay (psychomotrik seite15): Re-Injektion der modell-eigenen
    # L16-Zustands-Richtung d_width am post-recur Layer. sign=+1 → WIDE/expansiv,
    # −1 → NARROW/eng, 0 → relay off (default bei nicht-RELAY-Presets). alpha =
    # Bruchteil der L21-last-pos-Norm (seite15-validiert: 0.5). layer default 21.
    px_relay_sign: Optional[int] = None        # -1 | 0 | +1
    px_relay_alpha: Optional[float] = None     # fraction of last-pos norm (0.0–1.5)
    px_relay_layer: Optional[int] = None       # post-recur injection layer (default 21)


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    # PX extensions
    px_subjective: Optional[bool] = False
    px_gamma: Optional[float] = None
    px_routing_mode: Optional[str] = None
    px_config_preset: Optional[str] = None
    px_relay_sign: Optional[int] = None
    px_relay_alpha: Optional[float] = None
    px_relay_layer: Optional[int] = None


# ── Response Models ──

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: Usage


class StreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int
    delta: StreamDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:12]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "px-server"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]