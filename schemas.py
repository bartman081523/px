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
# Two states only: BASELINE (nackt) or ACTIVE_MANIFOLD (full PX).
# All old presets (SUBJECTIVE, RIGOR, RESONANCE_CITY, DMT-FULL, UNCENSORED)
# are migrated to ACTIVE_MANIFOLD at load time.
PXConfigPreset = Literal["BASELINE", "ACTIVE_MANIFOLD"]


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
    px_config_preset: Optional[PXConfigPreset] = None  # "BASELINE" | "ACTIVE_MANIFOLD"


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