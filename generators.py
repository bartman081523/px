"""
generators.py — Token Generation (Sync + SSE Streaming)
========================================================
Uses HuggingFace TextIteratorStreamer for streaming.
OpenAI-compatible SSE format: data: {json}\n\n, data: [DONE]\n\n
"""

import torch
import json
import time
import uuid
from typing import Generator, Optional, List, Union

from schemas import (
    ChatMessage, ChatCompletionChunk, StreamChoice, StreamDelta, Role
)


def _px_gen_kwargs(model, base: dict) -> dict:
    """Inject PX-specific kwargs (e.g. repetition_penalty, no_repeat_ngram_size)
    onto a generation kwargs dict. The patched model exposes
    `_px_repetition_penalty` as a model-level attribute when SR-59 /
    token-loop mitigations set a value > 1.0. We pass it through to
    `model.generate(...)` so the mitigation actually takes effect —
    otherwise the dict lives on `_px_config` and is never read.

    The attrs live on the text_model (e.g. model.model.language_model or
    model.model), not the top-level wrapper — we walk named_modules to
    find whichever sub-model has `_px_repetition_penalty` set.
    """
    base = dict(base)
    rp = _find_px_attr(model, "_px_repetition_penalty", default=1.0) or 1.0
    # Guard: only inject if the value is a real number > 1.0
    if isinstance(rp, (int, float)) and rp > 1.0:
        base["repetition_penalty"] = rp
    # For Gemma 4 in particular, long generations (≥200 tokens) drift
    # into a 4-token attractor loop even with rp=1.15. The
    # no_repeat_ngram_size=3 n-gram constraint catches the loop without
    # the brittleness of raising rp further (which destroys natural
    # repetition in German compounds). It's a no-op for short outputs.
    ngram = _find_px_attr(model, "_px_no_repeat_ngram_size", default=0) or 0
    if isinstance(ngram, (int, float)) and ngram:
        base["no_repeat_ngram_size"] = int(ngram)
    return base


def _find_px_attr(model, attr: str, default=None):
    """Walk the module tree to find the first submodule carrying `attr`."""
    val = getattr(model, attr, None)
    if val is not None:
        return val
    for _, mod in model.named_modules():
        val = getattr(mod, attr, None)
        if val is not None:
            return val
    return default


def _inject_eot_eos(base: dict, tokenizer) -> dict:
    """IT chat templates end with <end_of_turn> (token 106 for Gemma IT),
    not <eos> (token 1). Without this, the model emits the natural
    chat-end token and HF.generate keeps running until max_new_tokens
    is hit or the model falls into a degenerate attractor. We accept
    eos_token_id as either int or list and append the chat-template
    end-of-turn token if it is distinct from tokenizer.eos_token_id.
    """
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        return base
    if eot_id == tokenizer.eos_token_id:
        return base
    eos_field = base.get("eos_token_id")
    eos_ids: list
    if isinstance(eos_field, int):
        eos_ids = [eos_field]
    elif isinstance(eos_field, list):
        eos_ids = list(eos_field)
    elif eos_field is None:
        eos_ids = []
    else:
        return base
    if eot_id not in eos_ids:
        eos_ids.append(eot_id)
    base["eos_token_id"] = eos_ids
    return base


def _make_chunk(
    completion_id: str, created: int, model_id: str,
    index: int, delta: dict, finish_reason: Optional[str] = None
) -> ChatCompletionChunk:
    """Create a streaming chunk in OpenAI format."""
    return ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model_id,
        choices=[StreamChoice(
            index=index,
            delta=StreamDelta(**delta),
            finish_reason=finish_reason,
        )],
    )


async def generate_chat_completion(
    model_entry: dict,
    messages: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Optional[Union[str, List[str]]] = None,
) -> dict:
    """Non-streaming chat completion. Returns text + token counts."""
    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]

    # Build prompt using chat template
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # Generate
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 1e-10,
        top_p=top_p,
        do_sample=temperature > 0,
    )
    if stop:
        stop_list = stop if isinstance(stop, list) else [stop]
        gen_kwargs["stop_strings"] = stop_list
        gen_kwargs["tokenizer"] = tokenizer
    gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
    gen_kwargs = _inject_eot_eos(gen_kwargs, tokenizer)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only new tokens
    new_tokens = outputs[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Trim at stop strings (safety net)
    if stop:
        stop_list = stop if isinstance(stop, list) else [stop]
        for s in stop_list:
            idx = text.find(s)
            if idx >= 0:
                text = text[:idx]

    completion_tokens = len(new_tokens)

    return {
        "text": text,
        "prompt_tokens": input_len,
        "completion_tokens": completion_tokens,
    }


async def generate_chat_completion_stream(
    model_entry: dict,
    messages: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Optional[Union[str, List[str]]] = None,
    model_id: str = "",
) -> Generator[str, None, None]:
    """Streaming SSE generator for chat completions.

    Yields lines in OpenAI SSE format:
      data: {json}\n\n
      ...
      data: [DONE]\n\n
    """
    from transformers import TextIteratorStreamer
    from threading import Thread

    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Setup streamer
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Run generation in background thread
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 1e-10,
        top_p=top_p,
        do_sample=temperature > 0,
        streamer=streamer,
    )
    if stop:
        stop_list = stop if isinstance(stop, list) else [stop]
        gen_kwargs["stop_strings"] = stop_list
        gen_kwargs["tokenizer"] = tokenizer
    gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
    gen_kwargs = _inject_eot_eos(gen_kwargs, tokenizer)

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # SSE format
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # First chunk: role
    chunk = _make_chunk(completion_id, created, model_id, 0,
                        delta={"role": "assistant"}, finish_reason=None)
    yield f"data: {chunk.model_dump_json()}\n\n"

    # Stream tokens
    for text in streamer:
        if text:
            chunk = _make_chunk(completion_id, created, model_id, 0,
                                delta={"content": text}, finish_reason=None)
            yield f"data: {chunk.model_dump_json()}\n\n"

    thread.join()

    # Final chunk: finish_reason
    chunk = _make_chunk(completion_id, created, model_id, 0,
                        delta={}, finish_reason="stop")
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def generate_completion(
    model_entry: dict,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Optional[Union[str, List[str]]] = None,
) -> dict:
    """Non-streaming text completion. Returns text + token counts."""
    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 1e-10,
        top_p=top_p,
        do_sample=temperature > 0,
    )
    if stop:
        stop_list = stop if isinstance(stop, list) else [stop]
        gen_kwargs["stop_strings"] = stop_list
        gen_kwargs["tokenizer"] = tokenizer
    gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
    gen_kwargs = _inject_eot_eos(gen_kwargs, tokenizer)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    if stop:
        stop_list = stop if isinstance(stop, list) else [stop]
        for s in stop_list:
            idx = text.find(s)
            if idx >= 0:
                text = text[:idx]

    return {
        "text": text,
        "prompt_tokens": input_len,
        "completion_tokens": len(new_tokens),
    }


async def generate_completion_stream(
    model_entry: dict,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Optional[Union[str, List[str]]] = None,
    model_id: str = "",
) -> Generator[str, None, None]:
    """Streaming SSE generator for text completions."""
    from transformers import TextIteratorStreamer
    from threading import Thread

    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 1e-10,
        top_p=top_p,
        do_sample=temperature > 0,
        streamer=streamer,
    )
    if stop:
        stop_list = stop if isinstance(stop, list) else [stop]
        gen_kwargs["stop_strings"] = stop_list
        gen_kwargs["tokenizer"] = tokenizer
    gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
    gen_kwargs = _inject_eot_eos(gen_kwargs, tokenizer)

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    for text in streamer:
        if text:
            chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "text": text,
                    "index": 0,
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    thread.join()

    # Final chunk
    chunk = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "text": "",
            "index": 0,
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"