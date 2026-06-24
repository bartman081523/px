"""
generators.py — Token Generation (Sync + SSE Streaming)
========================================================
Uses HuggingFace TextIteratorStreamer for streaming.
OpenAI-compatible SSE format: data: {json}\n\n, data: [DONE]\n\n
"""

import torch
import json
import time
from fastapi import HTTPException
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnEOT(StoppingCriteria):
    """Hard-stop when chat delimiters are generated. (SR-61b)"""
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] == 0: return False
        # Handle batching if needed, though we serve single-user
        last_id = input_ids[0, -1].item()
        return last_id in self.stop_ids
import uuid
from typing import Generator, Optional, List, Union

from schemas import (
    ChatMessage, ChatCompletionChunk, StreamChoice, StreamDelta, Role
)


def _has_image_content(messages):
    """True if any message has list-content with image/image_url type items.
    Used to choose between the multimodal processor route and the text-only
    tokenizer route. text-only models with image-bearing requests get 400."""
    return any(
        isinstance(m.get("content"), list) and
        any(isinstance(c, dict) and c.get("type") in ("image", "image_url")
            for c in m["content"])
        for m in messages
    )


def _extract_images(messages):
    """Pull PIL.Image objects out of message content lists (base64-decoded
    from data: URLs) AND replace each image block in the cleaned message
    with an OpenAI-style `{"type": "image"}` (no URL) so the chat template
    still counts the image and substitutes <image_soft_token> placeholders.

    HTTP(S) URLs are skipped silently (out of scope this iteration) and the
    image block is dropped from the cleaned content (model sees text only).

    Returns (cleaned_messages, images_list). Order-preserving."""
    from PIL import Image
    import io, base64
    images = []
    cleaned = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_items = []
            text_parts = []
            for c in content:
                if isinstance(c, dict):
                    ctype = c.get("type")
                    if ctype in ("image", "image_url"):
                        url = (c.get("image_url") or {}).get("url", "") if ctype == "image_url" else c.get("url", "")
                        if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                            b64 = url.split(";base64,", 1)[1]
                            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                            images.append(img)
                            # Keep a stub image block so the chat template
                            # inserts the right number of <image_soft_token>
                            # placeholders (one block → one image token run).
                            new_items.append({"type": "image"})
                        # else: http(s) URL → skip block entirely
                    elif ctype == "text":
                        text_parts.append(c.get("text", ""))
                        new_items.append(c)
                    else:
                        new_items.append(c)
                elif isinstance(c, str):
                    text_parts.append(c)
            # Build final content: if all parts were images with no text,
            # produce a string "" so the role has a content field. Otherwise
            # use a list of {type:text} blocks plus the kept image stubs.
            if text_parts:
                final_content = [{"type": "text", "text": "\n".join(text_parts)}]
                # Append image stubs at the end (gemma3 chat-template inserts
                # <image_soft_token> per "image" dict regardless of position).
                for it in new_items:
                    if isinstance(it, dict) and it.get("type") == "image":
                        final_content.append(it)
            else:
                final_content = new_items if new_items else ""
            cleaned.append({"role": m.get("role", "user"), "content": final_content})
        else:
            cleaned.append(m)
    return cleaned, images


def _stringify_content(content):
    """Ensure content is a string for text-only templates."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item and "files" not in item:
                    parts.append(item["text"])
        return "\n".join(parts)
    if isinstance(content, dict):
        return content.get("text", str(content))
    return str(content)


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
    
    # SR-61b: Hard-stop criteria for chat delimiters (e.g. <end_of_turn>)
    eos_ids = base.get("eos_token_id", [])
    if isinstance(eos_ids, int): 
        eos_ids = [eos_ids]
    if eos_ids:
        clean_ids = list(set([int(eid) for eid in eos_ids if eid is not None]))
        if clean_ids:
            base["stopping_criteria"] = StoppingCriteriaList([StopOnEOT(clean_ids)])

    # Plan 3 Phase B/C: Auto-use_cache=False für lange Inputs auf 4b/E2B.
    # Begründung: mit PX-Patch (full-attention über alle Tokens) ist der
    # KV-Cache buildup bei T>4500 nicht in 12GB haltbar. Plan 3 Phase D:
    # wir setzen einen Marker `_px_use_chunked_prefill` statt use_cache=False.
    # Der Aufrufer (chat_completion / stream_chat_completion) erkennt den
    # Marker und ruft chunked_generate aus scratches/4b-image/chunked_prefill
    # statt model.generate. chunked_generate nutzt full-only cache +
    # chunked attention = 6.4 GB peak bei T=8002 (vs OOM bei full generate
    # und vs 196s bei use_cache=False).
    #
    # NICHT angewendet wenn:
    # - User hat use_cache explizit gesetzt (base["use_cache"] ist nicht None)
    # - Model ist 1b/270m (passt locker in 12GB)
    # - Input ist klein (< T_THRESHOLD)
    if base.get("use_cache") is None:
        input_len = base.get("_input_len", 0)
        is_small_model = _is_small_model(model)
        if not is_small_model and input_len > _LONG_INPUT_THRESHOLD:
            base["_px_use_chunked_prefill"] = True
            import sys as _sys
            print(f"[generate] auto chunked_prefill (input_len={input_len} > "
                  f"{_LONG_INPUT_THRESHOLD}; use_cache=False würde langsam, "
                  f"chunked_prefill ist 6x schneller)", file=_sys.stderr)

    base.pop("_input_len", None)  # cleanup internal marker

    return base


# Threshold: bei 4b + int8 funktioniert generate mit use_cache=True bis ~4500.
# Darüber zwingt die attention-matrix [T,T] OOM. use_cache=False löst das
# auf Kosten der Geschwindigkeit.
_LONG_INPUT_THRESHOLD = 4500


def _is_small_model(model) -> bool:
    """True wenn Model klein genug ist dass long-context use_cache=True OK ist.

    Heuristik: Anzahl Layer × hidden_size ist Proxy für Parameter-Count.
    270m (12 L × 640 H) und 1b (26 L × 1152 H) sind "klein" (low VRAM pressure).
    4b (34 L × 2560 H) und größer brauchen use_cache=False bei long inputs.
    """
    try:
        n_layers = 0
        hidden = 0
        for _, mod in model.named_modules():
            if hasattr(mod, "layers") and hasattr(mod, "rotary_emb"):
                n_layers = len(mod.layers)
                # hidden size via erstes embed
                if hasattr(mod, "embed_tokens"):
                    hidden = mod.embed_tokens.embedding_dim
                break
        return (n_layers * hidden) < 30_000  # 270m=7680, 1b=29952, 4b=87040
    except Exception:
        return False


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
    """Robust injection of chat-specific end tokens.
    Handles <end_of_turn> (106) and <end_of_thought> (107) for Gemma IT.
    """
    eot_tokens = ["<end_of_turn>", "<end_of_thought>", "<eos>", "</s>"]
    eot_ids = []
    for t in eot_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is not None and tid != tokenizer.unk_token_id:
            eot_ids.append(tid)
    
    # Also add standard EOS if not present
    if tokenizer.eos_token_id is not None:
        eot_ids.append(tokenizer.eos_token_id)
    
    eot_ids = list(set(eot_ids)) # Unique

    eos_field = base.get("eos_token_id")
    if isinstance(eos_field, int):
        eos_ids = [eos_field]
    elif isinstance(eos_field, list):
        eos_ids = list(eos_field)
    else:
        eos_ids = []
        
    for eid in eot_ids:
        if eid not in eos_ids:
            eos_ids.append(eid)
            
    base["eos_token_id"] = eos_ids
    # Set pad_token_id to first EOS if not set
    if base.get("pad_token_id") is None and eos_ids:
        base["pad_token_id"] = eos_ids[0]
        
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
    processor = model_entry.get("processor")

    has_images = _has_image_content(messages)
    if has_images and processor is None:
        raise HTTPException(
            status_code=400,
            detail="Model does not support images (text-only model). Use a multimodal model (gemma3-4b-it) or remove the image."
        )

    if has_images:
        # Gemma3 multimodal pattern (Transformers >=4.50):
        #   1) apply_chat_template with tokenize=False → text with <image> placeholders
        #   2) processor(text=..., images=...) → BatchFeature with input_ids + pixel_values
        # apply_chat_template(..., tokenize=True, images=...) errors on Gemma3Processor
        # because it forwards `images=...` internally and the kwarg collides.
        cleaned, images = _extract_images(messages)
        prompt_text = processor.apply_chat_template(
            cleaned, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=prompt_text, images=images, return_tensors="pt").to(model.device)
    else:
        # Text-only path (unchanged from prior behavior).
        processed_messages = [{"role": m.get("role", "user"), "content": _stringify_content(m.get("content", ""))} for m in messages]
        input_text = tokenizer.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=True
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
    # Plan 3: _input_len signals _px_gen_kwargs whether to auto-disable
    # use_cache for long inputs (4b/E2B with T > 4500)
    gen_kwargs["_input_len"] = input_len
    gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
    gen_kwargs = _inject_eot_eos(gen_kwargs, tokenizer)

    with torch.no_grad():
        if gen_kwargs.pop("_px_use_chunked_prefill", False):
            # Plan 3 Phase D: chunked_generate für lange Inputs (4b + T>4500).
            # Spart VRAM (chunked attention + full-only cache) und ist
            # signifikant schneller als use_cache=False. Greift nur, wenn
            # _px_gen_kwargs den Marker gesetzt hat (langer Input, kein
            # user-override, nicht-small-model).
            try:
                from chunked_prefill import chunked_generate
            except ImportError:
                # Fallback: scratches/4b-image nicht im path. Versuche es
                # explizit zu laden.
                import os as _os, sys as _sys
                _SCRATCHES = _os.path.join(
                    _os.path.dirname(_os.path.abspath(__file__)),
                    "scratches", "4b-image",
                )
                if _SCRATCHES not in _sys.path:
                    _sys.path.insert(0, _SCRATCHES)
                from chunked_prefill import chunked_generate
            eos_id = tokenizer.eos_token_id
            outputs = chunked_generate(
                model,
                inputs["input_ids"],
                max_new_tokens=gen_kwargs.get("max_new_tokens", 64),
                do_sample=gen_kwargs.get("do_sample", False),
                eos_token_id=eos_id,
            )
        else:
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
    processor = model_entry.get("processor")

    has_images = _has_image_content(messages)
    if has_images and processor is None:
        # Cannot raise HTTPException from inside a sync generator after the SSE
        # response has started. Yield a single error chunk then [DONE] so the
        # bridge surfaces a readable message instead of dying silently.
        err = {"error": {"message": "Model does not support images (text-only model). Use a multimodal model (gemma3-4b-it) or remove the image.", "type": "invalid_request_error"}}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
        return

    if has_images:
        cleaned, images = _extract_images(messages)
        prompt_text = processor.apply_chat_template(
            cleaned, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=prompt_text, images=images, return_tensors="pt").to(model.device)
    else:
        processed_messages = [{"role": m.get("role", "user"), "content": _stringify_content(m.get("content", ""))} for m in messages]
        input_text = tokenizer.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

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
    # Plan 3: _input_len signals _px_gen_kwargs whether to auto-disable
    # use_cache for long inputs (4b/E2B with T > 4500)
    gen_kwargs["_input_len"] = input_len
    gen_kwargs = _px_gen_kwargs(model, gen_kwargs)
    gen_kwargs = _inject_eot_eos(gen_kwargs, tokenizer)

    # Plan 3 Phase D: bei langem Input + 4b/E2B nutzen wir chunked_generate
    # mit Streamer statt model.generate (10x schneller als use_cache=False).
    use_chunked_stream = gen_kwargs.pop("_px_use_chunked_prefill", False)
    if use_chunked_stream:
        # Inline-Import: scratches/4b-image ist nicht im Standard-Pfad.
        try:
            from chunked_prefill import chunked_generate as _chunked_generate
        except ImportError:
            import os as _os, sys as _sys
            _SCRATCHES = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)),
                "scratches", "4b-image",
            )
            if _SCRATCHES not in _sys.path:
                _sys.path.insert(0, _SCRATCHES)
            from chunked_prefill import chunked_generate as _chunked_generate

        eos_field = gen_kwargs.pop("eos_token_id", None)
        # _inject_eot_eos setzt eos_token_id auf eine Liste — wir brauchen
        # einen einzelnen int für chunked_generate.
        if isinstance(eos_field, list):
            eos_id = eos_field[0] if eos_field else None
        elif isinstance(eos_field, int):
            eos_id = eos_field
        else:
            eos_id = tokenizer.eos_token_id
        do_sample = gen_kwargs.get("do_sample", False)

        def _chunked_worker():
            try:
                _chunked_generate(
                    model,
                    inputs["input_ids"],
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    eos_token_id=eos_id,
                    streamer=streamer,
                )
            except Exception as _e:
                import traceback as _tb
                _tb.print_exc()
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = Thread(target=_chunked_worker)
        thread.start()
    else:
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
    # Plan 3: _input_len signals _px_gen_kwargs whether to auto-disable
    # use_cache for long inputs (4b/E2B with T > 4500)
    gen_kwargs["_input_len"] = input_len
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
    # Plan 3: _input_len signals _px_gen_kwargs whether to auto-disable
    # use_cache for long inputs (4b/E2B with T > 4500)
    gen_kwargs["_input_len"] = input_len
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