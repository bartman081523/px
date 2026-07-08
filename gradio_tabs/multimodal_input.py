"""Pure adapter: normalize a Gradio multimodal input widget value into the
content format that chat_fn (gradio_tabs/chat_tab.py) already accepts.

No gradio import is required for the core logic; the module is pure Python
and depends only on the shape of the widget's output value.

Supported input shapes (handle ALL):
  - None or "" -> "" (plain string)
  - str -> str (plain text, no files)
  - dict {"text": str, "files": [paths]} ->
      * if no files: return the text string (or "" if text empty)
      * if files: return [{"type":"text","text":text}] +
        one block per file (see _file_block below)
  - dict missing "text" or "files" keys -> treat missing as "" / []

File classification (per attached file):
  - image extensions (png/jpg/jpeg/gif/bmp/webp) -> {"type":"image","image":path}
    (path stored, NOT read — the vision processor loads it later)
  - text extensions (txt/md/py/json/csv/log/xml/html/yml/yaml/tsv/js/ts/sh/
    ini/cfg/toml/rst/tex) -> inline the file contents as a text block
    (UTF-8, errors replaced, capped at MAX_TEXT_FILE_BYTES)
  - unknown extensions -> a text note "[attached file: <name> — not inlined]"
    (so the model is aware an unsupported file was attached)

File paths may be strings, or gradio file-dict objects with a "path" key
(e.g. {"path": "/tmp/x.png", "orig_name": "x.png", "size": 123}).
The path string is extracted robustly.

Plus (wip-tts a7f4643 port): history normalization for the Gradio Chatbot.
  - _convert_openai_block_to_gradio: OpenAI image_url/input_audio → Gradio file
  - _guess_mime_from_data_url: data:<mime>;base64 → <mime>
  - _normalize_history_for_chatbot: persisted history with mixed shapes
    (str / untyped list / Gradio-typed list / OpenAI blocks) → Gradio-safe
"""

import os

# 64 KiB cap per inlined text file — keeps prompts bounded for the 12 GB card.
MAX_TEXT_FILE_BYTES = 64 * 1024

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

TEXT_EXTS = {
    ".txt", ".md", ".py", ".json", ".csv", ".log", ".xml", ".html", ".htm",
    ".yml", ".yaml", ".tsv", ".js", ".ts", ".sh", ".ini", ".cfg", ".toml",
    ".rst", ".tex", ".c", ".cpp", ".h", ".rs", ".go", ".rb", ".sql",
}


def _extract_path(f) -> str:
    """Extract a filesystem path string from a file entry.

    A file entry may be a plain string path, or a dict-like object carrying a
    "path" key (as emitted by gradio.MultimodalTextbox / gr.File).
    """
    if isinstance(f, dict):
        p = f.get("path")
        if p is not None:
            return p
        # Some gradio file dicts use "url" as fallback; keep path as canonical.
        return f.get("url", "")
    return str(f)


def _extract_name(f) -> str:
    """Best-effort display name for a file entry (basename)."""
    if isinstance(f, dict):
        for k in ("orig_name", "name"):
            v = f.get(k)
            if v:
                return os.path.basename(str(v))
        p = f.get("path") or f.get("url") or ""
        return os.path.basename(str(p)) if p else "file"
    return os.path.basename(str(f))


def _ext_of(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def _load_text_file(path: str, cap: int = MAX_TEXT_FILE_BYTES) -> str:
    """Read up to `cap` bytes of a text file as UTF-8 (errors replaced).
    Returns the (possibly truncated) contents, or an error note."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(cap + 1)
        truncated = len(raw) > cap
        if truncated:
            raw = raw[:cap]
        text = raw.decode("utf-8", errors="replace")
        if truncated:
            text += "\n…[truncated]"
        return text
    except OSError as e:
        return f"[could not read {path}: {e}]"


def _file_block(f):
    """Build the content block for one attached file entry."""
    path = _extract_path(f)
    name = _extract_name(f)
    ext = _ext_of(path)
    if ext in IMAGE_EXTS:
        return {"type": "image", "image": path}
    if ext in TEXT_EXTS:
        label = ext.lstrip(".") or "txt"
        body = _load_text_file(path)
        return {"type": "text",
                "text": f"```{label} {name}\n{body}\n```"}
    return {"type": "text",
            "text": f"[attached file: {name} — unsupported type, not inlined]"}


def normalize_multimodal_message(value):
    """Normalize a multimodal input widget value into chat_fn's expected
    message content (plain str, or content-list with text + file blocks)."""
    # None -> empty
    if value is None:
        return ""

    # Plain string passthrough
    if isinstance(value, str):
        return value

    # Dict shape: {"text": ..., "files": [...]}
    if isinstance(value, dict):
        text = value.get("text", "") or ""
        files = value.get("files", []) or []

        if not files:
            return text

        content = [{"type": "text", "text": text}]
        for f in files:
            content.append(_file_block(f))
        return content

    # Unknown shape: coerce to string
    return str(value)


def is_empty_message(value) -> bool:
    """True if the normalized message would be empty (no text and no files)."""
    normalized = normalize_multimodal_message(value)
    if isinstance(normalized, str):
        return normalized == ""
    # content-list: empty only if it has no text and no image blocks
    return len(normalized) == 0


def extract_text_blocks(content) -> str:
    """Robust string-extraction from Gradio chat content.

    Handles all the shapes Gradio can hand us for a single message:
      - None -> ""
      - str -> unchanged
      - list of {"type": "text", "text": ...} blocks -> concatenated text
      - dict with "text" key -> str(dict["text"])
      - anything else -> str(content) (last-resort coercion)

    Used by chat_tab.chat_fn for the system-prompt preview, and by
    handle_load_saved to coerce persisted Multimodal-Listen back into
    strings for sys-preview widgets.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", "") or "")
                elif "text" in block:
                    parts.append(str(block["text"]) if block["text"] else "")
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", "") or "")
    return str(content)


# ── History normalization (ported from wip-tts a7f4643) ─────────────────────
# Persistierte Sessions können Multimodal-Blöcke in zwei problematischen
# Formen enthalten, die Gradio 6.15.2 Chatbot._postprocess_content ablehnt:
#   1. OpenAI image_url / input_audio → Gradio kennt nur "type":"file" + FileData
#   2. Untyped dicts ({"text": "..."}) ohne "type" → ValueError in _check_format
# Diese drei Helper normalisieren die History in ein Gradio-sicheres Format.


def _convert_openai_block_to_gradio(block):
    """Konvertiert einen einzelnen Multimodal-Block ins Gradio-Format.

    Akzeptiert:
      - text → passthrough
      - file / component → passthrough (bereits Gradio-Format)
      - image_url (OpenAI) → Gradio file-Block mit FileData
        (Gradio 6.x Chatbot kennt nur `{"type": "file", "file": {...}}`,
        `{"type": "image"}` ist KEIN gültiges Format → ValueError)
      - input_audio (OpenAI) → Gradio file-Block (analog)

    Die url/der Pfad wird in das `path`-Feld gesetzt (Gradio akzeptiert
    data:-URLs, http(s)-URLs und lokale Pfade dort).

    Returns None wenn der Block nicht konvertierbar ist (Caller droppt).
    """
    if not isinstance(block, dict):
        return None
    btype = block.get("type")
    if btype == "text":
        return block
    if btype in ("file", "component"):
        return block
    if btype == "image_url":
        # OpenAI: {"type": "image_url", "image_url": {"url": "data:..."}}.
        # → Gradio: {"type": "file", "file": {"path": url, "mime_type": "image/..."}}.
        url = (block.get("image_url") or {}).get("url")
        if not url:
            return None
        mime = _guess_mime_from_data_url(url, default="image/png")
        return {
            "type": "file",
            "file": {"path": url, "mime_type": mime},
        }
    if btype == "input_audio":
        url = (block.get("input_audio") or {}).get("url") or (
            block.get("input_audio") or {}
        ).get("data")
        if not url:
            return None
        mime = _guess_mime_from_data_url(url, default="audio/wav")
        return {
            "type": "file",
            "file": {"path": url, "mime_type": mime},
        }
    # Unbekannt: durchreichen. Caller filtert ggf. mit einer allow-list.
    return block


def _guess_mime_from_data_url(url: str, default: str) -> str:
    """Extrahiert den MIME-Type aus einem data:-URL, sonst default."""
    if url.startswith("data:"):
        # Format: data:<mime>;base64,<data>
        head = url[5:].split(";", 1)[0]
        if head:
            return head
    return default


def _normalize_history_for_chatbot(history):
    """Normalize a persisted history list so it survives Gradio's
    ``Chatbot._check_format`` + ``_postprocess_content``.

    Two failure modes observed when loading legacy sessions:

    1. ``content`` is a list of dicts WITHOUT the ``"type"`` key
       (e.g. ``[{"text": "[SYSTEM CONTEXT]\\n..."}]``). Gradio's
       ``_postprocess_content`` rejects these in its ``else: raise
       ValueError`` branch — the resulting error surfaces as
       ``"Data incompatible with messages format"`` even though the
       outer ``role``+``content`` keys are present.

    2. ``content`` is a plain ``str`` with embedded newlines or other
       control chars — usually fine, but we strip null bytes defensively.

    The function:
      - Drops None / non-dict entries.
      - Coerces ``content=list`` whose blocks are missing ``type`` into
        a single concatenated string (via ``extract_text_blocks``).
      - Preserves valid multimodal lists (``type=="text"|"file"|...``).
      - Converts OpenAI image_url/input_audio blocks via
        ``_convert_openai_block_to_gradio`` before storing.
      - Returns a NEW list — does not mutate the input.
    """
    if not history:
        return []
    out = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "user", "assistant"):
            continue
        if content is None:
            continue
        if isinstance(content, list):
            # Inspect each block: if any block has a recognised ``type``,
            # keep the list. Otherwise collapse to a single string.
            has_typed_block = any(
                isinstance(b, dict) and b.get("type") in (
                    "text", "file", "component",
                    # OpenAI-Multimodal-Blöcke werden in _convert_openai_block_to_gradio
                    # ins Gradio-Format übersetzt. Sie zählen hier als
                    # 'typed', damit sie nicht zu (leerem) String geflatted werden.
                    "image_url", "input_audio",
                )
                for b in content
            )
            if has_typed_block:
                # OpenAI-Multimodal-Blöcke (image_url / input_audio) in
                # Gradio-Format konvertieren — Gradio-Chatbot lehnt das
                # OpenAI-Schema in _postprocess_content mit ValueError ab.
                converted = [_convert_openai_block_to_gradio(b) for b in content]
                # Defensive: nur behalten wenn nach Konversion noch was da ist.
                converted = [b for b in converted if b is not None]
                if not converted:
                    continue
                out.append({"role": role, "content": converted})
            else:
                # Either no type keys, or unknown shapes — flatten.
                text = extract_text_blocks(content)
                if text:
                    out.append({"role": role, "content": text})
                # else: drop empty content.
        elif isinstance(content, str):
            # Strip null bytes; otherwise pass through.
            cleaned = content.replace("\x00", "")
            if cleaned:
                out.append({"role": role, "content": cleaned})
        elif isinstance(content, dict):
            # Single dict — coerce to its ``text`` field if present.
            text = content.get("text") or extract_text_blocks(content)
            if text:
                out.append({"role": role, "content": str(text)})
        else:
            # Fallback: stringify.
            out.append({"role": role, "content": str(content)})
    return out