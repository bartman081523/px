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


def strip_unsupported_model_kwargs(model, gen_kwargs: dict) -> dict:
    """Entfernt kwargs aus gen_kwargs, die das Model-Forward nicht akzeptiert.

    Hintergrund (Plan 7.2, 2026-06-30): Llama-basierte Modelle (z.B. MiniCPM5-1B)
    kennen `token_type_ids` nicht, aber einige Tokenizer (Gemma-kompatibel?)
    setzen es ungewollt. `model.generate()` validiert model_kwargs VOR dem
    ersten forward und lehnt unbekannte Keys mit ValueError ab.

    Vorher: der Strip war dupliziert in generate() und chunked_generate().
    Chat-Tab-Pfad (gradio_tabs/chat_tab.py:generate_with_lock) hatte KEINEN
    Strip → Live-Crash mit minicpm5-1b + ACTIVE_MANIFOLD_RELAY-Preset
    ("ValueError: The following `model_kwargs` are not used by the model:
    ['token_type_ids']").

    Diese Funktion ist die single source of truth: alle Aufrufer von
    model.generate(**kwargs) müssen sie vorher rufen.

    Implementierung: targeted Liste bekannter "Llama-weiß-nicht"-kwargs
    (`token_type_ids`). Generische Heuristik (alle kwargs prüfen, ob in
    forward.co_varnames) ist zu aggressiv — `streamer`, `max_new_tokens`,
    `do_sample`, `temperature` etc. sind generate()-spezifisch, NICHT
    forward()-spezifisch, würden aber fälschlich gestrippt.

    Erkenntnis: model.generate() selbst akzeptiert diese generate-spezifischen
    kwargs (sie sind Teil von GenerationConfig), nur model.forward nicht.
    transformers' _validate_model_kwargs prüft beide Schichten.
    """
    # Targeted: nur kwargs, von denen wir empirisch wissen, dass sie
    # model.generate() bei Llama-Pfaden crashen.
    KNOWN_LLAMA_UNSUPPORTED = ("token_type_ids",)
    forward = getattr(model, "forward", None)
    if forward is not None and hasattr(forward, "__code__"):
        supported = set(forward.__code__.co_varnames)
        return {
            k: v for k, v in gen_kwargs.items()
            if k not in KNOWN_LLAMA_UNSUPPORTED or k in supported
        }
    # Fallback: forward nicht inspizierbar — generischer Strip auf
    # known-unsupported kwargs.
    return {k: v for k, v in gen_kwargs.items() if k not in KNOWN_LLAMA_UNSUPPORTED}


def _chat_template_kwargs(thinking=None, thinking_effort=None) -> dict:
    """Qwen3.5-Denken-Schalter als Extra-Template-Variablen.

    enable_thinking=False injiziert einen leeren, geschlossenen Think-Block
    ("think open+close leer") in den Generation-Prompt — das Modell antwortet
    direkt, ohne Vorlauf-Monolog. reasoning_effort (xhigh|medium|low) steuert
    die Denkanweisungen bei thinking=an (Template-Default: xhigh). Templates,
    die diese Variablen nicht referenzieren (gemma3 & Co.), ignorieren sie —
    der Jinja-Kontext akzeptiert beliebige Extra-Variablen.
    """
    kw = {}
    if thinking is not None:
        kw["enable_thinking"] = bool(thinking)
    if thinking_effort:
        kw["reasoning_effort"] = thinking_effort
    return kw


# ── Gemma4 Thinking-Budget (Plan 2026-10-05) ────────────────────────────
# max_thinking_tokens als App-Level LogitsProcessor. Upstream transformers
# 5.13.0 kennt den Parameter NICHT (nur der ungemergte PR #42112 — von dem
# der dev.to-Artikel "Gemma 4's Thinking Mode" die Pipe-Syntax zeigt; dort
# ist er illustrativ, kein etablierter generate()-Parameter).
#
# Gemma-4 denkt im "thought channel", abgegrenzt durch ECHTE Tokens des
# lokalen Tokenizers (Probe scratches/gemma4_think_tokens_probe.py):
#   öffnend    "<|channel>"   id=100  (special/added)
#   bestätigend "thought"     id=45518 (plain vocab — folgt dem Open)
#   schließend "<channel|>"   id=101  (special/added)
# ("<|think|>" id=98 triggert den System-Turn; im Generation-Stream sind
# die Kanal-Delimiter open+confirm+close die Semantik. Die Artikel-Demo
# `<thinking>…</thinking>` existiert im Vokabular NICHT — unk-Fallback id 3.)
_THINK_OPEN_TOKEN = "<|channel>"
_THINK_CLOSE_TOKEN = "<channel|>"
_THINK_CONFIRM_TOKEN = "thought"


def _resolve_think_token(tokenizer, text):
    """Text → Token-ID, UNK-Fallback ausgeschlossen (round-trip check).

    Der Gemma4-Tokenizer mappt unbekannte Strings auf unk_token_id (3) —
    ein false positive wäre fatale Maskierung der falschen ID. Round-trip
    convert_ids_to_tokens(id) == text beweist echte Vokabular-Präsenz.
    """
    try:
        tid = tokenizer.convert_tokens_to_ids(text)
    except Exception:
        return None
    if tid is None or int(tid) < 0:
        return None
    tid = int(tid)
    if tid == getattr(tokenizer, "unk_token_id", None):
        return None
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            if tokenizer.convert_ids_to_tokens(tid) != text:
                return None
        except Exception:
            return None
    return tid


def _think_channel_token_ids(tokenizer):
    """(open_id, close_id, confirm_id) oder None (Infrastructure fehlt).

    None → jeder Budget-Helper no-op (nicht-Gemma4-Modelle bleiben exakt
    im bisherigen Verhalten). confirm_id None ist erlaubt (bestätigungslos
    gedegradet: der Open setzt direkt den Kanal).
    """
    if tokenizer is None:
        return None
    open_id = _resolve_think_token(tokenizer, _THINK_OPEN_TOKEN)
    if open_id is None:
        return None
    close_id = _resolve_think_token(tokenizer, _THINK_CLOSE_TOKEN)
    if close_id is None:
        return None
    confirm_id = _resolve_think_token(tokenizer, _THINK_CONFIRM_TOKEN)
    return (open_id, close_id, confirm_id)


try:
    from transformers.generation.logits_process import (
        LogitsProcessor, LogitsProcessorList,
    )
except ImportError:  # pragma: no cover — sehr alte transformers
    from transformers import LogitsProcessor, LogitsProcessorList


class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    r"""Erzwingt ein Token-Budget für den Gemma-4-Think-Kanal.

    Semantik (der max_thinking_tokens-Doku folgend, App-Level realisiert):
    sobald `budget` Tokens IM Kanal erzeugt wurden, wird die nächste
    Token-Verteilung auf `<channel|>` (= close) eingeschnürt — alle anderen
    Logits auf torch.finfo(dtype).min. Das Modell beendet den Think-Block
    deterministisch und läuft im normalen Channel weiter (kein Hard-Stop,
    kein OOD-Truncation wie in PR #42112 diskutiert).

    State machine über die generierten Tokens:
      außerhalb: open_id → pending; (pending + confirm_id) → im Kanal
                 — "thought" als WORT im Fließtext öffnet KEINEN Kanal
      im Kanal:  close_id → außerhalb (count resettet); sonst count += 1
                 (NUR close resettet — das Wort "thought" kann als INHALT
                 im Kanal auftauchen, darf die Zählung nicht zurücksetzen)

    Der erste __call__ sieht den Prompt (input_ids == Init-Länge) →
    base_len anlegen, scores unangetastet. Danach werden pro Call nur die
    NEUEN ids[base_len:] konsumiert — kein O(Prompt)-Rescan pro Schritt.

    Vorbereiteter Kanal (Template-Tool-Fall: add_generation_prompt liefert
    `<|channel>thought\` am Prompt-Ende): der __init__-Prompt-Scan erkennt
    den offenen Kanal → Generierung startet IM Kanal, count=0.

    Maskierung mit finfo.min statt -inf: der Prozessor läuft NACH den
    Built-ins (_merge_criteria_processor_list hängt Custom ans Ende);
    temperature/top_k/top_p haben dann die finfo.min-Werte schon gesehen —
    softmax/argmax wählen deterministisch close (min ≈ -3.4e38).
    Batch = 1 (Server/UI serve single-user, wie StopOnEOT).
    """

    def __init__(self, input_ids, open_id, close_id, confirm_id, budget):
        seq = input_ids[0] if input_ids.dim() > 1 else input_ids
        base = list(seq.tolist())
        self.open_id = int(open_id)
        self.close_id = int(close_id)
        self.confirm_id = int(confirm_id) if confirm_id is not None else None
        self.budget = int(budget)
        self.in_think = False
        self.count = 0
        self.pending_confirm = False
        i = 0
        while i < len(base):
            tid = base[i]
            if not self.in_think and tid == self.open_id and self.confirm_id is not None:
                # Vorbereiteter Kanal: Bestätigung muss IM Prompt folgen.
                if i + 1 < len(base) and base[i + 1] == self.confirm_id:
                    self.in_think = True
                    self.count = 0
                    i += 2
                    continue
                # Bestätigungsloses Open endet mit dem Prompt — Generierung
                # startet außerhalb (pending endet mit dem Prompt).
            elif self.in_think and tid == self.close_id:
                self.in_think = False
                self.count = 0
            i += 1
        self.base_len = len(base)

    def _advance(self, tid):
        if self.in_think:
            if tid == self.close_id:
                self.in_think = False
                self.count = 0
            else:
                self.count += 1
        else:
            if tid == self.open_id:
                self.pending_confirm = True
            elif self.pending_confirm:
                self.pending_confirm = False
                if self.confirm_id is None or tid == self.confirm_id:
                    self.in_think = True
                    self.count = 0

    def __call__(self, input_ids, scores):
        seq = input_ids[0] if input_ids.dim() > 1 else input_ids
        n = int(seq.shape[0])
        if n <= self.base_len:
            # Erste Scores-Call: input_ids == Prompt (nichts Neues zu zählen;
            # der __init__-Scan hat die Prompt-Semantik schon gesetzt).
            return scores
        # Zuerst alle neu sichtbaren Tokens konsumieren (normalerweise genau
        # das eine seit dem letzten Call) — sonst klebt der State am letzten
        # konsumierten Token und die Maske würde in JEDEM Folgeschritt
        # wieder feuern (close-spam) statt einmalig.
        for tid in seq[self.base_len:].tolist():
            self._advance(tid)
        self.base_len = n
        if self.in_think and (self.budget <= 0 or self.count >= self.budget):
            # scores beschreibt das NEXT-token: count == budget inhaltliche
            # Kanal-Tokens → genau HIER wird close erzwungen. Der Folgeschritt
            # konsumiert dieses close → in_think=False → Maske wieder aus.
            mask = torch.full_like(scores, torch.finfo(scores.dtype).min)
            mask[:, self.close_id] = scores[:, self.close_id]
            return mask
        return scores


def _thinking_budget_kwargs(gen_kwargs, input_ids, thinking_budget,
                            tokenizer=None):
    """Thinking-Budget als LogitsProcessor an ein gen_kwargs-dict hängen.

    - thinking_budget None/bool/<=0/non-numeric → gen_kwargs UNVERÄNDERT
      (None = kein Budget am Modell; 0 = unbegrenzt)
    - Tokenizer kennt die Kanal-Tokens nicht (gemma3/minicpm/bonsai/unk)
      → unverändert (kein Frickel-Fallback)
    - bestehender logits_processor wird komponiert (List append)

    Nur für den plain model.generate-Pfad verwenden — generate_long und
    chunked_generate haben keine LogitsProcessor-Route.
    """
    b = thinking_budget
    if b is None or isinstance(b, bool) or not isinstance(b, (int, float)):
        return gen_kwargs
    if int(b) <= 0:
        return gen_kwargs
    if input_ids is None:
        return gen_kwargs
    ids = _think_channel_token_ids(tokenizer)
    if ids is None:
        return gen_kwargs
    open_id, close_id, confirm_id = ids
    processor = ThinkingBudgetLogitsProcessor(
        input_ids, open_id, close_id, confirm_id, int(b))
    existing = gen_kwargs.get("logits_processor")
    if isinstance(existing, LogitsProcessorList):
        existing.append(processor)
    elif existing is not None:
        try:
            existing.append(processor)
        except Exception:
            gen_kwargs["logits_processor"] = LogitsProcessorList([processor])
    else:
        gen_kwargs["logits_processor"] = LogitsProcessorList([processor])
    return gen_kwargs


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
        if _is_long_ctx_capable(model) and input_len > _PX_LONGCTX_THRESHOLD:
            # Stufe-3e: GF(3)-Lauf (qwen35_ptq, build_and_load_gf3 setzt
            # _px_long_ctx) → long_context.generate_long (KV-4bit-Cache +
            # Chunked-Prefill + eigene Decode-Loop). Muss VOR dem
            # 4b-chunked-Dispatch greifen — chunked_generate aus
            # scratches/4b-image ist gemma3-geometrisch gebaut und würde
            # die qwen3.5-GDN/Cache-Semantik verletzen.
            base["_px_use_long_ctx"] = True
            import sys as _sys
            print(f"[generate] long-context Pfad (T={input_len} > "
                  f"{_PX_LONGCTX_THRESHOLD}, GF(3)-Lauf → KV-4bit-Cache + "
                  f"chunked prefill)", file=_sys.stderr)
        elif not is_small_model and input_len > _LONG_INPUT_THRESHOLD:
            base["_px_use_chunked_prefill"] = True
            import sys as _sys
            print(f"[generate] auto chunked_prefill (input_len={input_len} > "
                  f"{_LONG_INPUT_THRESHOLD}; use_cache=False würde langsam, "
                  f"chunked_prefill ist 6x schneller)", file=_sys.stderr)

    base.pop("_input_len", None)  # cleanup internal marker

    return base


# Threshold: bei 4b + int8 funktioniert generate mit use_cache=True bis ~4500.
# Plan 3: Schwellwert für Auto-Switch zu chunked_prefill (Plan 3 Phase D).
# Empirisch gemessen mit profile_threshold_sweep.py (T=4000..9000, 4b+int8+PX):
#   use_cache=True funktioniert bis T=9000 ohne OOM (peak 7.67 GB,
#   3.89 GB headroom auf 11.56 GB GPU). InfLLM/chunked-attention macht
#   Score-Matrix-Materialisierung bounded, deshalb skaliert das linear
#   mit T statt quadratisch.
#   Vorher: 4500 → cross_model_holographic_01 (T=5371 text-only via
#   streaming_bridge) lief durch chunked_generate (~22.8s) statt direkt
#   use_cache=True (~19.6s, 14% schneller).
#   Mit 8800 als Schwellwert landen ALLE realistischen Session-Größen
#   (≤ 9k tokens) im use_cache=True-Pfad; chunked_generate bleibt als
#   Fallback für extreme Längen (>8800) erhalten.
# Quelle: scratches/4b-image/profile_threshold_sweep_results.json
_LONG_INPUT_THRESHOLD = 8800

# Stufe 3g: Der GF(3)-Lauf (qwen35_ptq, 262k-Vocab) materialisiert im
# HF-Short-Pfad beim Prefill VOCAB-BREITE Logits: T × 262144 × bf16
# ≈ T/2 MiB. CitMind-Systemprompt ≈ 4.2k Token → 2.2 GiB allein; mit
# Modell (5.9 GiB) + Prefill-Aktivierungen passt das nicht mehr in 12 GiB
# (beobachtet: OOM bei 428 MiB Rest, 10.72 GiB live). Der long_context
# -Pfad rechnet Logits nur je Chunk-Ende/Decode-Schritt und ist deshalb
# der speichersichere Weg — also für ternary ein deutlich tieferer
# Schwellwert als der (gemma3-4b-tunierte) globale 8800.
_PX_LONGCTX_THRESHOLD = 3000


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


def _is_long_ctx_capable(model) -> bool:
    """Stufe-3e: True wenn der Lauf _px_long_ctx trägt (GF(3)-Runtime)."""
    return bool(_find_px_attr(model, "_px_long_ctx", default=False))


def _px_pre_generation_cache_flush(model):
    """Konvention (CLAUDE.md): Cache vor jeder Generierung leeren.

    Wichtig beim GF(3)-Lauf: sein Langpfad hinterlässt große
    Allocator-Blöcke (kv4-Prefill-Workspace); ohne Flush startet der
    nächste Request mit fast vollem VRAM und stößt evtl. an die 12-GiB
    -Marke (beobachtet: OOM bei CitMind-Turn-1 nach einem 9.2k-Langlauf).
    """
    if _is_long_ctx_capable(model) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _import_long_context():
    """px_patches.ternary_bonsai_27b_px.long_context laden (Root im Pfad)."""
    from px_patches.ternary_bonsai_27b_px.long_context import generate_long
    return generate_long


def _generate_long_completion(model, input_ids, tokenizer, gen_kwargs,
                              input_len, stop=None, max_total_seq=None,
                              verbose=False):
    """Stufe-3e: Non-Stream-Long-Context-Completion (GF(3)-Lauf).

    Übersetzt model.generate-Welt-Kwargs in long_context.generate_long-Args
    und formatiert identisch zu generate_chat_completion zurück (Text +
    Token-Zählung; Stop-Strings als Post-Trim, HF-Parität).
    """
    generate_long = _import_long_context()
    g = dict(gen_kwargs)
    gen_kwargs.clear()
    eos_field = g.pop("eos_token_id", None) or []
    if isinstance(eos_field, int):
        eos_field = [eos_field]
    for k in ("stop_strings", "tokenizer", "stopping_criteria",
              "_px_use_chunked_prefill", "use_cache", "pad_token_id"):
        g.pop(k, None)
    g.setdefault("top_k", 0)     # HF-Parität: server setzt nur top_p
    res = generate_long(model, input_ids, streamer=None,
                        eos_token_ids=tuple(eos_field),
                        max_total_seq=max_total_seq, verbose=verbose, **g)
    new_tokens = res["generated"][0]
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
    thinking: Optional[bool] = None,
    thinking_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
) -> dict:
    """Non-streaming chat completion. Returns text + token counts."""
    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]
    processor = model_entry.get("processor")

    has_images = _has_image_content(messages)
    _force_chunked = False  # Plan 4: chunked-vision-encoder Pfad
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
        #
        # Plan 4: chunked-vision-encoding. Bei 3+ Bildern allokiert der
        # standard processor >6 GB extra für Vision-Tower activations
        # → peak 11+ GB → OOM auf 12 GB GPU. Chunked encoder macht 1 Bild
        # pro forward und löscht intermediates → peak ~7 GB.
        # Auch bei kurzem Input: chunked encoder ist sicherer, ABER langsamer
        # (~3× wegen 3× vision_tower forward). Switch-Logik:
        #   - N images > CHUNKED_VISION_THRESHOLD: chunked encoder
        #   - sonst: standard processor (schneller)
        cleaned, images = _extract_images(messages)
        n_images = len(images)
        CHUNKED_VISION_THRESHOLD = 2  # >2 Bilder → chunked
        if n_images > CHUNKED_VISION_THRESHOLD:
            try:
                from chunked_vision_encoder import chunked_process_multimodal
                chunked_inputs = chunked_process_multimodal(
                    model, processor, tokenizer, cleaned, images,
                )
                # Replace `inputs` with the chunked-encoder result, in a form
                # that the downstream code can dispatch to chunked_generate
                # via the use_chunked branch below.
                inputs = {
                    "input_ids": chunked_inputs["input_ids"],
                    "inputs_embeds": chunked_inputs["inputs_embeds"],
                    "attention_mask": chunked_inputs["attention_mask"],
                }
                # Force chunked path (chunked_generate with inputs_embeds)
                _force_chunked = True
                import sys as _sys
                print(f"[generate] multimodal {n_images} images → "
                      f"chunked-vision-encoder (Plan 4)", file=_sys.stderr)
            except Exception as e:
                import sys as _sys
                print(f"[generate] chunked-vision-encoder failed "
                      f"({type(e).__name__}: {str(e)[:200]}) → fallback "
                      f"to standard processor", file=_sys.stderr)
                prompt_text = processor.apply_chat_template(
                    cleaned, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=prompt_text, images=images,
                                   return_tensors="pt").to(model.device)
                _force_chunked = False
        else:
            prompt_text = processor.apply_chat_template(
                cleaned, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=prompt_text, images=images,
                                return_tensors="pt").to(model.device)
            _force_chunked = False
    else:
        # Text-only path (unchanged from prior behavior).
        processed_messages = [{"role": m.get("role", "user"), "content": _stringify_content(m.get("content", ""))} for m in messages]
        input_text = tokenizer.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=True,
            **_chat_template_kwargs(thinking, thinking_effort)
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        # Plan 7.2: Llama-Modelle (z.B. MiniCPM5-1B) kennen token_type_ids
        # nicht. Single source of truth: strip_unsupported_model_kwargs.
        inputs = strip_unsupported_model_kwargs(model, inputs)
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

    # Konvention (CLAUDE.md): Cache vor jeder Generierung leeren — beim
    # GF(3)-Lauf zwingend, sonst OOM im Folge-Request nach einem Langlauf.
    _px_pre_generation_cache_flush(model)

    with torch.no_grad():
        # Stufe-3e: GF(3)-Lauf + langer Input → long_context.generate_long
        # (hat keinen Multimodal-Input; qwen35_ptq hat processor=None, d.h.
        # Bild-Anfragen wurden vorab mit 400 abgewiesen).
        if gen_kwargs.pop("_px_use_long_ctx", False):
            return _generate_long_completion(
                model, inputs["input_ids"], tokenizer, gen_kwargs, input_len,
                stop=stop,
                max_total_seq=model_entry.get("registry", {}).get("max_length"),
                verbose=True)
        use_chunked = gen_kwargs.pop("_px_use_chunked_prefill", False)
        # Plan 4: chunked-vision-encoder liefert pre-merged inputs_embeds.
        # In diesem Fall MUSS chunked_generate laufen (sonst kriegt das
        # Model die Vision-Embeds nicht).
        if _force_chunked:
            use_chunked = True
        # Multimodal + chunked: Vision-Tokens werden vom processor an einer
        # festen Position eingefügt, die nicht zwingend im ersten Chunk
        # liegt. Aktuell nicht unterstützt — fallback auf model.generate
        # mit use_cache=False (T>4500 Pfad).
        is_multimodal = inputs.get("pixel_values") is not None
        if use_chunked and is_multimodal:
            gen_kwargs["use_cache"] = False
            import sys as _sys
            print(f"[generate] multimodal + long context (T={input_len}) → "
                  f"use_cache=False Fallback (chunked_generate hat noch "
                  f"keine Vision-Token-Position-Heuristik)", file=_sys.stderr)
            outputs = model.generate(**inputs, **gen_kwargs)
        elif use_chunked:
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
                pixel_values=inputs.get("pixel_values"),
                **({"token_type_ids": inputs["token_type_ids"]} if inputs.get("token_type_ids") is not None else {}),
                # Plan 4: chunked-vision-encoder liefert pre-merged
                # text+image embeddings → inputs_embeds statt pixel_values.
                # Wenn chunked-vision-encoder aktiv war, ist
                # inputs["inputs_embeds"] gesetzt und inputs["pixel_values"]
                # NICHT gesetzt.
                inputs_embeds=inputs.get("inputs_embeds"),
            )
        else:
            # Plan 2026-10-05 (Gemma4 Thinking-Budget): App-Level
            # LogitsProcessor — nur im plain-Pfad (model.generate);
            # generate_long/chunked_generate haben keine LogitsProcessor-
            # Route (und dort — bonsai/gemma3 — fehlt die Kanal-Infrastruktur
            # ohnehin, Helper no-op).
            gen_kwargs = _thinking_budget_kwargs(
                gen_kwargs, inputs["input_ids"], thinking_budget, tokenizer)
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
    thinking: Optional[bool] = None,
    thinking_effort: Optional[str] = None,
    thinking_budget: Optional[int] = None,
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
            processed_messages, tokenize=False, add_generation_prompt=True,
            **_chat_template_kwargs(thinking, thinking_effort)
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        # Plan 7.2: Llama-Modelle (z.B. MiniCPM5-1B) kennen token_type_ids
        # nicht. Single source of truth: strip_unsupported_model_kwargs.
        inputs = strip_unsupported_model_kwargs(model, inputs)
    input_len = inputs["input_ids"].shape[1]

    # Konvention (CLAUDE.md): Cache vor jeder Generierung leeren — beim
    # GF(3)-Lauf zwingend, sonst OOM im Folge-Request nach einem Langlauf.
    _px_pre_generation_cache_flush(model)

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
    # Multimodal + chunked: aktuell nicht unterstützt (Vision-Token-Position
    # ist nicht in erstem Chunk garantiert). Fallback use_cache=False.
    use_long_stream = gen_kwargs.pop("_px_use_long_ctx", False)
    use_chunked_stream = gen_kwargs.pop("_px_use_chunked_prefill", False)
    is_multimodal = inputs.get("pixel_values") is not None
    if use_long_stream and not is_multimodal:
        # Stufe-3e: GF(3)-Lauf → long_context.generate_long mit Streamer
        # (decode_loop pusht je Token in den TextIteratorStreamer;
        # stop_strings werden nicht beachtet — der 4b-chunked-Stream-Pfad
        # verhält sich genauso, Post-Trim übernimmt der Bridge-Consumer).
        generate_long = _import_long_context()
        g = dict(gen_kwargs)
        # Doppel-Arg vermeiden: Im Stream-Pfad ist gen_kwargs ein
        # dict(**inputs, ...) (model.generate-Welt), generate_long bekommt
        # input_ids/streamer/max_new_tokens aber explizit. Ohne Pop: TypeError
        # "got multiple values for keyword argument 'streamer'/'input_ids'".
        g.pop("max_new_tokens", None)
        g.pop("streamer", None)
        g.pop("input_ids", None)
        for tensor_key in ("attention_mask", "token_type_ids",
                           "inputs_embeds"):
            g.pop(tensor_key, None)
        # Same Junk-Übersetzung wie _generate_long_completion (HF-Parität).
        for junk in ("stop_strings", "tokenizer", "stopping_criteria",
                     "_px_use_chunked_prefill", "use_cache", "pad_token_id",
                     "_input_len"):
            g.pop(junk, None)
        g.setdefault("top_k", 0)     # HF-Parität: server setzt nur top_p
        eos_field = g.get("eos_token_id", [])
        if isinstance(eos_field, int):
            eos_field = [eos_field]
        registry_max = model_entry.get("registry", {}).get("max_length")

        def _long_worker():
            try:
                generate_long(model, inputs["input_ids"],
                              max_new_tokens=max_tokens,
                              streamer=streamer,
                              eos_token_ids=tuple(eos_field),
                              max_total_seq=registry_max,
                              verbose=True, **g)
            except Exception:
                import traceback as _tb
                _tb.print_exc()
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = Thread(target=_long_worker)
        thread.start()
    elif use_chunked_stream and is_multimodal:
        gen_kwargs["use_cache"] = False
        import sys as _sys
        print(f"[generate_stream] multimodal + long context (T={input_len})"
              f" → use_cache=False Fallback (kein chunked+vision)", file=_sys.stderr)
        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
    elif use_chunked_stream:
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
                    pixel_values=inputs.get("pixel_values"),
                    **({"token_type_ids": inputs["token_type_ids"]} if inputs.get("token_type_ids") is not None else {}),
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
        # Plan 2026-10-05 (Gemma4 Thinking-Budget): nur im plain-Pfad
        # (model.generate) — long/chunked haben keine LogitsProcessor-Route.
        gen_kwargs = _thinking_budget_kwargs(
            gen_kwargs, inputs["input_ids"], thinking_budget, tokenizer)
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