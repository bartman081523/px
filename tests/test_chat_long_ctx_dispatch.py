"""Tests: UI-Lang-Kontext-Dispatch + OOM-Guard (Plan 2026-10-05).

Auslöser (Live-Crash): TXT-Anhang im UI-Chat → langer Prompt (14er-History
+ inlined ```txt-Block) → chat_fn lief IMMER den HF-Short-Pfad
(model.generate) → SDPA-math-Backend materialisiert die O(n²)-Score-Matrix
→ torch.OutOfMemoryError (2.48 GiB bei ~1.6 GiB frei) → crash_handler
(threading.excepthook) terminierte den GESAMTEN Server.

Fix (ohne Token-Cap — globale User-Regel, memory never-token-caps):
1. gen_kwargs["_input_len"] → _px_gen_kwargs (wie in allen generators-
   Pfaden: generate/generate_stream/_generate_long_completion) → Marker
   _px_use_long_ctx / _px_use_chunked_prefill.
2. generate_with_lock dispatcht auf die etablierten Lang-Kontext-Worker:
   long_context.generate_long (KV-4Bit + Chunked-Prefill, ternary/bonsai)
   bzw. scratches/4b-image chunked_generate (gemma3-4b) — identisch zum
   Server-Stream-Pfad (generators.stream_chat_completion).
3. OOM-Guard: OutOfMemoryError → streamer.end() + sichtbare Chat-Note,
   NIEMALS Prozess-Kill.

Runtime-Tests nur für die puren Helfer (_long_ctx_generate_kwargs,
_is_cuda_oom, _oom_note_text); die chat_fn-Verdrahtung ist Source-Pin
(kein GPU/Gradio-Lauf im Test).
"""
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gradio_tabs.chat_tab import (
    _long_ctx_generate_kwargs,
    _is_cuda_oom,
    _oom_note_text,
)


def _chat_fn_source():
    """Source des chat_fn-Blocks (bis zum nächsten Top-Level def/class)."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gradio_tabs", "chat_tab.py",
    )
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"^def chat_fn\(.*?(?=^def |^class )", src,
                  re.DOTALL | re.MULTILINE)
    assert m, "chat_fn-Block nicht gefunden"
    return m.group(0)


# ─── Runtime: Übersetzung model.generate-Welt → generate_long-Welt ──────

def test_long_ctx_generate_kwargs_strips_hf_junk():
    """Alle HF/Tensor-Keys, die decode_loop nicht kennt, sind weg — die
    Sampling-Keys und eos_token_id bleiben (generate_long filtert
    sample_cfg selbst, Server-Parität g.get('eos_token_id'))."""
    g = _long_ctx_generate_kwargs({
        "input_ids": "T", "attention_mask": "AM", "token_type_ids": "TT",
        "inputs_embeds": "IE", "streamer": "S", "max_new_tokens": 512,
        "temperature": 1e-10, "top_p": 0.95, "repetition_penalty": 1.15,
        "do_sample": False, "stop_strings": ["<end_of_turn>"],
        "tokenizer": "tok", "stopping_criteria": "crit",
        "use_cache": True, "pad_token_id": 0, "_input_len": 9123,
        "_px_use_long_ctx": True, "_px_use_chunked_prefill": False,
        "eos_token_id": [106, 151645],
    })
    for gone in ("input_ids", "attention_mask", "token_type_ids",
                 "inputs_embeds", "streamer", "max_new_tokens",
                 "stop_strings", "tokenizer", "stopping_criteria",
                 "use_cache", "pad_token_id", "_input_len",
                 "_px_use_long_ctx", "_px_use_chunked_prefill"):
        assert gone not in g, f"{gone} hätte gestrippt werden müssen"
    # Sampling-Keys durchgereicht (generate_long: s_cfg-Whitelist).
    assert g["temperature"] == 1e-10
    assert g["top_p"] == 0.95
    assert g["repetition_penalty"] == 1.15
    assert g["do_sample"] is False  # darf drin sein — wird von generate_long gedroppt
    assert g["eos_token_id"] == [106, 151645]  # Server-Parität: bleibt, wird explizit übergeben
    # HF-Parität: UI setzt kein top_k → decode-Default 0 (= off).
    assert "top_k" not in g or g["top_k"] == 0


def test_long_ctx_generate_kwargs_sets_top_k_default_zero():
    """top_k default 0 (HF-Parität: UI setzt nur top_p)."""
    g = _long_ctx_generate_kwargs({"temperature": 0.7, "top_p": 0.95})
    assert g["top_k"] == 0


def test_long_ctx_generate_kwargs_keeps_existing_top_k():
    """setdefault — ein vorhandenes top_k wird NICHT überschrieben."""
    g = _long_ctx_generate_kwargs({"top_k": 40})
    assert g["top_k"] == 40


def test_is_cuda_oom_runtimeerror_text():
    """Ältere Layouts: RuntimeError mit 'out of memory' im Text (der
    Live-Crash sah so aus: torch.OutOfMemoryError erbt heutzutage direkt)."""
    assert _is_cuda_oom(RuntimeError(
        "CUDA out of memory. Tried to allocate 2.48 GiB. GPU 0 has a total "
        "capacity of 11.56 GiB of which 1.61 GiB is free."))
    assert not _is_cuda_oom(RuntimeError("shape mismatch, something else"))


def test_is_cuda_oom_torch_class():
    """torch ≥ 2.5: OutOfMemoryError-Klasse direkt erkannt."""
    import torch
    oom_cls = getattr(torch, "OutOfMemoryError", None)
    if oom_cls is None:
        return  # alte torch-Version — erste Variante deckt ab
    assert _is_cuda_oom(oom_cls("out of memory"))
    assert not _is_cuda_oom(ValueError("nope"))


def test_oom_note_text_contains_prompt_count():
    """Note nennnt den Prompt-Umfang und ist OOM-diagnostisch."""
    note = _oom_note_text(9123)
    assert "OOM" in note
    assert "9123" in note
    assert note.startswith("\n\n")  # hängt an partial_text an


# ─── Source-Pins: chat_fn-Verdrahtung ────────────────────────────────────

def test_chat_fn_sets_input_len_before_px_gen_kwargs():
    """`_input_len` muss gesetzt sein, BEVOR _px_gen_kwargs routing
    entscheidet (Marker _px_use_long_ctx kann sonst nie feuern)."""
    block = _chat_fn_source()
    idx_len = block.find('gen_kwargs["_input_len"] = int(inputs["input_ids"].shape[1])')
    idx_px = block.find("gen_kwargs = _px_gen_kwargs(model, gen_kwargs)")
    assert idx_len != -1, "_input_len-Assignment fehlt in chat_fn"
    assert idx_px != -1
    assert idx_len < idx_px, (
        "_input_len muss VOR _px_gen_kwargs gesetzt werden — erst dann "
        "entscheidet das Routing (_PX_LONGCTX_THRESHOLD=3000 für GF3)."
    )


def test_chat_fn_initializes_dispatch_flags_before_import_guard():
    """ImportError-Sicherheit: Marker/oom_error sind VOR dem try initialisiert
    (sonst NameError in generate_with_lock, wenn generators fehlt)."""
    block = _chat_fn_source()
    assert "use_long_stream = False" in block
    assert "use_chunked_stream = False" in block
    assert "oom_error = None" in block
    idx_flags = block.find("use_long_stream = False")
    idx_px = block.find("gen_kwargs = _px_gen_kwargs(model, gen_kwargs)")
    assert idx_flags < idx_px


def test_chat_fn_pops_dispatch_markers_after_px_gen_kwargs():
    """Server-Parität: Marker-Pops (generate()-fremde Keys → ValueError in
    _validate_model_kwargs, wenn sie durchrutschen)."""
    block = _chat_fn_source()
    assert 'gen_kwargs.pop("_px_use_long_ctx", False)' in block
    assert 'gen_kwargs.pop("_px_use_chunked_prefill", False)' in block
    idx_pop = block.find('gen_kwargs.pop("_px_use_long_ctx", False)')
    idx_px = block.find("gen_kwargs = _px_gen_kwargs(model, gen_kwargs)")
    assert idx_pop > idx_px


def test_chat_fn_dispatches_to_etablierte_lang_context_worker():
    """generate_with_lock: drei Zweige wie der Server-Stream-Pfad —
    generate_long (KV-4Bit+Chunked-Prefill), multimodal-Fallback
    (use_cache=False), chunked_generate (gemma3-4b)."""
    block = _chat_fn_source()
    for pin in (
        "if use_long_stream and not is_multimodal_inputs:",
        "_run_long_stream()",
        "elif use_chunked_stream and is_multimodal_inputs:",
        'gen_kwargs["use_cache"] = False',
        "elif use_chunked_stream:",
        "_run_chunked_stream()",
    ):
        assert pin in block, f"Dispatch-Pin fehlt: {pin}"


def test_run_long_stream_mirrors_server_worker():
    """generate_long-Aufruf mit demselben Streamer + eos + registry_max."""
    block = _chat_fn_source()
    for pin in (
        "generate_long(model, inputs[\"input_ids\"],",
        "streamer=streamer,",
        "eos_token_ids=tuple(eos_field),",
        "max_total_seq=registry_max,",
        "_import_long_context()",
    ):
        assert pin in block, f"long-Worker-Pin fehlt: {pin}"


def test_oom_guards_in_all_three_paths():
    """OOM-Guard im plain-Pfad (re-raise für Nicht-OOM — Crash-Policy
    bleibt), Worker fangen Exception und beenden den Stream sauber."""
    block = _chat_fn_source()
    # plain: OOM abfangen, alles andere weiterwerfen.
    assert "if not _is_cuda_oom(_exc):" in block
    assert "raise" in block
    # Worker: streamer.end() (3 paths: plain + long + chunked).
    assert block.count("streamer.end()") >= 3
    # oom_error wird in allen drei Pfaden gesetzt.
    assert block.count("oom_error = ") >= 3
    assert "nonlocal oom_error" in block


def test_oom_note_displayed_but_not_persisted():
    """Note geht in die Chatbot-Anzeige (display_text), in die Session
    wird der reine Modell-Output persistiert (partial_text) — sie würde
    sonst im nächsten Turn im Kontext auftauchen."""
    block = _chat_fn_source()
    assert "_oom_note_text(" in block
    idx_note = block.find("_oom_note_text(")
    idx_save = block.find('"content": partial_text')
    assert idx_save != -1 and idx_save > idx_note, (
        "save_session muss partial_text (ohne Note) persistieren"
    )


def test_no_token_cap_introduced():
    """Globale User-Regel (2026-10-05): der Fix darf NIEMALS mit einem
    Token-/Größen-Cap arbeiten. Pins: kein MAX_TEXT_FILE_BYTES-Deal im
    chat_fn-Block, kein Truncation-Marker, max_new_tokens kommt aus dem
    UI-Slider (mt), der Prompt läuft vollständig durch."""
    block = _chat_fn_source()
    for forbidden in ("MAX_TEXT_FILE_BYTES", "…[truncated]",
                      "input_ids[:, -", "truncation=True", "max_length="):
        assert forbidden not in block, (
            f"Token-Cap/Truncation verboten (globale Regel): {forbidden}"
        )


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _main() else 0)