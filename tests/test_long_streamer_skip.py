"""Tests für den Streamer-Skip-Fix im Long-Pfad (G1, Bug f31eff3e idx17/19).

Bug: TextIteratorStreamer(skip_prompt=True) konsumiert das ERSTE put als
"Prompt-Skip", wenn die Prompt-Ids nie gepusht werden — im generate_long/
chunked_generate-Pfad frisst der Guard also das erste generierte Token
("elen Dank" statt "Vielen Dank").

Run: /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python tests/test_long_streamer_skip.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from px_patches.ternary_bonsai_27b_px.long_context import _consume_streamer_skip


class _FakeStreamer:
    """Minimalsurrogat für TextIteratorStreamer-Guard-Semantik."""
    def __init__(self, skip_prompt=True):
        self.next_tokens_are_prompt = skip_prompt


def test_consume_on_fresh_streamer():
    s = _FakeStreamer(skip_prompt=True)
    _consume_streamer_skip(s)
    assert s.next_tokens_are_prompt is False


def test_consume_is_idempotent():
    s = _FakeStreamer(skip_prompt=True)
    _consume_streamer_skip(s)
    _consume_streamer_skip(s)
    assert s.next_tokens_are_prompt is False


def test_consume_noop_on_already_consumed():
    s = _FakeStreamer(skip_prompt=True)
    s.next_tokens_are_prompt = False
    _consume_streamer_skip(s)
    assert s.next_tokens_are_prompt is False


def test_consume_tolerates_none():
    _consume_streamer_skip(None)  # Plain-Pfad: kein Streamer, kein Effekt


def test_consume_tolerates_attr_absent():
    class _Bare:
        pass
    s = _Bare()
    _consume_streamer_skip(s)   # hasattr-Fallback: kein AttributeError
    assert not hasattr(s, "next_tokens_are_prompt")


def test_real_text_iterator_streamer_attr():
    """Der echte Streamer trägt das Attribut (venv transformers 5.13.0) —
    sonst ist der Fix ins Leere gegriffen."""
    import torch
    from transformers.generation.streamers import TextIteratorStreamer

    class _Tok:
        def decode(self, ids, **kw):
            return ""

    s = TextIteratorStreamer(_Tok(), skip_prompt=True)
    assert getattr(s, "next_tokens_are_prompt", None) is True
    put = getattr(s, "put", None)
    assert callable(put)
    _consume_streamer_skip(s)
    assert s.next_tokens_are_prompt is False
    # Erstes put fließt jetzt durch (Guard ist konsumiert).
    s.put(torch.tensor([[7]]))
    assert next(iter(s)) == ""
    s.end()
    try:
        next(s)
    except StopIteration:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
    print("ALL PASS")