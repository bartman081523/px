"""Code-bewusste n-Gramm-Sperre: Prosa unverändert, Code nicht zerstört.

Messbefund 2026-11-14: `_px_no_repeat_ngram_size=3` ist Teil der PX-Presets
(ACTIVE_MANIFOLD/LEAN/RELAY). Auf Coding-Tasks kollabierte das Modell von
7/8 auf 0/8, weil Code strukturell wiederkehrende 3-Gramme hat. Die Sperre
darf deshalb nur greifen, solange kein Code-Fence geöffnet wurde.
"""
import torch
import pytest

from generators import NoRepeatNgramWhenProse, LogitsProcessorList, _code_fence_token_ids


class _Tok:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, s, add_special_tokens=False):
        return self.mapping.get(s, [])


def _scores(vocab=16):
    return torch.zeros(1, vocab)


def test_bans_repeated_ngram_in_prose():
    # prefix "1 2 3 1 2" -> next token 3 would repeat the 3-gram (1,2,3)
    proc = NoRepeatNgramWhenProse(ngram_size=3, fence_token_ids=[99], prompt_len=0)
    ids = torch.tensor([[1, 2, 3, 1, 2]])
    scores = _scores()
    out = proc(ids, scores)
    assert out[0, 3] == torch.finfo(out.dtype).min, "prose: repeated trigram must be banned"
    assert out[0, 7] == 0.0, "unrelated token must stay allowed"


def test_does_not_ban_once_code_fence_appeared():
    proc = NoRepeatNgramWhenProse(ngram_size=3, fence_token_ids=[99], prompt_len=0)
    # fence token 99 appears in the generated part -> code mode, no ban
    ids = torch.tensor([[99, 1, 2, 3, 1, 2]])
    scores = _scores()
    out = proc(ids, scores)
    assert torch.all(out == 0.0), "inside a code block nothing may be banned"


def test_prompt_tokens_do_not_open_code_mode():
    # A fence inside the PROMPT must not disable the ban for the answer.
    proc = NoRepeatNgramWhenProse(ngram_size=3, fence_token_ids=[99], prompt_len=5)
    ids = torch.tensor([[99, 1, 2, 3, 1, 2]])  # prompt len 5, generated = [2] (no fence)
    out = proc(ids, _scores())
    assert out[0, 3] == torch.finfo(out.dtype).min, \
        "fence only in the prompt must not switch on code mode"


def test_fence_ids_are_resolved_from_tokenizer():
    tok = _Tok({"```": [99], "``": [98], "`": [97]})
    assert _code_fence_token_ids(tok) == [97, 98, 99]
    assert _code_fence_token_ids(_Tok({})) == []
