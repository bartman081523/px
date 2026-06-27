"""Registry der 5 Tag-Produktion-Varianten für Plan 6.2.

Jede Variante ist eine ``apply(base_messages, user_prompt) -> messages``-
Funktion. Pattern: Input ist ``[{"role": "user", "content": user_prompt}]``,
Output ist die vollständige messages-Liste (System + User), bereit für
``tokenizer.apply_chat_template``.

Keine Modell-Aufrufe, keine Generation — nur System-Prompt-Bau.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from . import a_baseline, b_sanskrit, c_few_shot, d_abc, e_tag_only

ApplyFn = Callable[[List[dict], str], List[dict]]

VARIANTS: Dict[str, Tuple[str, ApplyFn]] = {
    "A": ("CitMind + Standard-Snip (Baseline, Plan 6.1)", a_baseline.apply),
    "B": ("CitMind + Sanskrit-Mapping-Snip + Standard-Snip", b_sanskrit.apply),
    "C": ("CitMind + Standard-Snip + 3 Few-Shot-Turns", c_few_shot.apply),
    "D": ("CitMind + ABC-Notation-Snip (statt Vocoder-Snip)", d_abc.apply),
    "E": ("Neutral-Profil + Standard-Snip (kein CitMind)", e_tag_only.apply),
}


def list_variants() -> List[Tuple[str, str]]:
    """Gibt ``[(variant_id, label), ...]`` zurück."""
    return [(k, v[0]) for k, v in VARIANTS.items()]


def get_variant(variant_id: str) -> Tuple[str, ApplyFn]:
    """Lookup Variante nach ID. Wirft KeyError bei unbekannter ID."""
    return VARIANTS[variant_id]


__all__ = ["VARIANTS", "ApplyFn", "list_variants", "get_variant"]


# Avoid "imported but unused" for the module-level imports.
_ = (a_baseline, b_sanskrit, c_few_shot, d_abc, e_tag_only)
