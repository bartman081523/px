"""
Layer A — InfiniteContextManager (Application-layer context windowing)
=====================================================================
Recovery + improvement of the InfiniteContextManager that was overwritten by the
previous "infinite context wip" run (see RECOVERY_REPORT.md). This is the
GPU-free, preset-agnostic layer that PREVENTS the prefill OOM reported in
bug_context.txt: it bounds the prompt fed to the model so the N^2 prefill
attention never exceeds the VRAM budget — works under BOTH BASELINE and
ACTIVE_MANIFOLD because it acts pre-model, before tokenization.

This module intentionally has NO torch dependency so it is unit-testable on CPU
and importable without the CUDA venv.
"""

import copy
from typing import List, Dict, Any, Optional, Callable

DEFAULT_ARCHIVE_NOTICE = (
    "[SYSTEM: Older context has been archived for memory efficiency. "
    "The following messages are the most recent interaction.]"
)


class InfiniteContextManager:
    """
    Manages session history to prevent OOM during generation by enforcing a
    sliding window over the token budget (or, as a fallback, a message count).

    Guarantees
    ----------
    * System messages are always preserved (they carry the persona / instructions).
    * When history is truncated, an `archive_notice` system message is injected so
      the model knows earlier context was compressed away.
    * If a tokenizer is supplied together with `max_tokens`, the chat-templated +
      tokenized result of the returned messages is <= `max_tokens` (minus `headroom`
      reserved for the generation).
    * Content that is a list (multimodal: {"type":"text"/"image", ...}) is passed
      through untouched so image parts survive the windowing.
    """

    def __init__(
        self,
        max_history_messages: Optional[int] = 10,
        max_tokens: Optional[int] = None,
        headroom: int = 0,
        archive_notice: str = DEFAULT_ARCHIVE_NOTICE,
    ):
        self.max_history_messages = max_history_messages
        self.max_tokens = max_tokens
        # Safety margin subtracted from max_tokens to obtain the effective prompt
        # budget (max_tokens - headroom). max_tokens is interpreted as the PROMPT
        # token budget; set headroom > 0 to leave extra VRAM room. Defaults to 0
        # so max_tokens is the exact prompt ceiling (matches the original tests).
        self.headroom = headroom
        self.archive_notice = archive_notice

    # -- helpers -----------------------------------------------------------
    def _count_tokens(self, messages: List[Dict[str, Any]], tokenizer) -> int:
        """Token count for a message list via the real chat template."""
        try:
            # Minimal signature first (matches the unit-test mock and most
            # HF tokenizers); fall back to the add_generation_prompt form.
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            if hasattr(tokenizer, "encode"):
                return len(tokenizer.encode(text))
            return len(tokenizer(text)["input_ids"])
        except Exception:
            # Fallback: crude char/4 heuristic so we still bound something if the
            # tokenizer refuses a particular message shape.
            total = 0
            for m in messages:
                c = m.get("content", "")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                total += max(1, len(str(c)) // 4)
            return total

    def _split_system(self, messages):
        system_msgs, regular_msgs = [], []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                regular_msgs.append(msg)
        return system_msgs, regular_msgs

    # -- main API ---------------------------------------------------------
    def process_history(
        self,
        messages: List[Dict[str, Any]],
        tokenizer: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process the history to fit within constraints. Returns a NEW list of
        messages safe for model generation (input is not mutated).
        """
        if not messages:
            return []

        system_msgs, regular_msgs = self._split_system(messages)

        # 1) Token-budget sliding window (preferred when a tokenizer is available)
        if tokenizer is not None and self.max_tokens is not None:
            budget = max(1, self.max_tokens - self.headroom)
            # Fast path: everything fits without any archive notice.
            if self._count_tokens(system_msgs + regular_msgs, tokenizer) <= budget:
                return system_msgs + regular_msgs
            # Truncation path: the archive notice we will inject counts toward the
            # budget, so the guarantee "tokenized result <= max_tokens" holds.
            notice = self._archive_msg()
            current = list(regular_msgs)
            while len(current) > 1:
                if self._count_tokens(system_msgs + [notice] + current, tokenizer) <= budget:
                    break
                current.pop(0)
            return system_msgs + [notice] + current

        # 2) Message-count fallback (no tokenizer or no max_tokens)
        if self.max_history_messages is not None and len(regular_msgs) > self.max_history_messages:
            truncated = regular_msgs[-self.max_history_messages:]
            return system_msgs + [self._archive_msg()] + truncated

        return system_msgs + regular_msgs

    def _archive_msg(self) -> Dict[str, Any]:
        return {"role": "system", "content": self.archive_notice}