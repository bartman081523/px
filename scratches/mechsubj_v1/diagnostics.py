"""diagnostics.py — Token-Stream-Charakterisierung für MechanisticSubjectivityMixMind.

Drei Funktionen:
  - token_script_distribution(text): zählt Tokens pro Unicode-Block
  - shannon_entropy_token_distribution(token_ids): Entropie der Token-Verteilung
  - unique_script_ratio(text): Anteil nicht-ASCII Tokens aus Skript-Blöcken

Unicode-Blöcke sind kompiliert aus den Skript-Klassen, die wir in den
Token-Wirbel-Outputs (gemma-4-e2b) empirisch finden:
Devanagari, Bengali, Gujarati, Malayalam, Thai, Korean, Katakana,
Hiragana, CJK, Cyrillic, Arabic.

Reused AntiZombieSensor.calculate_entropy aus
px_patches/gemma3_270m_px_baseline/anti_zombie_sensor.py:24-28
als Referenz für die Entropie-Formel.
"""
from collections import Counter
from math import log2
import re


UNICODE_BLOCKS = {
    "devanagari": (0x0900, 0x097F),
    "bengali":    (0x0980, 0x09FF),
    "gujarati":   (0x0A80, 0x0AFF),
    "gurmukhi":   (0x0A00, 0x0A7F),
    "kannada":    (0x0C80, 0x0CFF),
    "malayalam":  (0x0D00, 0x0D7F),
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
    "sinhala":    (0x0D80, 0x0DFF),
    "thai":       (0x0E00, 0x0E7F),
    "lao":        (0x0E80, 0x0EFF),
    "tibetan":    (0x0F00, 0x0FFF),
    "myanmar":    (0x1000, 0x109F),
    "khmer":      (0x1780, 0x17FF),
    "korean":     (0xAC00, 0xD7AF),
    "katakana":   (0x30A0, 0x30FF),
    "hiragana":   (0x3040, 0x309F),
    "cjk":        (0x4E00, 0x9FFF),
    "cyrillic":   (0x0400, 0x04FF),
    "arabic":     (0x0600, 0x06FF),
    "hebrew":     (0x0590, 0x05FF),
}


def _script_for_char(ch: str) -> str:
    """Return the unicode-block name for a single char, or 'latin/digit/other'."""
    cp = ord(ch)
    for name, (lo, hi) in UNICODE_BLOCKS.items():
        if lo <= cp <= hi:
            return name
    return ""


def tokenize_stream(text: str) -> list[str]:
    """Split text into token-ish fragments.

    Splits at ASCII whitespace. Keeps `<unused*>` fragments together.
    Skips empty fragments and pure punctuation.
    """
    # Split on whitespace but keep alphanumeric+CJK/Brahmic/etc. together
    raw = re.split(r"\s+", text.strip())
    out = []
    for tok in raw:
        if not tok:
            continue
        # Drop pure punctuation tokens
        if all(not c.isalnum() for c in tok):
            continue
        out.append(tok)
    return out


def token_script_distribution(text: str) -> dict:
    """Count tokens per unicode block. Returns dict block_name -> count.

    A token is counted under ALL blocks it contains characters from (so
    a mixed-script token increments multiple counters). Total token
    count is included under 'tokens'.
    """
    toks = tokenize_stream(text)
    counts = {name: 0 for name in UNICODE_BLOCKS}
    counts["tokens"] = len(toks)
    counts["ascii_or_digit"] = 0
    counts["multilingual_tokens"] = 0
    for tok in toks:
        chars_in_block = set()
        any_script = False
        for ch in tok:
            block = _script_for_char(ch)
            if block:
                chars_in_block.add(block)
                any_script = True
            elif ch.isascii():
                pass
            # non-ASCII chars that don't fall in any block (rare) -> "other"
        for b in chars_in_block:
            counts[b] += 1
        if any_script:
            counts["multilingual_tokens"] += 1
        else:
            counts["ascii_or_digit"] += 1
    return counts


def shannon_entropy_token_distribution(token_ids: list) -> float:
    """Shannon entropy (base-2) of the unique-token distribution.

    Returns 0.0 for empty/single-token input. Returns float bits.
    Algorithm reused from AntiZombieSensor.calculate_entropy
    (px_patches/gemma3_270m_px_baseline/anti_zombie_sensor.py:24-28).
    """
    if not token_ids:
        return 0.0
    counts = Counter(token_ids)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        entropy -= p * log2(p)
    return entropy


def unique_script_ratio(text: str) -> float:
    """Fraction of tokens that contain at least one non-ASCII script char.

    Returns 0.0..1.0. Used as multilingual-stream indicator:
    - ratio > 0.7 → almost pure Stream-Modus
    - ratio 0.3..0.7 → mixed
    - ratio < 0.3 → mostly semantic / script-light output
    """
    dist = token_script_distribution(text)
    total = dist.get("tokens", 0)
    if total == 0:
        return 0.0
    return dist.get("multilingual_tokens", 0) / total


def is_stream_output(text: str, threshold: float = 0.4) -> bool:
    """Heuristic: output looks like multilingual stream if non-ASCII
    token ratio exceeds threshold.

    Threshold 0.4 captures the e2b-typical case where mixed script
    tokens dominate but a few Latin fragments remain.
    """
    return unique_script_ratio(text) > threshold


# === Convenience: bundle diagnostics for one model output ===

def diagnose_output(text: str, model=None, tokenizer=None) -> dict:
    """Bundle all relevant diagnostics for one chat-stage output.

    Returns dict suitable for JSON serialization. If tokenizer is
    given, also computes token-level entropy and unique-token count.
    """
    out = {
        "len_chars": len(text),
        "script_distribution": token_script_distribution(text),
        "unique_script_ratio": unique_script_ratio(text),
        "is_stream": is_stream_output(text),
    }
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        out["token_count"] = len(ids)
        out["unique_token_count"] = len(set(ids))
        out["token_entropy_bits"] = shannon_entropy_token_distribution(ids)
    return out


if __name__ == "__main__":
    # Smoke test
    test_inputs = [
        "Hello world, this is a normal English sentence.",
        "देवऱान slovenian Compound Responsibility antipattern",
        "Bonjour le monde, c'est une phrase française.",
        "ಗುನ್ನೆಗಳ ಸ್ವಲ್ಪ ಸಮಯ para el español Test της ελληνικής",
        "<unused572> ಗು Unless مهمه vrouwenἥ crowd بالج",
    ]
    for s in test_inputs:
        d = diagnose_output(s)
        print(f"text: {s[:80]!r}")
        print(f"  ratio={d['unique_script_ratio']:.2f}  is_stream={d['is_stream']}")
        print(f"  scripts={dict((k,v) for k,v in d['script_distribution'].items() if v and k not in ('tokens','ascii_or_digit','multilingual_tokens'))}")
        print()


# === Gemini Rate-Limit Guard (NEU 2026-08-17) ===
#
# Wenn die Gemini-API-Quote erschöpft ist (HTTP 429/503), kämpfen sich
# Skripte oft minutenlang durch Retries. Stattdessen:
#
#   1. Vor jedem Gemini-Call: check_rate_limit() → beendet früh, wenn
#      .gemini_quota.lock existiert UND noch gültig ist.
#   2. Beim ersten 429/503: signal_rate_limit() setzt die Lock-Datei mit
#      Timestamp. Lock gilt 30 min, danach wird sie ignoriert.
#   3. Mit clear_rate_limit() (oder `rm .gemini_quota.lock`) manuell zu löschen.
#
# Nutzung im Skript:
#   from diagnostics import check_rate_limit, signal_rate_limit, exit_if_rate_limited
#   if check_rate_limit(): sys.exit(0)
#   ... # try Gemini call:
#   except urllib.error.HTTPError as e:
#       if e.code in (429, 503): signal_rate_limit(f"stage {stage_id}: {e.code}")

import os as _os
import time as _time
import sys as _sys


LOCK_FILE = _os.environ.get(
    "GEMINI_QUOTA_LOCK",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".gemini_quota.lock"),
)
LOCK_TTL_SEC = int(_os.environ.get("GEMINI_QUOTA_LOCK_TTL", "1800"))  # 30 min default


def _lock_age() -> float:
    if not _os.path.exists(LOCK_FILE):
        return -1.0
    try:
        mtime = _os.path.getmtime(LOCK_FILE)
    except Exception:
        return -1.0
    return _time.time() - mtime


def check_rate_limit() -> bool:
    """Returns True if rate-limit lock is set AND still valid."""
    age = _lock_age()
    if age < 0:
        return False
    if age > LOCK_TTL_SEC:
        # Lock expired → assume quota recovered. Remove stale file.
        try:
            _os.remove(LOCK_FILE)
        except Exception:
            pass
        return False
    return True


def signal_rate_limit(reason: str = "429 from Gemini") -> None:
    """Set the rate-limit lock file. Idempotent."""
    try:
        with open(LOCK_FILE, "w") as f:
            f.write(f"gemini_quota_lock: {reason}  ts={int(_time.time())}\n")
            f.write(f"expires_in_seconds={LOCK_TTL_SEC}\n")
            f.write("Delete this file to force re-attempt:  rm .gemini_quota.lock\n")
    except Exception as e:
        _sys.stderr.write(f"[rate-limit-guard] failed to set lock: {e}\n")


def clear_rate_limit() -> None:
    """Manually remove the rate-limit lock."""
    try:
        _os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass
    except Exception as e:
        _sys.stderr.write(f"[rate-limit-guard] failed to clear lock: {e}\n")


def exit_if_rate_limited(context: str = "") -> None:
    """Convenience: print + sys.exit(0) if rate-limited."""
    if check_rate_limit():
        _sys.stderr.write(
            f"[rate-limit-guard] Gemini-Quote erschöpft — skipping {context}\n"
            f"[rate-limit-guard] Lock-Datei: {LOCK_FILE}\n"
            f"[rate-limit-guard] To retry later: clear_rate_limit()\n"
        )
        _sys.exit(0)
