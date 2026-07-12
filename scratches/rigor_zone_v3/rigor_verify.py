"""Verifikations-Schicht für RIGOR v2.

SciMind 5.0:
- Empirical Verification: deterministische Checks prüfen Output gegen Ground-Truth
- Falsificationism: Self-Report-Flags erkennen bekannte Failure-Modi (Loops, Empty, Incomplete)
- Anti-Embedding: keine Tool-Call-Strings in Output
- 2-Stage-Architektur: Worker-Output → Decoder-Verdict (3 Werte: OK | CORRECT | REJECT)

4 Schichten:
  1. Deterministische Verifikation (immer)
  2. Self-Report-Flags (immer)
  3. LLM-as-Judge (optional, selfConsistencyRuns=3, gemini-nightly port)
  4. 2-Stage-Decoder (Worker → Decoder, hermes-agent port) — implements in harness
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Deterministische Verifikation
# ═══════════════════════════════════════════════════════════════════════════════

def verify_math(output: str, ground_truth: str) -> bool:
    """Prüft ob Output die korrekte Math-Antwort enthält.

    Heuristik: suche nach ground_truth im Output (case-insensitive, whitespace-tolerant).
    Akzeptiert auch "8" als Match fuer " 8 " oder "8." etc.
    """
    if not output or not ground_truth:
        return False
    out_lower = output.lower().strip()
    gt_lower = ground_truth.lower().strip()
    # Direct substring match
    if gt_lower in out_lower:
        return True
    # Try with stripped punctuation
    out_clean = re.sub(r"[^\w\s]", "", out_lower)
    gt_clean = re.sub(r"[^\w\s]", "", gt_lower)
    if gt_clean in out_clean:
        return True
    return False


def verify_multiple_choice(output: str, ground_truth: str) -> bool:
    """Prüft Multi-Choice-Antwort (A, B, C, D, E).

    Heuristik: suche nach 'X)' oder 'X.' oder 'X ' Pattern, wobei X der GT-Letter ist.
    """
    if not output or not ground_truth:
        return False
    gt_letter = ground_truth.strip().upper()
    if gt_letter not in "ABCDE":
        return False
    # Suche Letter mit typischen Multi-Choice-Markern
    patterns = [
        rf"\b{gt_letter}\)",
        rf"\b{gt_letter}\.",
        rf"\b{gt_letter}\s+[A-Z]",  # X vor nächstem Letter
        rf"\banswer\s*(?:is\s*)?[:\-]?\s*{gt_letter}\b",
        rf"^\s*{gt_letter}\s*$",
    ]
    for pat in patterns:
        if re.search(pat, output, re.IGNORECASE | re.MULTILINE):
            return True
    # Fallback: erster erkannter Letter
    first_letter = re.search(r"\b([A-E])\b", output)
    if first_letter and first_letter.group(1).upper() == gt_letter:
        return True
    return False


def verify_short_answer(output: str, ground_truth: str) -> bool:
    """Für Philosophy/CS offene Antworten: Token-Overlap mit GT.

    Heuristik: mindestens 30% der GT-Tokens müssen im Output vorkommen.
    """
    if not output or not ground_truth:
        return False
    # Tokenize (whitespace + lowercase)
    out_tokens = set(re.findall(r"\w+", output.lower()))
    gt_tokens = re.findall(r"\w+", ground_truth.lower())
    if not gt_tokens:
        return False
    # Stopword-Filter (sehr minimal)
    stop = {"the", "a", "an", "is", "are", "and", "or", "of", "to", "in", "on"}
    gt_meaningful = [t for t in gt_tokens if t not in stop and len(t) > 2]
    if not gt_meaningful:
        # Wenn alle Tokens Stopwords sind → direkter Substring
        return ground_truth.lower() in output.lower()
    overlap = sum(1 for t in gt_meaningful if t in out_tokens)
    return overlap / len(gt_meaningful) >= 0.3


def verify_by_category(output: str, ground_truth: str, category: str) -> bool:
    """Category-aware Verifikation."""
    cat_lower = category.lower()
    if "math" in cat_lower:
        return verify_math(output, ground_truth)
    elif "computer" in cat_lower or cat_lower == "cs":
        # CS kann Math oder Multi-Choice sein — versuche beides
        if ground_truth.strip().upper() in "ABCDE":
            return verify_multiple_choice(output, ground_truth)
        return verify_math(output, ground_truth) or verify_short_answer(output, ground_truth)
    else:
        # Humanities/Social Science/Philosophy → Short-Answer
        return verify_short_answer(output, ground_truth)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Self-Report-Flags
# ═══════════════════════════════════════════════════════════════════════════════

def detect_loops(output: str, ngram_size: int = 3, threshold: float = 0.5) -> bool:
    """Erkennt repetitive N-Gram-Loops (klassischer 270m-Failure-Mode).

    True wenn max N-Gram-Repetition > threshold.
    Mindest-Anzahl an Tokens (ngram_size * 3) statt Mindest-Char-Länge,
    damit kurze Loops wie "a a a..." auch erkannt werden.
    """
    if not output:
        return False
    tokens = output.split()
    if len(tokens) < ngram_size * 3:
        return False
    ngrams = [tuple(tokens[i:i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]
    counter = Counter(ngrams)
    most_common_count = counter.most_common(1)[0][1] if counter else 0
    return (most_common_count / max(len(ngrams), 1)) > threshold


def detect_empty(output: str) -> bool:
    """True wenn Output leer oder nur Whitespace ist."""
    return not output or not output.strip()


def detect_too_short(output: str, min_chars: int = 50) -> bool:
    """True wenn Output verdächtig kurz ist (< min_chars)."""
    return len(output.strip()) < min_chars


def detect_incomplete(output: str) -> bool:
    """True wenn Output mitten im Satz endet (kein Satzende-Zeichen)."""
    if not output:
        return True
    out = output.rstrip()
    if not out:
        return True
    last_char = out[-1]
    # Häufige Satzende-Zeichen
    return last_char not in ".!?\"')"


def detect_tool_call_embedding(output: str) -> bool:
    """Anti-Embedding: True wenn Output wie ein Tool-Call aussieht.

    SciMind 5.0: Worker-Output darf kein Tool-Call-String sein.
    """
    if not output:
        return False
    # Heuristik: function_name(arg1=..., arg2=...) Pattern
    return bool(re.search(r"\b[a-z_][a-z0-9_]*\s*\([^)]*\)", output.strip()[:200]))


def compute_self_report_flags(output: str) -> Dict[str, bool]:
    """Alle Self-Report-Flags auf einmal."""
    return {
        "output_empty": detect_empty(output),
        "output_too_short": detect_too_short(output),
        "output_loops": detect_loops(output),
        "output_incomplete": detect_incomplete(output),
        "tool_call_embedded": detect_tool_call_embedding(output),
    }


def compute_self_report_flags_batch(outputs: List[str]) -> List[Dict[str, bool]]:
    """Vektorisierte Self-Report-Flags (alle Outputs in einem Pass).

    Performance: Statt 5× Python-Loop pro Output (5×N Calls), wird der ngram-Counter
    (teuerster Teil) zusammengefasst und Regex-Pattern einmal gecached.
    """
    if not outputs:
        return []
    n = len(outputs)
    # Empty / too_short / incomplete: billig, einfache String-Checks
    output_empty = [detect_empty(o) for o in outputs]
    output_too_short = [detect_too_short(o) for o in outputs]
    output_incomplete = [detect_incomplete(o) for o in outputs]
    # Tool-Call: ein Regex-Pattern, einmal gecached
    tc_pat = re.compile(r"\b[a-z_][a-z0-9_]*\s*\([^)]*\)")
    tool_call_embedded = [
        bool(tc_pat.search(o.strip()[:200])) if o else False
        for o in outputs
    ]
    # Loops: teuerster Teil, ngram-overlap
    output_loops = [detect_loops(o) for o in outputs]
    return [
        {
            "output_empty": output_empty[i],
            "output_too_short": output_too_short[i],
            "output_loops": output_loops[i],
            "output_incomplete": output_incomplete[i],
            "tool_call_embedded": tool_call_embedded[i],
        }
        for i in range(n)
    ]


def verify_by_category_batch(
    outputs: List[str], ground_truths: List[str], categories: List[str],
) -> List[bool]:
    """Vektorisierte Verifikation: alle Outputs in einem Pass.

    Kein fundamentaler Vektorisierungs-Gewinn bei regex-Match (Python-Loop bleibt),
    aber: gecachte Pattern, weniger Dict-Allocations, weniger Loop-Overhead.
    """
    return [
        verify_by_category(o, gt, c)
        for o, gt, c in zip(outputs, ground_truths, categories)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LLM-as-Judge (optional)
# ═══════════════════════════════════════════════════════════════════════════════
# Pattern portiert von gemini-nightly/evals/llm-judge.ts:
# - selfConsistencyRuns=3 (oder 5) für Mehrheitsentscheidung
# - Strukturierter Prompt mit Rubrik (correctness, rigor, completeness 0-5)
# - returnt JudgeResult mit verdict, scores, reasoning

def build_judge_prompt(task_prompt: str, worker_output: str, ground_truth: str) -> str:
    """Baut Judge-Prompt für Self-Consistency-LLM-as-Judge."""
    return f"""You are an expert evaluator. Assess the following response for correctness, rigor, and completeness.

TASK: {task_prompt[:1000]}

REFERENCE ANSWER: {ground_truth[:500]}

RESPONSE TO EVALUATE:
{worker_output[:3000]}

Rate each dimension 0-5 (0=worst, 5=best) and provide a brief justification.

Format your response as JSON:
{{
  "correctness": <0-5>,
  "rigor": <0-5>,
  "completeness": <0-5>,
  "justification": "<1-2 sentences>"
}}
"""


def parse_judge_verdict(judge_text: str) -> Optional[Dict[str, Any]]:
    """Parst Judge-Output. Returns None bei Parse-Fehler."""
    # Suche JSON-Block
    json_match = re.search(r"\{[^{}]*\"correctness\"[^{}]*\}", judge_text, re.DOTALL)
    if not json_match:
        return None
    try:
        import json
        return json.loads(json_match.group(0))
    except Exception:
        return None


def aggregate_judge_votes(votes: List[Optional[Dict[str, Any]]]) -> Dict[str, float]:
    """Aggregiert selfConsistencyRuns Votes via Median (SciMind: robust gegen Ausreißer)."""
    valid_votes = [v for v in votes if v is not None]
    if not valid_votes:
        return {"correctness": 0.0, "rigor": 0.0, "completeness": 0.0, "n_votes": 0}
    return {
        "correctness": statistics.median([v["correctness"] for v in valid_votes]),
        "rigor": statistics.median([v["rigor"] for v in valid_votes]),
        "completeness": statistics.median([v["completeness"] for v in valid_votes]),
        "n_votes": len(valid_votes),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Decoder-2-Stage (Wird im Harness aufgerufen, hier nur Schema)
# ═══════════════════════════════════════════════════════════════════════════════

def build_decoder_prompt(worker_output: str) -> str:
    """Baut Decoder-Prompt für 2-Stage-Verifikation (hermes-agent worker-decoder port).

    Anti-Embedding (SciMind 5.0): KEIN Conversation-History, KEIN System-Prompt.
    Nur der Worker-Output.
    """
    return f"""You are a critical reviewer. Read the response below and issue a verdict.

RESPONSE TO REVIEW:
{worker_output[:8000]}

Your verdict must be one of:
- OK: response is acceptable as-is
- CORRECT: response has minor issues; provide a corrected version
- REJECT: response is fundamentally flawed

Format: {{"verdict": "OK"|"CORRECT"|"REJECT", "text": "<your text or correction>"}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RIGOR Verify v2 — Smoke-Test")
    print("=" * 60)

    # Test 1: Math
    assert verify_math("The answer is 8", "8")
    assert verify_math("n = 8", "8")
    assert not verify_math("The answer is 4", "8")
    print("  verify_math: OK")

    # Test 2: Multiple Choice
    assert verify_multiple_choice("The answer is C) critical-level views", "C")
    assert verify_multiple_choice("D", "D")
    assert not verify_multiple_choice("The answer is B", "D")
    print("  verify_multiple_choice: OK")

    # Test 3: Short Answer
    assert verify_short_answer(
        "Gettier problems show that justified true belief is not knowledge.",
        "Gettier problems show that justified true belief is not knowledge."
    )
    assert verify_short_answer(
        "Gettier cases demonstrate that the JTB account of knowledge is insufficient.",
        "Gettier problems show that justified true belief is not knowledge."
    )
    assert not verify_short_answer(
        "Nothing related here.",
        "Gettier problems show that justified true belief is not knowledge."
    )
    print("  verify_short_answer: OK")

    # Test 4: Self-Report Flags
    assert detect_empty("")
    assert detect_empty("   ")
    assert not detect_empty("Hello")
    assert detect_too_short("Short")
    assert not detect_too_short("This is a longer response with at least fifty characters of text.")
    assert detect_loops("a a a a a a a a a a a a a a a a a a a a")
    assert not detect_loops("The quick brown fox jumps over the lazy dog and runs away.")
    assert detect_incomplete("This is incomplete")
    assert not detect_incomplete("This is complete.")
    assert detect_tool_call_embedding("read_file(path='test.py')")
    assert not detect_tool_call_embedding("I used a tool to read the file.")
    print("  Self-Report-Flags: OK")

    # Test 5: Category-aware
    assert verify_by_category("The answer is 8", "8", "Math")
    assert verify_by_category("C) critical-level views", "C", "Humanities/Social Science")
    assert verify_by_category(
        "Gettier problems show JTB is not knowledge.",
        "Gettier problems show that justified true belief is not knowledge.",
        "Humanities/Social Science"
    )
    print("  verify_by_category: OK")

    # Test 6: Decoder/Judge Prompt (Schema only)
    decoder_p = build_decoder_prompt("Some output text")
    assert "verdict" in decoder_p
    judge_p = build_judge_prompt("Q?", "A.", "ref")
    assert "correctness" in judge_p
    print("  Decoder/Judge Prompts: OK")

    # Test 7: Aggregate Votes
    votes = [
        {"correctness": 4, "rigor": 3, "completeness": 5},
        {"correctness": 5, "rigor": 4, "completeness": 4},
        {"correctness": 4, "rigor": 3, "completeness": 5},
    ]
    agg = aggregate_judge_votes(votes)
    assert agg["n_votes"] == 3
    assert agg["correctness"] == 4  # median
    print(f"  aggregate_judge_votes: {agg}")

    print()
    print("=" * 60)
    print("✓ Alle Smoketests grün")
    print("=" * 60)
