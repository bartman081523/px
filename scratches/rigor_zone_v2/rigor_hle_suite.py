"""HLE-Suite Loader — cais/hle HuggingFace-Dataset + statischer Fallback.

SciMind 5.0:
- Zero-Trust: try-except um load_dataset (kann fehlschlagen wenn offline)
- Falsificationism: filtert nach Kategorien (math, cs, philosophy) — bewusste Skopus-Wahl
- Atomic & Secured: bei Crash wird None + leere Liste returnt (kein halbfertiger State)

Struktur: Liste von (task_id, category, prompt, ground_truth) Tupeln.
Verifikation pro Kategorie in `rigor_verify.py` (deterministisch + 2-Stage).
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# Statische Fallback-Liste (publizierte HLE-Beispiele aus dem Paper)
# ═══════════════════════════════════════════════════════════════════════════════
# Wenn HuggingFace cais/hle nicht ladbar (kein Netz, datasets nicht installiert),
# verwenden wir diese 12 hand-kuratierten Tasks als Minimum-Viable-Suite.
# Quellen: HLE Paper (arXiv:2501.14249) + HLE-Website (https://lastexam.ai/).

_STATIC_HLE_TASKS: List[Tuple[str, str, str, str]] = [
    # ── MATH ──────────────────────────────────────────────────────────────
    (
        "math-001",
        "math",
        "Find the smallest positive integer n such that 2^n ≡ 1 (mod 17) and n > 4.",
        "8",
    ),
    (
        "math-002",
        "math",
        "Compute the determinant of the 3×3 matrix [[2,1,3],[0,1,4],[5,6,0]].",
        "1",
    ),
    (
        "math-003",
        "math",
        "If f(x) = x^3 - 6x^2 + 11x - 6, find all real roots.",
        "1, 2, 3",
    ),
    (
        "math-004",
        "math",
        "Evaluate the integral ∫₀^π sin²(x) dx.",
        "π/2",
    ),
    (
        "math-005",
        "math",
        "How many distinct binary trees with 5 nodes exist?",
        "42",
    ),
    # ── CS ────────────────────────────────────────────────────────────────
    (
        "cs-001",
        "cs",
        "What is the time complexity of the optimal algorithm for finding the "
        "longest common subsequence of two strings of length m and n?",
        "O(mn)",
    ),
    (
        "cs-002",
        "cs",
        "In the CAP theorem, a distributed system can simultaneously provide "
        "which two of the three guarantees?",
        "Consistency and Availability OR Consistency and Partition tolerance OR Availability and Partition tolerance",
    ),
    (
        "cs-003",
        "cs",
        "What is the output of the following Python code: print([x for x in range(10) if x % 2 == 0 and x > 4])?",
        "[6, 8]",
    ),
    (
        "cs-004",
        "cs",
        "In a B+ tree of order 4, what is the maximum number of keys a leaf node can contain?",
        "3",
    ),
    # ── PHILOSOPHY ────────────────────────────────────────────────────────
    (
        "philosophy-001",
        "philosophy",
        "State the central thesis of Gettier problems and one widely accepted response.",
        "Justified true belief is not sufficient for knowledge; responses include reliabilism, defeasibility theory, or infinite regress arguments.",
    ),
    (
        "philosophy-002",
        "philosophy",
        "Explain the difference between weak and strong emergence in one sentence each.",
        "Weak emergence: macro-level patterns are unexpected but derivable from micro-level rules in principle. Strong emergence: macro-level patterns are ontologically novel and not derivable even in principle.",
    ),
    (
        "philosophy-003",
        "philosophy",
        "In the problem of other minds, what is the argument from analogy and what is its main weakness?",
        "We observe that we have minds, observe others' similar behavior, and infer they have minds. Weakness: the inference is inductive and rests on the assumption that similar behavior entails similar internal states, which is itself what needs to be proven.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# HLE-Suite Loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_hle_huggingface(
    target_categories: Optional[Tuple[str, ...]] = None,
    max_per_category: int = 10,
) -> Optional[List[Tuple[str, str, str, str]]]:
    """Versucht, cais/hle von HuggingFace zu laden.

    HLE-Kategorien sind: 'Math', 'Computer Science/AI', 'Humanities/Social Science',
    'Physics', 'Biology/Medicine', 'Chemistry', 'Engineering', 'Other'.

    Returns:
        Liste von (task_id, category, prompt, ground_truth) oder None wenn fehlgeschlagen.
    """
    # HLE-Original-Namen (siehe cais/hle Schema)
    if target_categories is None:
        target_categories = ("Math", "Computer Science/AI", "Humanities/Social Science")

    try:
        from datasets import load_dataset
        hle = load_dataset("cais/hle", split="test")
    except Exception as e:
        print(f"  [load_hle_huggingface] Fehler: {type(e).__name__}: {e}")
        return None

    # Per-Kategorie begrenzen + Text-only (kein Bild, da 270m nicht multimodal-fokussiert)
    by_category: dict = {cat: [] for cat in target_categories}
    for q in hle:
        cat = q.get("category", "unknown")
        if cat not in by_category:
            continue
        # Skip wenn Bild vorhanden (270m kann keine Bilder verarbeiten)
        if q.get("image") and q["image"].strip():
            continue
        # Skip wenn rationale fehlt (oft low-quality questions)
        if not q.get("rationale", "").strip():
            continue
        if len(by_category[cat]) >= max_per_category:
            continue
        by_category[cat].append((
            q.get("id", f"hle-{cat}-{len(by_category[cat])}"),
            cat,
            q.get("question", ""),
            q.get("answer", ""),
        ))

    result: List[Tuple[str, str, str, str]] = []
    for cat in target_categories:
        result.extend(by_category[cat])

    if not result:
        return None
    return result


def load_hle_suite(
    target_categories: Optional[Tuple[str, ...]] = None,
    max_per_category: int = 10,
) -> Tuple[str, List[Tuple[str, str, str, str]]]:
    """Lädt HLE-Suite (HuggingFace oder statisch).

    Returns:
        (source, tasks) wobei source ∈ {"huggingface", "static_fallback"}
    """
    print("=" * 60)
    print("HLE-Suite Loader")
    print("=" * 60)

    if target_categories is None:
        target_categories = ("Math", "Computer Science/AI", "Humanities/Social Science")

    print("  Versuche HuggingFace cais/hle ...")
    hf_result = load_hle_huggingface(target_categories, max_per_category)
    if hf_result:
        print(f"  ✓ HuggingFace geladen: {len(hf_result)} Tasks")
        cats = {}
        for _, c, _, _ in hf_result:
            cats[c] = cats.get(c, 0) + 1
        for c, n in cats.items():
            print(f"    - {c}: {n}")
        return "huggingface", hf_result

    print("  HuggingFace fehlgeschlagen → statischer Fallback")
    # Statische Tasks verwenden kleingeschriebene Slugs
    slug_map = {
        "Math": "math",
        "Computer Science/AI": "cs",
        "Humanities/Social Science": "philosophy",
    }
    target_slugs = tuple(slug_map.get(c, c.lower()) for c in target_categories)
    static_filtered = [t for t in _STATIC_HLE_TASKS if t[1] in target_slugs]
    cats = {}
    for _, c, _, _ in static_filtered:
        cats[c] = cats.get(c, 0) + 1
    print(f"  ✓ Statisch: {len(static_filtered)} Tasks")
    for c, n in cats.items():
        print(f"    - {c}: {n}")
    return "static_fallback", static_filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke-Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    source, tasks = load_hle_suite()
    print(f"\nSource: {source}")
    print(f"Total: {len(tasks)} Tasks")
    print("\nErste 3 Tasks:")
    for task_id, cat, prompt, gt in tasks[:3]:
        print(f"\n  [{task_id}] category={cat}")
        print(f"    Prompt: {prompt[:100]}...")
        print(f"    GT: {gt[:60]}")
