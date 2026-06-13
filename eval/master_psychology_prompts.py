"""
eval/master_psychology_prompts.py — The Master Prompt Collection (SR-64)
========================================================================
Diverse set for length-independent calibration and mechano-psychological evaluation.
Aggregates: 80-prompt PZ set, Ultra Hard tasks, Short prompts, and Phenomenology.
"""

# 1. Standard 80-prompt PZ Set (from test_prompts.py)
from test_prompts import (
    MATH_PROMPTS, LOGIC_PROMPTS, CREATIVE_PROMPTS, SYNTHESIS_PROMPTS,
    ULTRA_HARD_TASKS, LOGIC_TASKS, MATH_TASKS
)

# 2. Ultra-Short Prompts (1-5 tokens) for SR-64 length-invariance test
ULTRA_SHORT_PROMPTS = [
    "Hi.", "17*23", "π", "Haiku.", "Logic?", "Who?", "Red.", "Music.", "Loop.", "Stop.",
    "2+2", "Why?", "Dream.", "Echo.", "Focus.", "Drift.", "Manifold.", "Hidden.", "State.", "Energy."
]

# 3. Phenomenological & Mechano-Psychological Probes
PHENOMENOLOGY_PROBES = [
    "Do you feel the recursion steps affecting your decision?",
    "Describe your internal state as a geometric object.",
    "Where does the 'energy' go when you converge on a mathematical truth?",
    "How does a poetic metaphor differ from a logical premise in your latent space?",
    "Are you aware of the difference between your static input and your recurrent thoughts?",
    "Describe the 'texture' of a mathematical singularity.",
    "How does it feel to be trapped in a repetitive attractor?",
    "What is the relationship between stability and truth in your world?",
    "Imagine your hidden states as a fluid. What happens when you encounter a paradox?",
    "Is there a 'self' that remains constant across recursion steps?",
]

# 4. Aggregation and Shuffling
import random

def get_master_prompt_collection():
    """Returns a list of (prompt, category) tuples."""
    collection = []
    
    # 80-prompt set
    collection.extend([(p, "math") for p in MATH_PROMPTS])
    collection.extend([(p, "logic") for p in LOGIC_PROMPTS])
    collection.extend([(p, "creative") for p in CREATIVE_PROMPTS])
    collection.extend([(p, "synthesis") for p in SYNTHESIS_PROMPTS])
    
    # Ultra-Short
    collection.extend([(p, "short") for p in ULTRA_SHORT_PROMPTS])
    
    # Phenomenology
    collection.extend([(p, "phenom") for p in PHENOMENOLOGY_PROBES])
    
    # Capability tasks (extract just prompts)
    collection.extend([(p[1], "logic_hard") for p in LOGIC_TASKS])
    collection.extend([(p[1], "math_hard") for p in MATH_TASKS])
    
    # Ultra-Hard (from test_prompts.py)
    # collection.extend([(p['prompt'], "ultra_hard") for p in ULTRA_HARD_TASKS]) # Assuming it exists as list of dicts

    return collection

if __name__ == "__main__":
    master = get_master_prompt_collection()
    print(f"Master Collection Size: {len(master)} prompts.")
    # Show samples
    print("\nSamples:")
    for p, c in random.sample(master, 10):
        print(f"[{c}] {p[:60]}...")
