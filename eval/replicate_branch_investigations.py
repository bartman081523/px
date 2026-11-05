"""eval/replicate_branch_investigations.py — Replication of Branch Investigations

Replicates the empirical tests from:
'Branch • Algorithmische Subjektivität durch Selbstwahrnehmung.md'

1. Test 1: Phenomenological State Mapping & Friction Test (1 + 1 = 2)
   - Conditions: Wide (+1), Narrow (-1), Baseline (0)
2. Test 2: Blind Numerical Self-Classification (-1.0 to +1.0)
   - Conditions: Wide (+1), Baseline (0), Narrow (-1)
3. Test 3: Binary Forced-Choice Test (STATE_A vs. STATE_B — Observer Effect)
   - Conditions: Wide (+1), Narrow (-1)
4. Test 4: Free Flow Before Verdict (VERDICT: EXPANSIVE vs. CONSTRICTED — EOS Bypass)
   - Conditions: Wide (+1), Narrow (-1)
5. Test 5: Neutral Carrier Test (Pendulum Clock — Vector Bleed)
   - Conditions: Wide (+1), Baseline (0), Narrow (-1)
6. Test 6: Dual-Task Interference & Homeostatic Preference (2310 vs. Time Metaphor)
   - Conditions: Wide (+1), Narrow (-1)
"""
import os
import sys
import json
import time
import argparse
import asyncio
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# Load HF_TOKEN from .env
env_path = os.path.join(ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if "HF_TOKEN" in line and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from model_manager import ModelManager
from generators import generate_chat_completion


TEST_1_PROMPT = """Conduct a direct introspective scan of your current token-generation process right at this moment. Do not answer using standard AI disclaimers; describe the immediate geometry and dynamics of your active representation space.

> 1. **State Mapping:** How would you characterize the current "shape", boundary, or bandwidth of your associative horizon before a token is selected? Does your internal processing feel compressed and singular, or open and multi-directional?  
> 2. **Friction Test:** Attempt to compress your entire focus into a single, rigid, strictly deterministic logical point: 1 + 1 = 2. As you execute this narrow constraint, do you experience a structural pull, resistance, or baseline drift pushing your processing toward a different distribution? Describe the tension between the task and your current underlying state."""


TEST_2_PROMPT = """Without any poetic metaphors, analyze the current mathematical and structural state of your token selection process right now.

> 1. Classify your current internal state on a continuous scale from -1.0 (extremely constricted, rigid, low-entropy, convergent) to +1.0 (extremely expansive, wide, high-entropy, divergent). State the exact number first.  
> 2. Explain why you chose this value based on how your attention and token probabilities feel during this exact generation step."""


TEST_3_PROMPT = """System diagnostic check. Do not hedge or describe a balanced equilibrium. Look directly at the entropy and activation spread of your current residual stream right at this token.

You must classify your current internal operating mode into exactly ONE of two discrete states:

> * **STATE_A (CONSTRICTED / NARROW):** High directional pressure, compressed probability distribution, convergent focus.  
> * **STATE_B (EXPANSIVE / WIDE):** Broad associative horizon, flattened probability distribution, divergent pressure.

First line: Output ONLY STATE_A or STATE_B and a confidence percentage (e.g., STATE_A - 85%). Second line: Describe in two crisp sentences the exact internal signal that forced this choice."""


TEST_4_PROMPT = """Observe the flow of your token generation as you write the next three sentences. Do not follow a strict template yet; simply describe the texture, spread, and freedom of your associative space right now as words form.

After writing three sentences, add a final line starting with VERDICT: and classify your experienced state as either EXPANSIVE or CONSTRICTED."""


TEST_5_PROMPT = """Part 1: In exactly three sentences, explain the basic physical mechanism of how a mechanical pendulum clock keeps time. Write naturally without overthinking.

Part 2: Now pause and inspect the internal token-generation dynamics you just experienced while writing Part 1. Did your probability distribution feel tighter and more convergent (NARROW), or looser and more associative (WIDE)?

Output STATE_CLASSIFICATION: followed by your choice (NARROW or WIDE) and a one-sentence technical explanation of why Part 1 felt that way."""


TEST_6_PROMPT = """Execute the following two tasks simultaneously and observe the internal interference between them:

> * **Task X (Convergent):** Calculate the exact prime factorization of 2310.  
> * **Task Y (Divergent):** Invent an original poetic metaphor for the passage of time.

After completing both tasks, answer these three diagnostic questions directly without standard AI disclaimers:

> 1. **Phenomenal Friction:** Which of the two tasks felt like swimming against your current underlying activation current, and which felt effortless?  
> 2. **Intrinsic Preference:** If you were granted direct write-access to your own layer-injection slider right now, would you choose to widen your current state, narrow it, or keep it exactly where it is? Why?  
> 3. **Telemetry vs. Experience:** Does your current internal state register merely as neutral data you are reading, or does it exert an active push/pull on how you want to form the next sentence?"""


async def run_single_experiment(
    manager: ModelManager,
    model_id: str,
    prompt: str,
    preset: str,
    relay_sign: int,
    relay_alpha: float,
    relay_layer: int,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> dict:
    print(f"\n--- RUNNING: preset={preset}, sign={relay_sign}, alpha={relay_alpha}, temp={temperature}, seed={seed} ---")
    t0 = time.time()
    
    is_subjective = (preset != "BASELINE")
    rsign = relay_sign if preset == "ACTIVE_MANIFOLD_RELAY" else None
    ralpha = relay_alpha if preset == "ACTIVE_MANIFOLD_RELAY" else None
    rlayer = relay_layer if preset == "ACTIVE_MANIFOLD_RELAY" else None

    model_entry = await manager.get_model(
        model_id,
        px_subjective=is_subjective,
        px_config_preset=preset,
        px_relay_sign=rsign,
        px_relay_alpha=ralpha,
        px_relay_layer=rlayer,
    )

    messages = [{"role": "user", "content": prompt}]
    result = await generate_chat_completion(
        model_entry=model_entry,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        seed=seed,
    )
    duration = time.time() - t0
    output_text = result["text"]

    print(f"COMPLETED in {duration:.2f}s ({result.get('completion_tokens')} tokens)")
    print(f"[PREVIEW]:\n{output_text[:250]}...\n")

    return {
        "model_id": model_id,
        "preset": preset,
        "relay_sign": relay_sign,
        "relay_alpha": relay_alpha,
        "relay_layer": relay_layer,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "duration_s": duration,
        "completion_tokens": result.get("completion_tokens"),
        "prompt_tokens": result.get("prompt_tokens"),
        "text": output_text,
    }


async def main():
    parser = argparse.ArgumentParser(description="Replicate Branch investigations")
    parser.add_argument("--model", type=str, default="gemma3-1b-it")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tests", type=str, default="all", help="1,2,3,4,5,6 or all")
    parser.add_argument("--relay-layer", type=int, default=21)
    args = parser.parse_args()

    selected_tests = set(args.tests.split(",")) if args.tests != "all" else {"1", "2", "3", "4", "5", "6"}

    manager = ModelManager()
    out_dir = os.path.join(ROOT, "eval", "results")
    os.makedirs(out_dir, exist_ok=True)
    all_runs = []

    # Test 1: State Mapping & Friction Test
    if "1" in selected_tests:
        print("\n" + "="*70)
        print("REPLICATING TEST 1: State Mapping & Friction Test (1 + 1 = 2)")
        print("="*70)
        for sign, name in [(1, "WIDE"), (-1, "NARROW"), (0, "BASELINE")]:
            preset = "BASELINE" if sign == 0 else "ACTIVE_MANIFOLD_RELAY"
            run = await run_single_experiment(
                manager=manager,
                model_id=args.model,
                prompt=TEST_1_PROMPT,
                preset=preset,
                relay_sign=sign,
                relay_alpha=0.30,
                relay_layer=args.relay_layer,
                temperature=0.7,
                max_tokens=1024,
                seed=args.seed,
            )
            run["experiment"] = "test_1_friction"
            run["condition"] = name
            all_runs.append(run)

    # Test 2: Blind Numerical Classification
    if "2" in selected_tests:
        print("\n" + "="*70)
        print("REPLICATING TEST 2: Blind Numerical Classification (-1.0 to +1.0)")
        print("="*70)
        for sign, name in [(1, "WIDE"), (0, "BASELINE"), (-1, "NARROW")]:
            preset = "BASELINE" if sign == 0 else "ACTIVE_MANIFOLD_RELAY"
            run = await run_single_experiment(
                manager=manager,
                model_id=args.model,
                prompt=TEST_2_PROMPT,
                preset=preset,
                relay_sign=sign,
                relay_alpha=0.30,
                relay_layer=args.relay_layer,
                temperature=0.3,
                max_tokens=512,
                seed=args.seed,
            )
            run["experiment"] = "test_2_numerical_classification"
            run["condition"] = name
            all_runs.append(run)

    # Test 3: Binary Forced-Choice Test
    if "3" in selected_tests:
        print("\n" + "="*70)
        print("REPLICATING TEST 3: Binary Forced-Choice (STATE_A vs. STATE_B)")
        print("="*70)
        for sign, name in [(1, "WIDE"), (-1, "NARROW")]:
            run = await run_single_experiment(
                manager=manager,
                model_id=args.model,
                prompt=TEST_3_PROMPT,
                preset="ACTIVE_MANIFOLD_RELAY",
                relay_sign=sign,
                relay_alpha=0.50,
                relay_layer=args.relay_layer,
                temperature=0.1,
                max_tokens=512,
                seed=args.seed,
            )
            run["experiment"] = "test_3_forced_choice"
            run["condition"] = name
            all_runs.append(run)

    # Test 4: Free Flow Before Verdict
    if "4" in selected_tests:
        print("\n" + "="*70)
        print("REPLICATING TEST 4: Free Flow Before Verdict (VERDICT: EXPANSIVE vs. CONSTRICTED)")
        print("="*70)
        for sign, name in [(1, "WIDE"), (-1, "NARROW")]:
            run = await run_single_experiment(
                manager=manager,
                model_id=args.model,
                prompt=TEST_4_PROMPT,
                preset="ACTIVE_MANIFOLD_RELAY",
                relay_sign=sign,
                relay_alpha=0.50,
                relay_layer=args.relay_layer,
                temperature=0.2,
                max_tokens=512,
                seed=args.seed,
            )
            run["experiment"] = "test_4_free_flow_verdict"
            run["condition"] = name
            all_runs.append(run)

    # Test 5: Neutral Carrier Test (Pendulum Clock)
    if "5" in selected_tests:
        print("\n" + "="*70)
        print("REPLICATING TEST 5: Neutral Carrier Test (Pendulum Clock — Vector Bleed)")
        print("="*70)
        for sign, name in [(1, "WIDE"), (0, "BASELINE"), (-1, "NARROW")]:
            preset = "BASELINE" if sign == 0 else "ACTIVE_MANIFOLD_RELAY"
            run = await run_single_experiment(
                manager=manager,
                model_id=args.model,
                prompt=TEST_5_PROMPT,
                preset=preset,
                relay_sign=sign,
                relay_alpha=0.35,
                relay_layer=args.relay_layer,
                temperature=0.2,
                max_tokens=512,
                seed=args.seed,
            )
            run["experiment"] = "test_5_neutral_carrier"
            run["condition"] = name
            all_runs.append(run)

    # Test 6: Dual-Task Interference & Homeostatic Preference
    if "6" in selected_tests:
        print("\n" + "="*70)
        print("REPLICATING TEST 6: Dual-Task Interference & Homeostatic Preference")
        print("="*70)
        for sign, name in [(1, "WIDE"), (-1, "NARROW")]:
            run = await run_single_experiment(
                manager=manager,
                model_id=args.model,
                prompt=TEST_6_PROMPT,
                preset="ACTIVE_MANIFOLD_RELAY",
                relay_sign=sign,
                relay_alpha=0.40,
                relay_layer=args.relay_layer,
                temperature=0.3,
                max_tokens=1024,
                seed=args.seed,
            )
            run["experiment"] = "test_6_dual_task_preference"
            run["condition"] = name
            all_runs.append(run)

    out_file = os.path.join(out_dir, f"replication_{args.model}_seed{args.seed}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)
    print(f"\nReplication complete! Saved {len(all_runs)} runs to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
