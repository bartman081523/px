# PX Mod Improvement Plan: MiniCPM5-1B (Llama Architecture)

## Context & Motivation
Initial comprehensive evaluations show that the PX patch on MiniCPM5-1B yields a modest improvement in logic/arithmetic capabilities (40.0% -> 41.67% overall). However, the ultra-hard benchmark shows mixed results (Base: 0.6, PX Peak: 0.4, PX Subj: 0.6). We need to empirically improve the patch across all cognitive zones, especially focusing on arithmetic, logic riddles, and HLE tasks, without introducing regressions.

## SciMind4 Rigor Protocol

### Core Hypotheses for Improvement
1.  **Recurrence Depth (H_loops)**:
    *   *Steelman Hypothesis*: MiniCPM5-1B has a different depth-to-reasoning ratio than Gemma3. The default 6 loops might be insufficient for complex arithmetic/logic on this specific architecture. Increasing loops to 8 or 10, or making them dynamically dependent on the cognitive zone, will improve performance.
    *   *Antithesis*: More loops only increase computational cost and lead to representation collapse (oversmoothing), decreasing performance.
2.  **Grounding Injection Strength (H_gamma)**:
    *   *Steelman Hypothesis*: Llama's RMSNorm behaves differently under the LTI/ADC injection. The current gamma of 0.06 is a linear interpolation. Tuning gamma specifically for the math/logic zones (e.g., higher gamma for rigid grounding) will improve arithmetic.
    *   *Antithesis*: Gamma tuning is a Texas Sharpshooter fallacy that overfits to the benchmark. A uniform gamma is architecturally soundest.
3.  **Zone Routing Boundaries (H_bounds)**:
    *   *Steelman Hypothesis*: The recurrent block `L9-L18` misses late-stage logical synthesis layers. Shifting the block to `L12-L20` or expanding it will capture more complex reasoning paths.
    *   *Antithesis*: Shifting the block arbitrarily disrupts the pretrained layer specializations and causes regressions in general language modeling (HLE).

## Experimental Roadmap

### Step 1: Establish Extended Baseline
Run the comprehensive benchmark (already done, Base: 40.0%, PX: 41.67%) and the ultra-hard benchmark (Base: 0.6, PX: 0.4).

### Step 2: Gamma & Loop Sweeps (Empirical Tuning)
We will systematically vary the core PX parameters in Peak mode (to isolate the core mechanism from subjective auto-tuning effects).
*   Test Gamma: `[0.04, 0.08, 0.12]`
*   Test Loops: `[4, 8, 10]`
Evaluate using the comprehensive suite.

### Step 3: Architecture-Specific Refinements (Llama)
*   Investigate the `create_causal_mask` behavior further.
*   Implement zone-specific `n_loops` and `gamma` in `auto_tune.py` (Subjective mode). Rigid zones (math) get more loops/higher gamma; fluid zones (creative) get fewer loops/lower gamma.

### Step 4: Cross-Architecture Evaluation (Gemma3 vs MiniCPM)
Run the exact same tests (Capability + Ultra Hard) on Gemma3-270M (Baseline, Peak, Subjective) to provide a comparative grounding. Does the Llama architecture inherently struggle with certain reasoning loops that Gemma3 handles well? 

### Step 5: Final Validation
Run the full comprehensive benchmark and ultra-hard benchmark. Compare the optimized PX patch against the unpatched baseline and the cross-architecture results.
