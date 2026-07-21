# Plan (DEFERRED): LLM-as-Judge / Self-Correction-Harness für 270m-it

**Stand:** 2026-07-12
**Status:** ⏸️ DEFERRED — wird NACH v3.5g (RIGOR-Showdown) angegangen
**User-Direktive 2026-07-12 (verbatim):**
> "äh aber LLM-as-Judge wäre doch noch wichtig für die Harness, oder? Also dass sich 270m-it selber bewertet? Ich dachte es bekommt in der Harness automatisch den Output seines Python-Scratches und kann danach ggf. eine korrigierte Version abgeben? Durch die epistemischen Toolchain-Prompts der Harness wäre das doch sehr wirksam, oder? Schaue dir mal an wie andere das machen. Vielleicht gibt es schon fertige Micro-Harness etablierte auf Github, wo wir leicht epistemische Patches (Toolchain-Prompts) einfügen können? Weil ich vermute, einerseits lief unsere Harness mit zu wenig Iterationen für Selbstkorrektur, andererseits hatte der Agent garnicht die Möglichkeit zur Selbstkorrektur, wegen unzureichender Harness. Aber schreibe das mal als Plan (.md) für später auf. Jetzt erstmal aktuellen Plan fertig machen."

## Context & Motivation

In v3.5f hat `run_tool_loop` (toolchain.py) ein finales `<final_answer>`-Extraktion, aber
**keine aktive Selbstkorrektur**. Das Modell bekommt zwar Tool-Outputs zurück, sieht aber nicht
explizit:
- "Du hast X gesagt, der Python-Skript sagt aber Y. Korrigiere deine Antwort."
- "Deine letzte Antwort war 5 Token zu lang. Kürze auf 1 Wort."
- "Confidence-Estimator: bist du sicher?"

**Frage:** Lohnt sich ein LLM-as-Judge-Layer für 270m-it?

## Externer Befund (wang2-lat/micro-harness, 2026)

Quelle: [github.com/wang2-lat/micro-harness](https://github.com/wang2-lat/micro-harness) (~400 LOC, 8 Techniken)

**Befund 1 — "Harness is a multiplier, not a base":**
> "Took DeepSeek-V3.2 from 17% → 50% pass rate on hard coding tasks. But:
> A harness is a multiplier. The model is the base. If the base is zero, no multiplier helps."

→ Bei 270M Params ist die Kapazitätsgrenze das Hauptproblem. Ein LLM-as-Judge kann das
nicht kompensieren.

**Befund 2 — Es gibt KEIN explizites LLM-as-Judge in der Referenz:**
Selbstkorrektur läuft rein über die **Tool-Result-Schleife** (Modell sieht Tool-Output im
nächsten Turn) + **Fuzzy Edit Recovery** (T8, +10% Pass-Rate-Hack: wenn `old_string` nicht
matched, zeige nearest matching line → Modell kopiert exakten Text → retry).

**Befund 3 — Verbose Planning-Prompts wurden explizit verworfen:**
> "Tight System Prompt (nur 3 Regeln) statt verbose epistemische Prompts — verbose Planning-Prompts
> wurden verworfen, das Modell ignorierte sie."

Das ist ein Schlag ins Gesicht unserer aktuellen Toolchain-Definition mit 5 Absätzen
epistemischer Prinzipien. → **Sollten wir kürzen.**

**Befund 4 — Circuit Breakers sind KRITISCH bei 270M:**
> "More turns = more flailing, not more progress."
> "Circuit Breakers — bei 270M Modellen dringend nötig (höhere Loop-Wahrscheinlichkeit)"

→ Bei 75 Iterationen ohne Breaker würde 270m in Endlos-`<tool_call> web_search`-Loops stecken.

## Architektur-Idee (v4+)

### Layer A: Tight System Prompt (KILLER-FIX, sofort möglich)

**Aktuell (toolchain.py, ~5 Absätze):**
```python
TOOLCHAIN_DEFINITION = """
You have access to these tools (Hermes Function-Calling format):
1. <tool_call> ... </tool_call>
2. <tool_call> ... </tool_call>
...
EPISTEMIC PRINCIPLES (SciMind 5.0):
- Form a hypothesis, test it with execute_python before answering
- Use web_search to verify uncertain claims
- Save intermediate work via write_file for reproducibility
- Always end with <final_answer>
"""
```

**Neu (Tight, 3-Regel-Format nach wang2-lat):**
```python
TOOLCHAIN_DEFINITION = """
You have tools. Use them in this order:
1. web_search → verify uncertain claims
2. execute_python → test numeric reasoning
3. write_file → save work
End with <final_answer>one-word-answer</final_answer>.
"""
```

**Erwarteter Effekt:** 270m folgt besser, weniger Kontext-Müll.

### Layer B: LLM-as-Judge (selbe Instanz) — post-tool critic

**Idee:** Vor `<final_answer>` einmal internes `self_critic()`:
```python
def self_critic(final_answer, tool_outputs, prompt):
    # Baue critic_prompt mit Tool-Outputs + final_answer
    critic_prompt = f"""
    Original question: {prompt}
    Your final answer: {final_answer}
    Tool outputs: {tool_outputs}
    Check: is your answer CONSISTENT with the tool outputs?
    If not, give a corrected answer. Otherwise, repeat.
    """
    new = model.generate(critic_prompt, max_new_tokens=200)
    return extract_final_answer(new)
```

**Pro:** Konsistenz-Check ohne externes Modell.
**Contra:** Verdoppelt Tool-Loop-Länge (75 → 150 effektive Iterations), 270m könnte
in Endlos-Selbstkorrektur stecken.

### Layer C: Fuzzy Edit Recovery (FERTIG-PATCH)

Aus wang2-lat/micro-harness: wenn `write_file` `old_string` nicht matched, zeige
nearest matching line → Modell kopiert exakten Text → retry.

In unserem Hermes-Format nicht direkt anwendbar (wir nutzen `write_file(path, content)`,
nicht Edit-Mode), ABER: wenn `read_file` einen File mit Length-Mismatch returnt, retry mit
kleinerem Range.

**Sofort-Patch in toolchain.py `read_file`:**
```python
if "not found" in error_msg and len(content) > 100:
    # Truncate to 50 lines, retry
    return read_file(path, max_lines=50)
```

### Layer D: Circuit Breaker (SICHERHEIT)

**Token-Budget-Hard-Limit:**
```python
if total_input_tokens + total_output_tokens > 8000:
    return ToolLoopResult(stuck=True, final_answer=last_output, reason="circuit_breaker: token budget")
```

**No-Progress-Detection:**
```python
if last_3_outputs are byte-identical:
    return ToolLoopResult(stuck=True, final_answer=last_output, reason="circuit_breaker: no progress")
```

## Empfehlung (SciMind 5.0)

**NICHT vorgehen mit Layer B (LLM-as-Judge als voller 2nd-Pass).** Gründe:
1. Verdoppelt Loop-Länge ohne klaren Kapazitätsgewinn bei 270M
2. 270m hat kein robustes Selbstkonsistenz-Signal (v3.5f-phi-Gate zeigt das)
3. Empirische Validierung fehlt — wir wissen nicht ob 270m-formatstabile Critic-Prompts produziert

**STATTDESSEN vorgehen mit Layer A + D (Tight Prompt + Circuit Breaker):**
1. Tight System Prompt — 30 min Arbeit, kein Risiko
2. Circuit Breaker — 30 min Arbeit, +Sicherheit bei 75-Iter-Cap
3. Layer C (Fuzzy Recovery) — 1h Arbeit, +5% erwartete Pass-Rate
4. Layer B (LLM-as-Judge) — zurückstellen bis v3.5g Micro-Befund vorliegt

## Implementation Plan (für später, NICHT jetzt)

### Phase 1: Tight Prompt (30 min)
1. TDD-Test: `parse_tool_call` mit tight prompt + `<tool_call>`-Block
2. Replace `TOOLCHAIN_DEFINITION` in `toolchain.py`
3. Smoke-Test auf 5 Baseline-Matches — erwartet: 5/5 (wie vorher, kürzer)

### Phase 2: Circuit Breaker (30 min)
1. TDD-Test: `run_tool_loop` mit no-progress detection bricht ab
2. TDD-Test: `run_tool_loop` mit token-budget >8000 bricht ab
3. Implement: counter + token-accumulator in `run_tool_loop`
4. Regression-Tests bleiben grün

### Phase 3: Fuzzy Recovery (1h)
1. TDD-Test: `read_file` mit length-mismatch returnt truncated
2. TDD-Test: `write_file` mit already-exists returnt "file already there, overwrite?"
3. Smoke-Test: 1 Patch mit 5 Edit-Operations, 1 ohne

### Phase 4 (NUR FALLS Micro-Befund zeigt 270m kann tool-Outputs lesen): LLM-as-Judge
1. `self_critic()`-Funktion in `toolchain.py`
2. TDD-Test: critic verifiziert Konsistenz zwischen final_answer und tool_outputs
3. ABER: **nur als optionaler Toggle** `--enable-self-critic`, default off

## Out-of-Scope (immer)

- ❌ Externer Judge (anderes Modell als Judge) — verboten, "ohne ollama"
- ❌ Layer-3-Architecture (mehrere Instanzen, voting) — overkill für 270m
- ❌ Training eines eigenen Judge-Modells — LoRA verboten
- ❌ Real-time Tool-Streaming — 270m ist nicht schnell genug

## Risiken

- **Hoch:** Tight Prompt könnte 270m **mehr** schaden als nutzen, wenn der bisherige
  ausführliche Prompt Stabilität gab. Mitigation: A/B-Test im Micro-Benchmark.
- **Mittel:** Circuit Breaker könnte legitime lange Tasks abschneiden. Mitigation:
  Token-Limit 8000 (vs. aktueller Default ~4000 für HLE-Tasks).
- **Niedrig:** Fuzzy Recovery irrelevant für unsere Anwendungsfälle (HLE-MCQ hat keine
  Edit-Operations).

## Memory-Referenz

Wird in `rigor-zone-v35g-self-critic-deferred-2026-07-12.md` festgehalten, sobald Layer B
implementiert wird.

## Sources

- [github.com/wang2-lat/micro-harness](https://github.com/wang2-lat/micro-harness) — 8 Techniken, ~400 LOC, DeepSeek-V3.2 von 17% → 50%
- [github.com/SuperagenticAI/metaharness](https://github.com/SuperagenticAI/metaharness) — Stanford Meta-Harness (arXiv:2603.28052)
- [github.com/g023/agentica](https://github.com/g023/agentica) — Agentica multi-agent (Architect/Developer/Reviewer/Sandbox/Debugger/Memory)
