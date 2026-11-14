#!/usr/bin/env python3
"""bench/capability_bench.py — Messharness für die Gemma-4-PX-Fähigkeitskampagne.

Benutzt ausschließlich die eigene OpenAI-kompatible API (server.py, /v1) —
kein direkter Modellzugriff, damit jede Zahl dem entspricht, was ein Client
(opencode, LM Studio, ai-sdk) tatsächlich sieht.

Subkommandos
------------
  coding   Pass@1 auf einem verifizierbaren Coding-Set, pro Konfiguration
  speed    TTFT / tok/s / Gesamtlatenz, pro Konfiguration und Kontextlänge
  context  Needle-in-a-Haystack, Recall vs. Kontextlänge
  size     Gewichts-Bytes bf16 vs. GF5 vs. GF3 auf echten Modelltensoren

Konfigurationen sind px_config_preset (+ optional Relay), also genau die
Hebel, die der Server ohnehin anbietet. Nichts wird im Modell verändert.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback
import urllib.error
import urllib.request

API = os.environ.get("PX_API", "http://127.0.0.1:7860/v1")
MODEL = os.environ.get("PX_MODEL", "gemma4-e2b-it")


# ─────────────────────────── API-Client ───────────────────────────
NO_THINK = os.environ.get("PX_NO_THINK", "0") == "1"


def chat(messages, *, model=MODEL, max_tokens=400, temperature=0.0,
         preset=None, relay_sign=None, relay_alpha=None, timeout=600, stream=False):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if preset:
        body["px_config_preset"] = preset
    if relay_sign is not None:
        body["px_relay_sign"] = relay_sign
    if relay_alpha is not None:
        body["px_relay_alpha"] = relay_alpha
    if NO_THINK:
        # llama.cpp/GGUF-Pfad: Denkkanal abschalten, damit das Token-Budget
        # der Antwort und nicht dem Reasoning gilt.
        body["chat_template_kwargs"] = {"enable_thinking": False}

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}/chat/completions", data=data,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    dt = time.time() - t0
    payload = json.loads(raw)
    msg = payload["choices"][0]["message"]
    usage = payload.get("usage", {})
    return {
        "text": msg.get("content") or "",
        "wall_s": dt,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": payload["choices"][0].get("finish_reason"),
    }


# ─────────────────────────── Coding-Set ───────────────────────────
CODING_TASKS = [
    ("add",
     "Write a Python function `add(a, b)` that returns the sum of two numbers.",
     ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0.5, 0.5) == 1.0"]),
    ("is_even",
     "Write a Python function `is_even(n)` that returns True if n is even.",
     ["assert is_even(2) is True", "assert is_even(3) is False", "assert is_even(0) is True"]),
    ("fizzbuzz",
     "Write a Python function `fizzbuzz(n)` that returns a list of strings for 1..n: "
     "'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, else the number as string.",
     ["assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']",
      "assert fizzbuzz(15)[-1] == 'FizzBuzz'",
      "assert fizzbuzz(3) == ['1','2','Fizz']"]),
    ("reverse_string",
     "Write a Python function `reverse_string(s)` that returns the string reversed.",
     ["assert reverse_string('abc') == 'cba'", "assert reverse_string('') == ''",
      "assert reverse_string('a') == 'a'"]),
    ("gcd",
     "Write a Python function `gcd(a, b)` that returns the greatest common divisor of two positive integers.",
     ["assert gcd(12, 18) == 6", "assert gcd(7, 13) == 1", "assert gcd(100, 10) == 10"]),
    ("is_prime",
     "Write a Python function `is_prime(n)` that returns True if n is a prime number.",
     ["assert is_prime(2) is True", "assert is_prime(17) is True",
      "assert is_prime(1) is False", "assert is_prime(9) is False"]),
    ("count_vowels",
     "Write a Python function `count_vowels(s)` that counts the vowels (aeiou) in a string, case-insensitive.",
     ["assert count_vowels('hello') == 2", "assert count_vowels('AEIOU') == 5",
      "assert count_vowels('xyz') == 0"]),
    ("fibonacci",
     "Write a Python function `fibonacci(n)` that returns a list of the first n Fibonacci numbers starting 0, 1.",
     ["assert fibonacci(5) == [0,1,1,2,3]", "assert fibonacci(1) == [0]",
      "assert fibonacci(2) == [0,1]"]),
    ("binary_search",
     "Write a Python function `binary_search(arr, target)` that returns the index of target in a sorted list, or -1.",
     ["assert binary_search([1,3,5,7], 5) == 2", "assert binary_search([1,3,5,7], 2) == -1",
      "assert binary_search([], 1) == -1"]),
    ("flatten",
     "Write a Python function `flatten(nested)` that flattens arbitrarily nested lists into one list.",
     ["assert flatten([1,[2,[3,4]],5]) == [1,2,3,4,5]", "assert flatten([]) == []",
      "assert flatten([[1],[2]]) == [1,2]"]),
    ("word_freq",
     "Write a Python function `word_freq(s)` that returns a dict of lowercase word frequencies.",
     ["assert word_freq('a b a') == {'a':2,'b':1}", "assert word_freq('') == {}"]),
    ("caesar",
     "Write a Python function `caesar(s, k)` that shifts each lowercase letter by k positions (wrapping), "
     "leaving other characters unchanged.",
     ["assert caesar('abc', 1) == 'bcd'", "assert caesar('xyz', 3) == 'abc'",
      "assert caesar('a b', 1) == 'b c'"]),
]



HARD_TASKS = [
    ("lru_cache",
     "Write a Python class `LRUCache(capacity)` with `get(key)` (returns value or -1) and "
     "`put(key, value)`. Both must be O(1) amortized. Evict the least recently used key.",
     ["c = LRUCache(2)",
      "c.put(1, 1); c.put(2, 2)",
      "assert c.get(1) == 1",
      "c.put(3, 3)          # evicts key 2",
      "assert c.get(2) == -1",
      "c.put(4, 4)          # evicts key 1",
      "assert c.get(1) == -1",
      "assert c.get(3) == 3",
      "assert c.get(4) == 4"]),
    ("top_k_frequent",
     "Write a Python function `top_k_frequent(nums, k)` that returns the k most frequent "
     "values, sorted by descending frequency (ties: smaller value first).",
     ["assert top_k_frequent([1,1,1,2,2,3], 2) == [1,2]",
      "assert top_k_frequent([4,4,4,4], 1) == [4]",
      "assert top_k_frequent([1,2], 2) == [1,2]"]),
    ("merge_intervals",
     "Write a Python function `merge_intervals(intervals)` that merges all overlapping "
     "intervals and returns them sorted by start.",
     ["assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
      "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]",
      "assert merge_intervals([]) == []"]),
    ("valid_parentheses",
     "Write a Python function `valid_parentheses(s)` that returns True iff all brackets "
     "()[]{} are correctly nested and closed.",
     ["assert valid_parentheses('()[]{}') is True",
      "assert valid_parentheses('(]') is False",
      "assert valid_parentheses('([)]') is False",
      "assert valid_parentheses('{[]}') is True",
      "assert valid_parentheses('') is True"]),
    ("spiral_order",
     "Write a Python function `spiral_order(matrix)` that returns the elements of an m x n "
     "matrix in clockwise spiral order.",
     ["assert spiral_order([[1,2,3],[4,5,6],[7,8,9]]) == [1,2,3,6,9,8,7,4,5]",
      "assert spiral_order([[1,2,3,4]]) == [1,2,3,4]",
      "assert spiral_order([[1],[2],[3]]) == [1,2,3]"]),
    ("bfs_shortest_path",
     "Write a Python function `bfs_shortest_path(graph, start, end)` where graph is a dict "
     "adjacency list of unweighted edges. Return the number of edges on the shortest path, "
     "or -1 if unreachable.",
     ["g = {'a':['b','c'],'b':['d'],'c':['d'],'d':[]}",
      "assert bfs_shortest_path(g,'a','d') == 2",
      "assert bfs_shortest_path(g,'a','a') == 0",
      "assert bfs_shortest_path({'a':[]},'a','z') == -1"]),
    ("edit_distance",
     "Write a Python function `edit_distance(a, b)` that returns the Levenshtein distance "
     "between two strings (insert, delete, substitute).",
     ["assert edit_distance('kitten','sitting') == 3",
      "assert edit_distance('','abc') == 3",
      "assert edit_distance('abc','abc') == 0"]),
    ("word_break",
     "Write a Python function `word_break(s, words)` that returns True iff s can be "
     "segmented into a sequence of dictionary words (words may be reused).",
     ["assert word_break('leetcode', ['leet','code']) is True",
      "assert word_break('applepenapple', ['apple','pen']) is True",
      "assert word_break('catsandog', ['cats','dog','sand','and','cat']) is False"]),
]



FRONTIER_TASKS = [
    ("parse_ini",
     "Write a Python function `parse_ini(text)` that parses an INI-style config into a dict of "
     "section -> {key: value}. Lines starting with '#' or ';' are comments, blank lines are ignored, "
     "keys may have spaces around '=', values may contain '=' and are stripped. Keys before any "
     "section go into the section ''.",
     ["t = '[a]\\nx = 1\\n# c\\n[b]\\ny = 2 = 3\\n'",
      "assert parse_ini(t) == {'a': {'x': '1'}, 'b': {'y': '2 = 3'}}",
      "assert parse_ini('k=v') == {'': {'k': 'v'}}",
      "assert parse_ini('') == {}"]),
    ("eval_expr",
     "Write a Python function `eval_expr(expr)` that evaluates an arithmetic expression with "
     "integers, + - * / ( ), unary minus and correct precedence, WITHOUT using eval(). "
     "Division is integer division truncating toward zero. Raise ValueError on malformed input.",
     ["assert eval_expr('1+2*3') == 7",
      "assert eval_expr('(1+2)*3') == 9",
      "assert eval_expr('-2*3') == -6",
      "assert eval_expr('7/2') == 3",
      "assert eval_expr('2*(3+4)-5') == 9"]),
    ("topological_sort",
     "Write a Python function `topological_sort(graph)` where graph is a dict node -> list of "
     "successors. Return a list with a valid topological order. Raise ValueError if the graph "
     "contains a cycle.",
     ["g = {'a':['b','c'],'b':['d'],'c':['d'],'d':[]}",
      "o = topological_sort(g)",
      "assert o.index('a') < o.index('b') < o.index('d')",
      "assert o.index('a') < o.index('c') < o.index('d')",
      "import pytest as _p",
      "try:\n    topological_sort({'a':['b'],'b':['a']})\n    raise AssertionError('no error')\nexcept ValueError:\n    pass"]),
    ("min_window_substring",
     "Write a Python function `min_window_substring(s, t)` that returns the shortest substring of s "
     "containing all characters of t (with multiplicity), or '' if none exists.",
     ["assert min_window_substring('ADOBECODEBANC', 'ABC') == 'BANC'",
      "assert min_window_substring('a', 'a') == 'a'",
      "assert min_window_substring('a', 'aa') == ''"]),
    ("count_inversions",
     "Write a Python function `count_inversions(arr)` that returns the number of inversions "
     "(pairs i<j with arr[i]>arr[j]) in O(n log n).",
     ["assert count_inversions([2,4,1,3,5]) == 3",
      "assert count_inversions([]) == 0",
      "assert count_inversions([5,4,3,2,1]) == 10",
      "assert count_inversions([1,2,3]) == 0"]),
    ("tree_codec",
     "Write a Python function `serialize(root)` and `deserialize(data)` for a binary tree given as "
     "nested tuples `(value, left, right)` where leaves are `(value, None, None)`. serialize returns a "
     "string; deserialize restores an equal tuple. Empty tree is None.",
     ["t = (1, (2, None, None), (3, (4, None, None), None))",
      "assert deserialize(serialize(t)) == t",
      "assert deserialize(serialize(None)) is None"]),
    ("roman_roundtrip",
     "Write `int_to_roman(n)` (1..3999) and `roman_to_int(s)` (validate, raise ValueError on bad input).",
     ["assert int_to_roman(1994) == 'MCMXCIV'",
      "assert roman_to_int('MCMXCIV') == 1994",
      "assert roman_to_int(int_to_roman(3999)) == 3999",
      "try:\n    roman_to_int('IIII')\n    raise AssertionError('no error')\nexcept ValueError:\n    pass"]),
]


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    # Fallback: ganze Antwort, wenn sie wie Code aussieht
    if "def " in text:
        return text
    return ""


def run_task_tests(code: str, tests: list[str], timeout=5) -> tuple[bool, str]:
    """Execute code + asserts in a subprocess-like sandbox (thread + timeout)."""
    import subprocess
    program = code + "\n\n" + "\n".join(tests) + "\nprint('OK')\n"
    try:
        p = subprocess.run([sys.executable, "-c", program], capture_output=True,
                           text=True, timeout=timeout)
        if p.returncode == 0 and "OK" in p.stdout:
            return True, ""
        return False, (p.stderr.strip().splitlines() or ["no OK"])[-1][:200]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:200]


def cmd_coding(args):
    configs = parse_configs(args.configs)
    results = {}
    for name, kwargs in configs:
        print(f"\n=== Konfiguration: {name} {kwargs} ===", flush=True)
        rows = []
        # Preset-Wechsel braucht einen Load/Re-Patch: einmal warmlaufen
        t_warm = time.time()
        chat([{"role": "user", "content": "Say OK."}], max_tokens=8,
             preset=kwargs.get("preset"), relay_sign=kwargs.get("relay_sign"),
             relay_alpha=kwargs.get("relay_alpha"))
        warm = time.time() - t_warm
        print(f"  warmup/load: {warm:.1f}s", flush=True)
        tasks = {"easy": CODING_TASKS, "hard": HARD_TASKS, "frontier": FRONTIER_TASKS}[args.tier]
        for tid, prompt, tests in tasks:
            msgs = [{"role": "user", "content": prompt +
                     "\n\nReturn only one ```python code block, no explanation."}]
            ok, err, total_wall, total_tok, tps, attempts = False, "", 0.0, 0, None, 0
            best_code = ""
            try:
                for attempt in range(max(1, args.best_of)):
                    r = chat(msgs, max_tokens=args.max_tokens, temperature=0.0,
                             preset=kwargs.get("preset"), relay_sign=kwargs.get("relay_sign"),
                             relay_alpha=kwargs.get("relay_alpha"))
                    attempts += 1
                    total_wall += r["wall_s"]
                    total_tok += r["completion_tokens"] or 0
                    code = extract_code(r["text"])
                    if not code:
                        ok, err, best_code = False, "no code block", ""
                        continue
                    best_code = code
                    ok, err = run_task_tests(code, tests)
                    if ok:
                        break
                # Agentic self-repair: feed the failing test output back
                for _rep in range(args.repair):
                    if ok:
                        break
                    r = chat(msgs + [{"role": "assistant", "content": "```python\n" + best_code + "\n```"},
                                     {"role": "user", "content":
                                      f"The tests fail with: {err}\nFix the code. "
                                      "Return only the corrected ```python code block."}],
                             max_tokens=args.max_tokens, temperature=0.0,
                             preset=kwargs.get("preset"), relay_sign=kwargs.get("relay_sign"),
                             relay_alpha=kwargs.get("relay_alpha"))
                    attempts += 1
                    total_wall += r["wall_s"]
                    total_tok += r["completion_tokens"] or 0
                    code = extract_code(r["text"])
                    if not code:
                        continue
                    best_code = code
                    ok, err = run_task_tests(code, tests)
            except (urllib.error.URLError, TimeoutError) as e:
                rows.append({"task": tid, "ok": False, "err": f"api:{e}", "wall_s": None})
                continue
            tps = (total_tok / total_wall) if total_wall else None
            rows.append({"task": tid, "ok": ok, "err": err, "wall_s": round(total_wall, 2),
                         "completion_tokens": total_tok, "tok_s": round(tps, 2) if tps else None,
                         "attempts": attempts, "code_len": len(best_code)})
            print(f"  {tid:18s} {'PASS' if ok else 'FAIL':4s} {total_wall:6.1f}s "
                  f"x{attempts} {(round(tps,1) if tps else '-')!s:>6} tok/s {('' if ok else err)[:50]}", flush=True)
        n = len(rows)
        passed = sum(1 for r in rows if r["ok"])
        tps_vals = [r["tok_s"] for r in rows if r["tok_s"]]
        results[name] = {
            "config": kwargs,
            "n": n, "passed": passed, "pass_at_1": round(passed / n, 4) if n else 0.0,
            "mean_tok_s": round(statistics.mean(tps_vals), 2) if tps_vals else None,
            "median_wall_s": round(statistics.median([r["wall_s"] for r in rows if r["wall_s"]]), 2),
            "warmup_s": round(warm, 1),
            "rows": rows,
        }
        print(f"  -> {passed}/{n} = {passed/n*100:.0f} %", flush=True)

    out = {"api": API, "model": MODEL, "max_tokens": args.max_tokens,
           "tier": args.tier, "repair": args.repair, "best_of": args.best_of,
           "results": results}
    os.makedirs(args.out_dir, exist_ok=True)
    _safe_model = MODEL.replace("/", "_").replace(":", "_")
    path = os.path.join(args.out_dir,
                        f"coding_bench_{_safe_model}_{args.tier}_r{args.repair}_n{args.best_of}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")
    print("\n| Konfiguration | Pass@1 | tok/s | median wall |")
    print("|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['passed']}/{r['n']} ({r['pass_at_1']*100:.0f}%) | "
              f"{r['mean_tok_s']} | {r['median_wall_s']}s |")
    return out


# ─────────────────────────── Speed ───────────────────────────
def cmd_speed(args):
    configs = parse_configs(args.configs)
    out = {"api": API, "model": MODEL, "runs": {}}
    prompt = ("Explain in exactly 200 words how attention works in a transformer.")
    for name, kwargs in configs:
        runs = []
        for rep in range(args.reps):
            r = chat([{"role": "user", "content": prompt}],
                     max_tokens=args.max_tokens, temperature=0.0,
                     preset=kwargs.get("preset"), relay_sign=kwargs.get("relay_sign"),
                     relay_alpha=kwargs.get("relay_alpha"))
            tps = r["completion_tokens"] / r["wall_s"] if r["completion_tokens"] else None
            runs.append({"wall_s": round(r["wall_s"], 3),
                         "completion_tokens": r["completion_tokens"],
                         "tok_s": round(tps, 2) if tps else None})
            print(f"  {name} rep{rep}: {r['wall_s']:.2f}s "
                  f"{r['completion_tokens']} tok {round(tps,1) if tps else '-'} tok/s", flush=True)
        good = [x["tok_s"] for x in runs if x["tok_s"]]
        out["runs"][name] = {"config": kwargs, "runs": runs,
                             "median_tok_s": round(statistics.median(good), 2) if good else None}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "speed_bench.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")
    return out


# ─────────────────────────── Context ───────────────────────────
FILLER = ("The archive contains routine maintenance notes about ventilation, "
          "lighting schedules and inventory counts. ")


def build_haystack(approx_tokens: int, needle: str, depth: float = 0.5) -> str:
    # ~4 chars/token for this filler
    target_chars = approx_tokens * 4
    filler = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]
    pos = int(len(filler) * depth)
    return filler[:pos] + "\n\n" + needle + "\n\n" + filler[pos:]


def cmd_context(args):
    configs = parse_configs(args.configs)
    out = {"api": API, "model": MODEL, "results": {}}
    needle = "The secret maintenance code is KESTREL-7741."
    question = ("Read the text above and answer with the secret maintenance code only. "
                "Format: KESTREL-XXXX")
    for name, kwargs in configs:
        rows = []
        for n_tok in args.lengths:
            doc = build_haystack(n_tok, needle)
            msgs = [{"role": "user", "content": doc + "\n\n" + question}]
            try:
                r = chat(msgs, max_tokens=24, temperature=0.0,
                         preset=kwargs.get("preset"), relay_sign=kwargs.get("relay_sign"),
                         relay_alpha=kwargs.get("relay_alpha"))
                hit = "KESTREL-7741" in (r["text"].upper().replace(" ", "-")) or "7741" in r["text"]
                ok = bool(hit)
                rows.append({"approx_tokens": n_tok, "hit": ok, "wall_s": round(r["wall_s"], 2),
                             "prompt_tokens": r["prompt_tokens"], "answer": r["text"][:120]})
                print(f"  {name} ~{n_tok} tok: {'HIT' if ok else 'miss'} ({r['wall_s']:.1f}s) "
                      f"-> {r['text'][:60]!r}", flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append({"approx_tokens": n_tok, "hit": False, "error": str(e)[:200]})
                print(f"  {name} ~{n_tok} tok: ERROR {str(e)[:80]}", flush=True)
        out["results"][name] = {"config": kwargs, "rows": rows,
                                "recall": round(sum(1 for x in rows if x.get("hit")) / len(rows), 3)}
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "context_bench.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")
    return out


# ─────────────────────────── Size ───────────────────────────
def cmd_size(args):
    """Byte-Inventar eines echten Modells + GF5/GF3-Packgröße auf denselben Tensoren."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    repo = args.hf_id
    path = snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])
    files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    total_bytes = 0
    linear_bytes = 0
    n_tensors = 0
    shapes = []
    for fn in files:
        with safe_open(os.path.join(path, fn), framework="pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                shp = sl.get_shape()
                n = 1
                for d in shp:
                    n *= d
                nb = n * 2  # bf16
                total_bytes += nb
                if len(shp) == 2:
                    linear_bytes += nb
                    n_tensors += 1
                    shapes.append((k, shp))
    bits_gf5 = 2.625  # gemessen: 13 Ququints + fp16-Block-Skalar pro 128
    bits_gf3 = 1.875
    out = {
        "hf_id": repo,
        "n_safetensors": len(files),
        "total_params_bf16_bytes": total_bytes,
        "linear_bytes_bf16": linear_bytes,
        "linear_tensors": n_tensors,
        "gf5_bytes": int(linear_bytes / 16 * bits_gf5),
        "gf3_bytes": int(linear_bytes / 16 * bits_gf3),
        "ratio_bf16_to_gf5": round(16 / bits_gf5, 2),
        "ratio_bf16_to_gf3": round(16 / bits_gf3, 2),
        "sample_shapes": shapes[:8],
    }
    print(json.dumps(out, indent=2))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "size_bench.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ─────────────────────────── CLI ───────────────────────────
DEFAULT_CONFIGS = [
    "BASELINE",
    "ACTIVE_MANIFOLD",
    "ACTIVE_MANIFOLD_LEAN",
    "ACTIVE_MANIFOLD_RELAY:+1",
    "ACTIVE_MANIFOLD_RELAY:-1",
]


def parse_configs(spec: str):
    names = [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for n in names:
        if ":" in n and n.split(":")[0].endswith("RELAY"):
            preset, sign = n.split(":", 1)
            out.append((n, {"preset": preset, "relay_sign": int(sign), "relay_alpha": 0.5}))
        else:
            out.append((n, {"preset": n}))
    return out


def main():
    global MODEL
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (("coding", cmd_coding), ("speed", cmd_speed), ("context", cmd_context)):
        sp = sub.add_parser(name)
        sp.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
        sp.add_argument("--out-dir", default="bench/results")
        sp.add_argument("--model", default=MODEL)
        if name == "coding":
            sp.add_argument("--max-tokens", type=int, default=400)
            sp.add_argument("--tier", choices=["easy", "hard", "frontier"], default="easy")
            sp.add_argument("--repair", type=int, default=0,
                            help="agentic self-repair rounds (test feedback loop)")
            sp.add_argument("--best-of", type=int, default=1,
                            help="independent samples per task (oracle-selected)")
        if name == "speed":
            sp.add_argument("--reps", type=int, default=3)
            sp.add_argument("--max-tokens", type=int, default=256)
        if name == "context":
            sp.add_argument("--lengths", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
            sp.add_argument("--max-tokens", type=int, default=24)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("size")
    sp.add_argument("--hf-id", default="google/gemma-4-E2B-it")
    sp.add_argument("--out-dir", default="bench/results")
    sp.set_defaults(func=cmd_size, configs="", model="")

    args = p.parse_args()
    if getattr(args, "model", ""):
        MODEL = args.model
    args.func(args)


if __name__ == "__main__":
    main()
