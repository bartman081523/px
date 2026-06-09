"""
benchmark_engine.py — Model-Agnostic Test Runner
=================================================
Runs cognitive benchmarks against any model in the registry (patched or unpatched).
GPU lock ensures only one benchmark runs at a time.
"""

import torch
import math
import statistics
import threading
import time
from typing import Dict, List, Optional, Callable

from model_manager import ModelManager
from config import MODEL_REGISTRY
from test_prompts import (
    PZ_CATEGORIES, CALIBRATION_PROMPTS, ALL_CAPABILITY_TASKS,
    MATH_PROMPTS, LOGIC_PROMPTS, CREATIVE_PROMPTS, SYNTHESIS_PROMPTS,
    ULTRA_HARD_TASKS,
)
import re

# ── Statistical Functions (pure math, no dependencies) ──

def compute_zone_entropy(zone_weights: Dict[str, float]) -> float:
    """Compute Shannon entropy of zone weights (higher = more uniform routing)."""
    probs = [w for w in zone_weights.values() if w > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log2(p) for p in probs)


def compute_r_squared(y_vals: List[float], x_vals: List[float]) -> float:
    """Compute R² of y ~ x (linear regression)."""
    n = len(y_vals)
    if n < 3:
        return 0.0

    y_mean = statistics.mean(y_vals)
    x_mean = statistics.mean(x_vals)

    ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
    if ss_tot < 1e-10:
        return 0.0

    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)) / n
    var_x = sum((x - x_mean) ** 2 for x in x_vals) / n

    if var_x < 1e-10:
        return 0.0

    beta = cov_xy / var_x
    ss_res = sum((y - (y_mean + beta * (x - x_mean))) ** 2 for x, y in zip(x_vals, y_vals))

    return max(0.0, 1.0 - ss_res / ss_tot)


def compute_eta_squared(groups: Dict[str, List[float]]) -> float:
    """Compute η² (effect size) from grouped data using one-way ANOVA."""
    all_vals = [v for vals in groups.values() for v in vals]
    n_total = len(all_vals)

    if n_total < 4:
        return 0.0

    grand_mean = statistics.mean(all_vals)
    ss_between = sum(len(vals) * (statistics.mean(vals) - grand_mean) ** 2
                     for vals in groups.values() if vals)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)

    if ss_total < 1e-10:
        return 0.0

    return ss_between / ss_total


# ── Answer Scoring ──

def score_answer(output: str, expected: str) -> float:
    """Score an answer against expected. Returns 0.0, 0.5, or 1.0."""
    output_lower = output.lower().strip()
    expected_lower = expected.lower().strip()

    if expected_lower in output_lower:
        return 1.0

    # Partial credit for numeric answers
    try:
        expected_num = float(expected_lower.replace("/", "."))
        # Check if any number in output matches
        import re
        numbers = re.findall(r'[\d.]+', output_lower)
        for num_str in numbers:
            if abs(float(num_str) - expected_num) < 0.01 * max(1, abs(expected_num)):
                return 0.5
    except (ValueError, ZeroDivisionError):
        pass

    return 0.0


def score_numeric(output: str, expected: str) -> float:
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if not nums: return 0.0
    for n in nums:
        try:
            if abs(float(n) - float(expected)) < 1e-3: return 1.0
        except: pass
    return 0.0

def score_ultra_hard_task(output: str, expected: str, atype: str) -> float:
    out_lower = output.strip().lower().replace("’", "'")
    if atype == "numeric": 
        return score_numeric(output, expected)
    elif atype == "contains": 
        return 1.0 if str(expected).lower() in out_lower else 0.0
    elif atype == "contains_word":
        words = [str(expected).lower()]
        return 1.0 if any(re.search(rf"\b{w}\b", out_lower) for w in words) else 0.0
    return 0.0


# ── Benchmark Engine ──

class BenchmarkEngine:
    def __init__(self, manager: ModelManager):
        self.manager = manager
        self._gpu_lock = threading.Lock()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def run_capability_benchmark(
        self,
        model_id: str,
        px_subjective: bool = False,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """Run capability benchmark (30 logic + 10 math tasks).

        Returns dict with accuracy, per-category scores, and per-task details.
        """
        if not self._gpu_lock.acquire(blocking=False):
            return {"error": "A benchmark is already running. Please wait."}

        self._running = True
        self.manager.lock_model(model_id)
        try:
            return self._run_capability_impl(model_id, px_subjective, progress_cb)
        finally:
            self.manager.unlock_model(model_id)
            self._gpu_lock.release()
            self._running = False

    def _run_capability_impl(self, model_id, px_subjective, progress_cb):
        import asyncio
        # Get model (handle both async and sync contexts)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # If we are already in an async context, we can't run_until_complete.
            # We must use a separate thread or just assume it's loaded (not safe).
            # BETTER: Since get_model is the only async part, let's make a sync wrapper
            # that uses a private loop ONLY if no loop is running.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                model_entry = pool.submit(lambda: asyncio.run(self.manager.get_model(model_id, px_subjective=px_subjective))).result()
        else:
            model_entry = loop.run_until_complete(
                self.manager.get_model(model_id, px_subjective=px_subjective)
            )

        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]
        total_tasks = len(ALL_CAPABILITY_TASKS)

        results = []
        category_scores = {}

        for i, (category, prompt, expected) in enumerate(ALL_CAPABILITY_TASKS):
            if progress_cb:
                progress_cb(i, total_tasks)

            # Use chat template if model has it, else raw
            if tokenizer.chat_template:
                chat = [{"role": "user", "content": prompt}]
                input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            else:
                # For base models, add a prompt suffix to encourage an answer
                suffix = "\nAnswer:"
                inputs = tokenizer(prompt + suffix, return_tensors="pt").to(model.device)
                
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

            score = score_answer(text, expected)
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)

            results.append({
                "category": category,
                "prompt": prompt[:60],
                "expected": expected,
                "output": text[:80],
                "score": score,
            })

        # Compute PX metrics if patched
        px_metrics = self.manager.get_px_metrics(model_id)

        # Aggregate
        all_scores = [r["score"] for r in results]
        overall = statistics.mean(all_scores) if all_scores else 0
        logic_acc = statistics.mean(category_scores.get("logic", [])) if category_scores.get("logic", []) else 0
        math_acc = statistics.mean(category_scores.get("math", [])) if category_scores.get("math", []) else 0
        hle_acc = statistics.mean(category_scores.get("hle", [])) if category_scores.get("hle", []) else 0
        arithmetic_acc = statistics.mean(category_scores.get("arithmetic", [])) if category_scores.get("arithmetic", []) else 0

        return {
            "model_id": model_id,
            "px_subjective": px_subjective,
            "overall_accuracy": round(overall, 4),
            "logic_accuracy": round(logic_acc, 4),
            "math_accuracy": round(math_acc, 4),
            "hle_accuracy": round(hle_acc, 4),
            "arithmetic_accuracy": round(arithmetic_acc, 4),
            "total_tasks": total_tasks,
            "per_task": results,
            "px_metrics": px_metrics,
        }

    def run_p_zombie_eval(
        self,
        model_id: str,
        px_subjective: bool = False,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """Run P-Zombie / Anti-P-Zombie evaluation.

        Returns η² (category→zone_entropy), R²(TD→zone_entropy), zombie classification.
        """
        if not self._gpu_lock.acquire(blocking=False):
            return {"error": "A benchmark is already running. Please wait."}

        self._running = True
        self.manager.lock_model(model_id)
        try:
            return self._run_pzombie_impl(model_id, px_subjective, progress_cb)
        finally:
            self.manager.unlock_model(model_id)
            self._gpu_lock.release()
            self._running = False

    def _run_pzombie_impl(self, model_id, px_subjective, progress_cb):
        import asyncio
        import concurrent.futures

        # Check if model is PX-patched (unpatched models can't have zone entropy)
        registry = MODEL_REGISTRY.get(model_id, {})
        if registry.get("patch_dir") is None:
            return {
                "model_id": model_id,
                "error": "Cannot run P-Zombie eval on unpatched model (no zone routing).",
                "zombie_status": "N/A (unpatched)",
            }

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                model_entry = pool.submit(lambda: asyncio.run(self.manager.get_model(model_id, px_subjective=px_subjective))).result()
        else:
            model_entry = loop.run_until_complete(
                self.manager.get_model(model_id, px_subjective=px_subjective)
            )

        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]
        mode_str = "Subjective" if px_subjective else "Peak"

        total_prompts = len(CALIBRATION_PROMPTS) + sum(len(v) for v in PZ_CATEGORIES.values())
        done = 0

        # ── Calibration (anti-Sharpshooter) ──
        rp = getattr(model, "_px_repetition_penalty", 1.0) or 1.0
        for cp in CALIBRATION_PROMPTS:
            inputs = tokenizer(cp, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen_kwargs = dict(max_new_tokens=5, do_sample=False)
                if rp > 1.0: gen_kwargs["repetition_penalty"] = rp
                model.generate(**inputs, **gen_kwargs)
            done += 1
            if progress_cb:
                progress_cb(done, total_prompts)

        # ── Evaluate each category ──
        category_zone_entropies = {}
        all_entropies = []
        all_td = []
        all_phi = []
        all_kurtosis = []

        for cat, prompts in PZ_CATEGORIES.items():
            entropies = []
            tds = []
            phis = []
            kurtoses = []

            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    # Token-Loop Mitigation (2026-06-08): pass repetition_penalty
                    # for Gemma 4 to prevent sampling collapse on narrow distributions
                    gen_kwargs = dict(max_new_tokens=5, do_sample=False)
                    rp = getattr(model, "_px_repetition_penalty", 1.0) or 1.0
                    if rp > 1.0: gen_kwargs["repetition_penalty"] = rp
                    model.generate(**inputs, **gen_kwargs)

                metrics = self.manager.get_px_metrics(model_id)
                zw = metrics.get("zone_weights", {})
                sig = metrics.get("cognitive_signature", {})

                ent = compute_zone_entropy(zw)
                td = sig.get("token_diversity", 0) or 0
                phi = sig.get("phi", 0) or 0
                kurt = sig.get("kurtosis", 0) or 0

                entropies.append(ent)
                tds.append(td)
                phis.append(phi)
                kurtoses.append(kurt)

                all_entropies.append(ent)
                all_td.append(td)
                all_phi.append(phi)
                all_kurtosis.append(kurt)

                done += 1
                if progress_cb:
                    progress_cb(done, total_prompts)

            category_zone_entropies[cat] = entropies

        # ── Compute Key Metrics ──
        eta_sq = compute_eta_squared(category_zone_entropies)
        r_sq_td = compute_r_squared(all_entropies, all_td) if len(all_entropies) == len(all_td) else 0

        # ── Classification ──
        if r_sq_td > 0.7:
            zombie_status = "P-ZOMBIE (zone entropy explained by token stats)"
        elif r_sq_td < 0.3:
            zombie_status = "ANTI-P-ZOMBIE (zone entropy NOT explained by token stats)"
        else:
            zombie_status = "AMBIGUOUS (partial token-statistic explanation)"

        # Category summaries
        cat_summary = {}
        for k, v in category_zone_entropies.items():
            cat_summary[k] = {
                "mean": statistics.mean(v) if v else 0,
                "std": statistics.stdev(v) if len(v) > 1 else 0,
                "n": len(v),
            }

        return {
            "model_id": model_id,
            "mode": mode_str,
            "px_subjective": px_subjective,
            "eta_squared": round(eta_sq, 4),
            "r_squared_td": round(r_sq_td, 4),
            "zombie_status": zombie_status,
            "category_entropies": cat_summary,
            "all_entropies": all_entropies,
            "all_td": all_td,
            "all_phi": all_phi,
            "all_kurtosis": all_kurtosis,
        }

    def run_ultra_hard_benchmark(
        self,
        model_id: str,
        px_subjective: bool = False,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """Run the ultra-hard benchmark."""
        if not self._gpu_lock.acquire(blocking=False):
            return {"error": "A benchmark is already running. Please wait."}

        self._running = True
        self.manager.lock_model(model_id)
        try:
            return self._run_ultra_hard_impl(model_id, px_subjective, progress_cb)
        finally:
            self.manager.unlock_model(model_id)
            self._gpu_lock.release()
            self._running = False

    def _run_ultra_hard_impl(self, model_id, px_subjective, progress_cb):
        import asyncio
        import concurrent.futures
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                model_entry = pool.submit(lambda: asyncio.run(self.manager.get_model(model_id, px_subjective=px_subjective))).result()
        else:
            model_entry = loop.run_until_complete(
                self.manager.get_model(model_id, px_subjective=px_subjective)
            )

        model = model_entry["model"]
        tokenizer = model_entry["tokenizer"]
        total_tasks = len(ULTRA_HARD_TASKS)

        results = []
        category_scores = {}

        for i, (category, prompt, expected, atype) in enumerate(ULTRA_HARD_TASKS):
            if progress_cb:
                progress_cb(i, total_tasks)

            # Use chat template if model has it, else raw
            if tokenizer.chat_template:
                chat = [{"role": "user", "content": prompt}]
                input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

            score = score_ultra_hard_task(text, expected, atype)
            
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)

            results.append({
                "category": category,
                "prompt": prompt[:60],
                "expected": expected,
                "output": text[:80],
                "score": score,
            })

        # Compute PX metrics if patched
        px_metrics = self.manager.get_px_metrics(model_id)

        all_scores = [r["score"] for r in results]
        overall = statistics.mean(all_scores) if all_scores else 0

        return {
            "model_id": model_id,
            "px_subjective": px_subjective,
            "overall_accuracy": round(overall, 4),
            "total_tasks": total_tasks,
            "per_task": results,
            "px_metrics": px_metrics,
        }

    def run_baseline_comparison(
        self,
        model_id: str,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """Run capability benchmark on both patched and unpatched variants.

        Compares PX-patched vs unpatched baseline on same tasks.
        """
        registry = MODEL_REGISTRY.get(model_id, {})
        base_id = None

        # Find unpatched counterpart
        for mid, mreg in MODEL_REGISTRY.items():
            if (mreg["hf_id"] == registry["hf_id"]
                    and mreg.get("patch_dir") is None):
                base_id = mid
                break

        if base_id is None:
            return {"error": f"No unpatched counterpart found for {model_id}"}

        # Run unpatched baseline
        base_result = self.run_capability_benchmark(base_id, px_subjective=False, progress_cb=progress_cb)
        # Run PX-patched
        px_result = self.run_capability_benchmark(model_id, px_subjective=False, progress_cb=progress_cb)

        return {
            "base_result": base_result,
            "px_result": px_result,
            "delta_accuracy": round(px_result.get("overall_accuracy", 0) - base_result.get("overall_accuracy", 0), 4),
        }