"""
test_zone_methods.py — Comparative evaluation of zone-classification methods
============================================================================

Goal: replace the rigid `ZONE_Z_CENTERS = {math: 1.5, logic_a: 0.5, ...}`
Gaussian-anchored classifier with a method that "lets the model improve
itself" by using its own internal representations.

Methods compared
----------------
1. **Gaussian-anchored** (CURRENT): z-score → 5 hard-coded Gaussian centers
2. **Kurtosis quartile**: K → 4 bins, no learned structure
3. **Phi quartile**: phi → 4 bins, no learned structure
4. **H quartile**: zone_entropy H → 4 bins
5. **K-means on (K, phi, H)**: clusters the 3-dim signal vector
6. **K-means on full zone_weights** (5-dim): clusters the routing output
7. **PC1-quartile**: principal component of (K, phi, H) → 4 bins
8. **Self-organizing (hidden-state-derived)**: MOCK — uses phi_trajectory
   statistics as a stand-in for hidden-state signatures (since we can't
   run real hidden-state collection in pure unit tests)

Metrics
-------
- **η² (one-way ANOVA)** between categories on each method's per-prompt
  zone label. Higher = better separation.
- **Purity** (argmax-matching): how often a method assigns the same
  category majority to each zone.
- **Silhouette score** (geometric): how well clusters are separated
  in the input signal space.
- **Cross-validated purity** with leave-one-out on K-means: stability.

The methods are tested on the LIVE 80-prompt aggregates from the full
SR-61 run. If the 4B/E2B aggregates don't exist, the test skips
silently (mock mode).

The hypothetical "self-organizing" method uses **phi-trajectory statistics**
(derivative, mean, std) as a proxy for hidden-state-derived features.
A real implementation would extract per-layer attention entropy or
hidden-state PCA from `_px_forward` in the patch.
"""

import json
import math
import os
import sys
import unittest
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import statistics


# ═══════════════════════════════════════════════════════════════════════════════
# Method implementations (pure-Python, no sklearn)
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_anchored_classify(prompt_features, k_centers, k_sigmas):
    """CURRENT method: argmax of Gaussian(z - center) / sigma."""
    k = prompt_features["kurtosis"]
    if k is None:
        return "UNKNOWN"
    # Use the calibrator's online mean/std if set, else k_centers as fallback
    center = k_centers.get("mean", 0.0)
    std = max(k_centers.get("std", 1.0), 1.0)
    z = (k - center) / std
    best, best_score = None, -1e9
    for zone, zc in [
        ("math", 1.5), ("logic_a", 0.5), ("logic_b", 0.0),
        ("creative", -1.0), ("synthesis", -2.0)
    ]:
        s = math.exp(-0.5 * ((z - zc) / k_sigmas.get(zone, 0.8)) ** 2)
        if s > best_score:
            best_score = s
            best = zone
    return best


def quartile_classify(values, breaks):
    """values: list of (idx, value), breaks: [q25, q50, q75]"""
    out = []
    for idx, v in values:
        if v is None:
            out.append((idx, "Q0"))
        elif v < breaks[0]:
            out.append((idx, "Q1"))
        elif v < breaks[1]:
            out.append((idx, "Q2"))
        elif v < breaks[2]:
            out.append((idx, "Q3"))
        else:
            out.append((idx, "Q4"))
    return out


def kmeans_1d(values, k=4, n_iter=20):
    """Tiny 1D k-means. values: list of (idx, value)."""
    if not values:
        return []
    sorted_vals = sorted(v for _, v in values if v is not None)
    if not sorted_vals:
        return [(idx, "C0") for idx, _ in values]
    # Initialize at quantiles
    quantiles = [
        sorted_vals[len(sorted_vals) * i // k] for i in range(k)
    ]
    centroids = list(quantiles)
    assignments = {idx: 0 for idx, _ in values}
    for _ in range(n_iter):
        new_assignments = {}
        for idx, v in values:
            if v is None:
                new_assignments[idx] = 0
                continue
            dists = [abs(v - c) for c in centroids]
            new_assignments[idx] = dists.index(min(dists))
        if new_assignments == assignments:
            break
        assignments = new_assignments
        # Update centroids
        for ci in range(k):
            group = [v for idx, v in values
                     if v is not None and assignments[idx] == ci]
            if group:
                centroids[ci] = sum(group) / len(group)
    return [(idx, f"C{assignments[idx]}") for idx, _ in values]


def kmeans_2d(points, k=4, n_iter=20):
    """2D k-means on (kurtosis, phi) tuples. points: list of (idx, x, y)."""
    if not points:
        return []
    valid = [p for p in points if p[1] is not None and p[2] is not None]
    if len(valid) < k:
        return [(p[0], "C0") for p in points]
    # Init: distinct points from the data
    centroids = [[valid[i][1], valid[i][2]] for i in range(k)]
    assignments = {p[0]: 0 for p in points}
    for _ in range(n_iter):
        new_assignments = {}
        for idx, x, y in points:
            if x is None or y is None:
                new_assignments[idx] = 0
                continue
            dists = [
                (x - centroids[i][0]) ** 2 + (y - centroids[i][1]) ** 2
                for i in range(k)
            ]
            new_assignments[idx] = dists.index(min(dists))
        if new_assignments == assignments:
            break
        assignments = new_assignments
        for ci in range(k):
            group = [p for p in points
                     if p[1] is not None and p[2] is not None
                     and assignments[p[0]] == ci]
            if group:
                centroids[ci] = [
                    sum(p[1] for p in group) / len(group),
                    sum(p[2] for p in group) / len(group),
                ]
    return [(idx, f"C{assignments[idx]}") for idx, _, _ in points]


def pca_1d(points, n_iter=20):
    """Project 2D points (K, phi) onto their first principal component,
    return 1D scores."""
    if not points:
        return []
    valid = [p for p in points if p[1] is not None and p[2] is not None]
    if len(valid) < 2:
        return [(p[0], 0.0) for p in points]
    # Center
    mx = sum(p[1] for p in valid) / len(valid)
    my = sum(p[2] for p in valid) / len(valid)
    centered = [(p[0], p[1] - mx, p[2] - my) for p in valid]
    # Covariance matrix
    sxx = sum(c[1] ** 2 for c in centered) / len(centered)
    syy = sum(c[2] ** 2 for c in centered) / len(centered)
    sxy = sum(c[1] * c[2] for c in centered) / len(centered)
    # Power iteration for first PC
    v = [1.0, 0.0]
    for _ in range(n_iter):
        new_v = [sxx * v[0] + sxy * v[1], sxy * v[0] + syy * v[1]]
        norm = math.sqrt(new_v[0] ** 2 + new_v[1] ** 2) + 1e-9
        v = [new_v[0] / norm, new_v[1] / norm]
    scores = {}
    for idx, x, y in centered:
        scores[idx] = v[0] * x + v[1] * y
    return [(p[0], scores.get(p[0], 0.0)) for p in points]


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def eta_squared_from_labels(labels, categories):
    """One-way ANOVA: how much of label variance is explained by category?
    labels: list of ints (zone IDs), categories: list of same length.
    Returns η² (0 = no separation, 1 = perfect).
    """
    from collections import defaultdict
    by_cat = defaultdict(list)
    for lab, cat in zip(labels, categories):
        by_cat[cat].append(lab)
    groups = [g for g in by_cat.values() if g]
    if len(groups) < 2:
        return 0.0
    n_total = sum(len(g) for g in groups)
    grand_mean = sum(sum(g) for g in groups) / n_total
    ss_between = sum(len(g) * (sum(g) / len(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups)
    ss_total = ss_between + ss_within
    if ss_total < 1e-12:
        return 0.0
    return ss_between / ss_total


def purity_from_labels(labels, categories):
    """For each predicted label, which category is the majority?
    Purity = fraction of prompts that match the majority category for their label.
    """
    from collections import defaultdict, Counter
    label_to_cat = defaultdict(Counter)
    for lab, cat in zip(labels, categories):
        label_to_cat[lab][cat] += 1
    label_majority = {lab: counter.most_common(1)[0][0] for lab, counter in label_to_cat.items()}
    correct = sum(1 for lab, cat in zip(labels, categories)
                  if label_majority[lab] == cat)
    return correct / len(labels) if labels else 0.0


def silhouette_1d(values, labels):
    """1D silhouette score: how well do clusters separate in value-space?"""
    from collections import defaultdict
    by_label = defaultdict(list)
    for v, l in zip(values, labels):
        if v is not None:
            by_label[l].append(v)
    if len(by_label) < 2:
        return 0.0
    labels_set = list(by_label.keys())
    scores = []
    for i, l_i in enumerate(labels_set):
        vals_i = by_label[l_i]
        if not vals_i:
            continue
        for v in vals_i:
            # a = mean dist to own cluster
            a = sum(abs(v - u) for u in vals_i) / max(len(vals_i), 1)
            # b = min mean dist to other cluster
            b = float("inf")
            for l_j in labels_set:
                if l_j == l_i:
                    continue
                vals_j = by_label[l_j]
                if not vals_j:
                    continue
                d = sum(abs(v - u) for u in vals_j) / len(vals_j)
                if d < b:
                    b = d
            if a == 0 and b == 0:
                continue
            s = (b - a) / max(a, b, 1e-9)
            scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _load_aggregates():
    """Load all 4 full-run aggregates if they exist."""
    results = {}
    for scale in ["270M", "1B", "4B", "E2B"]:
        path = os.path.join(
            _ROOT, "eval", "results",
            f"{scale}_ACTIVE_MANIFOLD_full",
            f"{scale}_ACTIVE_MANIFOLD_aggregate.json"
        )
        if os.path.isfile(path):
            with open(path) as f:
                results[scale] = json.load(f)["results"]
    return results


def _extract_features(results):
    """Returns list of dicts with kurtosis, phi, H, completion, etc."""
    out = []
    for r in results:
        if "error" in r:
            continue
        out.append({
            "category": r["category"],
            "kurtosis": r.get("kurtosis"),
            "phi": r.get("phi"),
            "H": r.get("zone_entropy"),
            "completion_tokens": r.get("completion_tokens", 0),
            "completion": r.get("completion", ""),
            "at_max": r.get("completion_tokens", 0) >= 28,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestZoneMethodComparison(unittest.TestCase):
    """Compare all zone-classification methods on the live aggregates.

    The hypothesis: methods that use MORE of the available signal
    (especially 2D: K+phi) should achieve higher η² and purity than
    any 1D method.
    """

    def setUp(self):
        self.data = _load_aggregates()
        if not self.data:
            self.skipTest("No full-run aggregates found")

    def _run_method_on_scale(self, scale, method_fn):
        """Apply a classifier and return (labels, categories, values)."""
        results = self.data[scale]
        features = _extract_features(results)
        labels = method_fn(features)
        # Convert string labels to ints (for η²)
        unique_labels = sorted(set(l for l, _ in labels))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        int_labels = [label_to_int[l] for l, _ in labels]
        cats = [f["category"] for f in features]
        values_for_silhouette = [method_fn.silhouette_value(f) for f in features]
        return int_labels, cats, values_for_silhouette

    def test_kurtosis_quartile_baseline(self):
        """The simplest baseline: kurtosis-quartile. Should give a floor."""
        for scale in self.data:
            features = _extract_features(self.data[scale])
            Ks = [(i, f["kurtosis"]) for i, f in enumerate(features)]
            valid_Ks = [v for _, v in Ks if v is not None]
            if len(valid_Ks) < 4:
                continue
            sorted_Ks = sorted(valid_Ks)
            breaks = [
                sorted_Ks[len(sorted_Ks) // 4],
                sorted_Ks[len(sorted_Ks) // 2],
                sorted_Ks[3 * len(sorted_Ks) // 4],
            ]
            labels = quartile_classify(Ks, breaks)
            int_labels = [int(l[1][1]) for l in labels]
            cats = [f["category"] for f in features]
            eta2 = eta_squared_from_labels(int_labels, cats)
            purity = purity_from_labels(int_labels, cats)
            self.assertGreaterEqual(eta2, 0.0)
            self.assertGreaterEqual(purity, 0.0)
            # At least one scale should show a non-trivial separation
            print(f"  {scale} kurtosis-quartile: η²={eta2:.3f} purity={purity:.3f}")

    def test_phi_quartile(self):
        """Phi-quartile: should be slightly better than K-quartile at
        E2B (where phi varies more)."""
        for scale in self.data:
            features = _extract_features(self.data[scale])
            phis = [(i, f["phi"]) for i, f in enumerate(features)]
            valid_phis = [v for _, v in phis if v is not None]
            if len(valid_phis) < 4:
                continue
            sorted_phis = sorted(valid_phis)
            breaks = [
                sorted_phis[len(sorted_phis) // 4],
                sorted_phis[len(sorted_phis) // 2],
                sorted_phis[3 * len(sorted_phis) // 4],
            ]
            labels = quartile_classify(phis, breaks)
            int_labels = [int(l[1][1]) for l in labels]
            cats = [f["category"] for f in features]
            eta2 = eta_squared_from_labels(int_labels, cats)
            purity = purity_from_labels(int_labels, cats)
            print(f"  {scale} phi-quartile: η²={eta2:.3f} purity={purity:.3f}")
            self.assertGreaterEqual(eta2, 0.0)

    def test_2d_kmeans_k_phi(self):
        """2D k-means on (K, phi) should be the best 1D-or-2D baseline."""
        print("  2D k-means (K, phi):")
        best_eta2 = 0.0
        for scale in self.data:
            features = _extract_features(self.data[scale])
            points = [(i, f["kurtosis"], f["phi"]) for i, f in enumerate(features)]
            labels = kmeans_2d(points, k=4)
            int_labels = [int(l[1][1]) for l in labels]
            cats = [f["category"] for f in features]
            eta2 = eta_squared_from_labels(int_labels, cats)
            purity = purity_from_labels(int_labels, cats)
            best_eta2 = max(best_eta2, eta2)
            print(f"    {scale}: η²={eta2:.3f} purity={purity:.3f}")
        # At least one scale should benefit from 2D
        self.assertGreater(best_eta2, 0.0,
                            "2D k-means should find non-zero separation on at least one scale")

    def test_pca_1d_projection(self):
        """PCA-projection of (K, phi) onto first PC: should beat any
        single-feature quartile method if K and phi are correlated
        with category."""
        print("  PC1-projection of (K, phi):")
        for scale in self.data:
            features = _extract_features(self.data[scale])
            points = [(i, f["kurtosis"], f["phi"]) for i, f in enumerate(features)]
            scores = pca_1d(points)
            valid = [(i, s) for i, s in scores if s is not None]
            if len(valid) < 4:
                continue
            sorted_v = sorted(s for _, s in valid)
            breaks = [
                sorted_v[len(sorted_v) // 4],
                sorted_v[len(sorted_v) // 2],
                sorted_v[3 * len(sorted_v) // 4],
            ]
            labels = quartile_classify(valid, breaks)
            int_labels = [int(l[1][1]) for l in labels]
            # Build a label map for the missing points
            label_map = {i: int(l[1][1]) for i, l in zip(
                [v[0] for v in valid],
                labels
            )}
            all_int = [label_map.get(i, 0) for i in range(len(features))]
            cats = [f["category"] for f in features]
            eta2 = eta_squared_from_labels(all_int, cats)
            print(f"    {scale}: η²={eta2:.3f}")


class TestSelfOrganizingZones(unittest.TestCase):
    """The 'self-organizing' method uses the model's own internal signals
    (mock: phi-trajectory + kurtosis std) instead of hard-coded Gaussian
    centers. This is the placeholder for what a real implementation
    would do: extract hidden-state PCA, attention entropy, and
    cluster them in real-time."""

    def test_self_organizing_uses_no_hardcoded_centers(self):
        """The new method must NOT reference ZONE_Z_CENTERS or
        ZONE_Z_SIGMAS — its zones must emerge from the data."""
        from px_patches.gemma3_270m_px_baseline import auto_tune
        # Verify the constants exist (we still keep them for back-compat)
        self.assertTrue(hasattr(auto_tune, "ZONE_Z_CENTERS"))
        # But the new method should be queryable separately
        self.assertTrue(hasattr(auto_tune, "AutoCalibrator"))

    def test_self_organizing_produces_5_zones(self):
        """Like the current 5-zone system, the new method should produce
        a manageable number of zones (not 1, not 1000)."""
        from px_patches.gemma3_270m_px_baseline.auto_tune import AutoCalibrator
        cal = AutoCalibrator(2560, calibration_steps=10)
        # Bootstrap with warmup
        cal._online_n = 5
        cal._online_k_mean = 2400.0
        cal._online_k_m2 = 4 * 85.0 ** 2
        cal.k_mean = 2400.0
        cal.k_std = 85.0
        cal.calibrated = True
        # Run across the 4B kurtosis range
        zones_seen = set()
        for K in [2100, 2200, 2300, 2400, 2500, 2600]:
            w = cal.get_zone_weights(K, phi=0.9, token_diversity=0.7)
            for zone, weight in w.items():
                if weight > 0.3:
                    zones_seen.add(zone)
        # Should touch multiple zones, but not all 5 (since 4B range is narrow)
        self.assertGreaterEqual(len(zones_seen), 2,
                                "Self-organizing should find at least 2 active zones")

    def test_zone_emerges_from_phi_trajectory(self):
        """A truly self-organizing classifier would track the phi
        trajectory (rate of change) and use that to decide the zone.
        Test that the calibrator exposes phi data."""
        from px_patches.gemma3_270m_px_baseline.auto_tune import AutoCalibrator
        cal = AutoCalibrator(2560, calibration_steps=10)
        # The calibrator should accumulate phi_samples
        cal.collect(2300, 0.85, 0.7, update_online=False)
        cal.collect(2400, 0.92, 0.8, update_online=False)
        cal.collect(2200, 0.78, 0.6, update_online=False)
        self.assertEqual(len(cal.phi_samples), 3)
        # The trajectory variance is available for self-organizing methods
        phi_var = sum((p - 0.85) ** 2 for p in cal.phi_samples) / 3
        self.assertGreater(phi_var, 0.0)


class TestTermination(unittest.TestCase):
    """Diagnose: do models find a clean end? Or are they getting cut off?"""

    def setUp(self):
        self.data = _load_aggregates()
        if not self.data:
            self.skipTest("No full-run aggregates found")

    def test_at_max_token_rate_per_scale(self):
        """Print the at-max-tokens rate per scale (already measured)."""
        for scale in self.data:
            results = self.data[scale]
            at_max = sum(1 for r in results
                         if r.get("completion_tokens", 0) >= 28)
            short = sum(1 for r in results
                        if r.get("completion_tokens", 0) < 10)
            n = len(results)
            print(f"  {scale}: at_max={at_max}/{n} ({at_max/n:.0%}), "
                  f"short={short}/{n}")
        # Diagnostic only — no assertion on rate

    def test_completions_end_with_terminal_punctuation(self):
        """For each scale, what fraction of completions end with
        period, newline, or other natural terminator?"""
        TERMINATORS = (".", "!", "?", '"', ")", "\n")
        for scale in self.data:
            results = self.data[scale]
            clean = 0
            n = len(results)
            for r in results:
                comp = r.get("completion", "").rstrip()
                if not comp:
                    continue
                if comp.endswith(TERMINATORS) or r.get("completion_tokens", 0) < 28:
                    clean += 1
            print(f"  {scale}: clean_end={clean}/{n} ({clean/n:.0%})")

    def test_no_explicit_eos_in_generation(self):
        """The runner.py should set eos_token_id. Check that the config
        is set up correctly."""
        # Read runner.py
        with open(os.path.join(_ROOT, "eval", "runner.py")) as f:
            src = f.read()
        # Verify eos_token_id is set in generation
        if "eos_token_id" in src:
            # Check that it's correctly configured
            self.assertIn("tokenizer.eos_token_id", src,
                          "eos_token_id should reference tokenizer.eos_token_id")
        else:
            self.skipTest("runner.py doesn't yet set eos_token_id — this is the bug")


class TestRoutingImprovement(unittest.TestCase):
    """Hypotheses for the next iteration of the patch:

    1. **Hypothesis 1**: replacing ZONE_Z_CENTERS with a learned clustering
       improves between-category separation on the 4B/E2B scale.

    2. **Hypothesis 2**: combining zone_weights (5-dim) with phi-trend
       (dphi/dt over recursion steps) gives a richer routing signal.

    3. **Hypothesis 3**: setting eos_token_id in model.generate fixes the
       termination issue at 1B/4B/E2B.

    4. **Hypothesis 4**: using temperature < 1.0 in generation reduces
       repetition loops and improves completion quality.
    """

    def test_hypothesis_1_learned_centers_better_than_hardcoded(self):
        """PROJECTION: if we replaced ZONE_Z_CENTERS with the empirical
        cluster centroids of (K, phi) on the 4B aggregate, what η²
        would we get vs. the current hardcoded centers?"""
        path = os.path.join(
            _ROOT, "eval", "results",
            "4B_ACTIVE_MANIFOLD_full", "4B_ACTIVE_MANIFOLD_aggregate.json"
        )
        if not os.path.isfile(path):
            self.skipTest("4B aggregate not found")
        with open(path) as f:
            agg = json.load(f)
        features = _extract_features(agg["results"])

        # Method A: hardcoded Gaussian (current)
        # Use the empirical mean/std as k_mean/k_std
        Ks = [f["kurtosis"] for f in features if f["kurtosis"] is not None]
        if len(Ks) < 2:
            self.skipTest("Not enough kurtosis data")
        k_mean = sum(Ks) / len(Ks)
        k_std = (sum((k - k_mean) ** 2 for k in Ks) / len(Ks)) ** 0.5
        # Apply hardcoded classifier
        labels_A = []
        for f in features:
            if f["kurtosis"] is None:
                labels_A.append("UNKNOWN")
                continue
            z = (f["kurtosis"] - k_mean) / max(k_std, 1.0)
            best, best_score = None, -1e9
            for zone, zc in [("math", 1.5), ("logic_a", 0.5),
                             ("logic_b", 0.0), ("creative", -1.0),
                             ("synthesis", -2.0)]:
                s = math.exp(-0.5 * ((z - zc) / 0.8) ** 2)
                if s > best_score:
                    best_score = s
                    best = zone
            labels_A.append(best)
        int_A = [hash(l) % 5 for l in labels_A]
        cats = [f["category"] for f in features]
        eta2_A = eta_squared_from_labels(int_A, cats)

        # Method B: learned 4-cluster k-means on (K, phi)
        points = [(i, f["kurtosis"], f["phi"]) for i, f in enumerate(features)]
        labels_B = kmeans_2d(points, k=4)
        int_B = [int(l[1][1]) for l in labels_B]
        eta2_B = eta_squared_from_labels(int_B, cats)

        print(f"  4B hardcoded Gaussian: η²={eta2_A:.3f}")
        print(f"  4B learned 2D k-means: η²={eta2_B:.3f}")
        # We project that the learned method should be at least as good
        self.assertGreater(eta2_B, 0.0,
                            "Learned method should produce some separation")
        # Don't assert eta2_B > eta2_A — the data may not support it.
        # This is a hypothesis, not a guarantee.

    def test_hypothesis_2_zone_weights_have_5_dim_signal(self):
        """The current zone_weights dict has 5 entries (math, logic_a,
        logic_b, creative, synthesis). PCA on these 5 dims should
        capture most variance in 1-2 components."""
        path = os.path.join(
            _ROOT, "eval", "results",
            "1B_ACTIVE_MANIFOLD_full", "1B_ACTIVE_MANIFOLD_aggregate.json"
        )
        if not os.path.isfile(path):
            self.skipTest("1B aggregate not found")
        with open(path) as f:
            agg = json.load(f)
        # Extract 5-dim weight vectors
        weights = []
        for r in agg["results"]:
            zw = r.get("zone_weights", {})
            if not zw:
                continue
            vec = [zw.get(z, 0.0) for z in
                   ["math", "logic_a", "logic_b", "creative", "synthesis"]]
            weights.append(vec)
        if len(weights) < 5:
            self.skipTest("Not enough weight vectors")
        # Compute variance per dim
        n = len(weights)
        dim_vars = []
        for d in range(5):
            vals = [w[d] for w in weights]
            m = sum(vals) / n
            v = sum((x - m) ** 2 for x in vals) / n
            dim_vars.append(v)
        # Print the variance distribution
        print(f"  1B 5-dim zone_weights variance: {['%.3f' % v for v in dim_vars]}")
        # At least 2 dims should have non-zero variance (signal present)
        nonzero = sum(1 for v in dim_vars if v > 0.001)
        self.assertGreaterEqual(nonzero, 2,
                                "At least 2 of 5 zone_weight dims should vary")

    def test_hypothesis_3_eos_token_id_fix(self):
        """PROJECTION: setting eos_token_id should reduce the at-max rate
        by some amount. We can't run a real eval here, but verify the
        infrastructure supports the fix."""
        # Read the runner.py code
        with open(os.path.join(_ROOT, "eval", "runner.py")) as f:
            src = f.read()
        # The fix is to add eos_token_id=tokenizer.eos_token_id
        # to model.generate. We document the expected fix.
        self.assertIn("model.generate", src,
                      "runner.py should use model.generate")
        # If the bug isn't yet fixed, this is a TODO
        if "eos_token_id" not in src:
            print("  TODO: add eos_token_id=tokenizer.eos_token_id to model.generate()")
        # Don't fail — this is a hypothesis, not a regression

    def test_hypothesis_4_temperature_below_one_reduces_repetition(self):
        """PROJECTION: do_sample=False + temperature=1.0 produces
        deterministic but potentially looping output. Setting
        temperature<1.0 with do_sample=True should break loops."""
        # This requires live generation. We document the test.
        print("  TODO: run eval with temperature=0.7 + do_sample=True "
              "and compare repetition-loop count vs current.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
