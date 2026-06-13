import json
import math
import statistics
from collections import defaultdict
import numpy as np

def one_way_anova(categories_dict):
    """
    Computes One-Way ANOVA for a dictionary mapping category labels to lists of values.
    Returns F-statistic, p-value (approx), and eta-squared.
    """
    groups = [g for g in categories_dict.values() if len(g) > 1]
    if len(groups) < 2: return {"F": 0, "p": 1, "eta2": 0}
    
    all_vals = [x for g in groups for x in g]
    grand_mean = sum(all_vals) / len(all_vals)
    
    ss_between = sum(len(g) * (sum(g)/len(g) - grand_mean)**2 for g in groups)
    ss_within = sum(sum((x - sum(g)/len(g))**2 for x in g) for g in groups)
    
    df_between = len(groups) - 1
    df_within = len(all_vals) - len(groups)
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    F = ms_between / (ms_within + 1e-9)
    eta2 = ss_between / (ss_between + ss_within + 1e-9)
    
    return {"F": F, "eta2": eta2}

def benchmark_scale(scale_name, aggregate_path):
    with open(aggregate_path) as f:
        agg = json.load(f)
    
    results = [r for r in agg['results'] if 'zone_entropy' in r and r['zone_entropy'] is not None]
    if not results: return
    
    print(f"\n=== Benchmarking Scale: {scale_name} (n={len(results)}) ===")
    
    # Baseline: Current Zone Entropy separation
    by_cat_h = defaultdict(list)
    for r in results: by_cat_h[r['category']].append(r['zone_entropy'])
    base_anova = one_way_anova(by_cat_h)
    print(f"Current H-separation (η²): {base_anova['eta2']:.4f}")
    
    # Method 1: Kurtosis only (Z-score bins)
    Ks = [r['kurtosis'] for r in results]
    mean_k, std_k = statistics.mean(Ks), statistics.stdev(Ks)
    
    def classify_k(k):
        z = (k - mean_k) / (std_k + 1e-9)
        if z > 1.0: return 4
        if z > 0.0: return 3
        if z > -1.0: return 2
        return 1
    
    # Method 2: Phi only (Z-score bins)
    Ps = [r['phi'] for r in results]
    mean_p, std_p = statistics.mean(Ps), statistics.stdev(Ps)
    def classify_p(p):
        z = (p - mean_p) / (std_p + 1e-9)
        return 4 if z > 1.0 else 3 if z > 0 else 2 if z > -1.0 else 1

    # Method 3: 2D K-Means (Self-Organizing)
    # Mocking k-means clusters by finding centers in the 2D space
    points = np.array([[r['kurtosis'], r['phi']] for r in results])
    # Normalize
    p_mean = points.mean(axis=0)
    p_std = points.std(axis=0) + 1e-9
    norm_points = (points - p_mean) / p_std
    
    # 4 clusters (logic_a, logic_b, math, creative)
    # Simple k-means implementation
    centers = norm_points[np.random.choice(len(norm_points), 4, replace=False)]
    for _ in range(10):
        dists = np.linalg.norm(norm_points[:, None] - centers, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([norm_points[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(4)])
        if np.allclose(centers, new_centers): break
        centers = new_centers
        
    # Evaluate 2D clusters: Purity (how much each cluster maps to a category)
    cat_to_int = {cat: i for i, cat in enumerate(set(r['category'] for r in results))}
    cat_labels = np.array([cat_to_int[r['category']] for r in results])
    
    cluster_purities = []
    for i in range(4):
        mask = (labels == i)
        if not np.any(mask): continue
        cats_in_cluster = cat_labels[mask]
        most_common = np.bincount(cats_in_cluster).max()
        purity = most_common / len(cats_in_cluster)
        cluster_purities.append(purity)
    
    avg_purity = sum(cluster_purities) / len(cluster_purities)
    print(f"2D K-Means (K, Phi) Avg Purity: {avg_purity:.4f}")
    
    # ANOVA on Cluster Labels
    by_cat_cluster = defaultdict(list)
    for r, label in zip(results, labels):
        by_cat_cluster[r['category']].append(label)
    cluster_anova = one_way_anova(by_cat_cluster)
    print(f"Cluster-separation (η²): {cluster_anova['eta2']:.4f}")

if __name__ == "__main__":
    import os
    scales = ["270M", "1B", "4B", "E2B"]
    for s in scales:
        path = f"eval/results/{s}_ACTIVE_MANIFOLD_full/{s}_ACTIVE_MANIFOLD_aggregate.json"
        if os.path.exists(path):
            benchmark_scale(s, path)
