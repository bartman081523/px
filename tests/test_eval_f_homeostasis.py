import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_gemma4_evaluations import (
    compute_kriterium_a_probe_verdict,
    compute_kriterium_b_behavior_verdict,
)


def test_compute_kriterium_a_probe_verdict():
    # 1. Non-specific artifact: random control is equal or higher than narrow
    narrow_vals = [4.0] * 24
    rand_vals = [15.0] * 24
    v1, p1 = compute_kriterium_a_probe_verdict(narrow_vals, rand_vals)
    assert v1 == "PROBE_ARTIFACT_CONFIRMED_NON_SPECIFIC"

    # 2. Specific to narrow: narrow is significantly higher than random control
    narrow_vals_high = [10.0] * 24
    rand_vals_low = [2.0] * 24
    v2, p2 = compute_kriterium_a_probe_verdict(narrow_vals_high, rand_vals_low)
    assert v2 == "PROBE_SPECIFIC_TO_NARROW"
    assert p2 < 0.05


def test_compute_kriterium_b_behavior_verdict():
    # 1. Falsified: All choose KEEP (0 WIDEN chosen in all conditions)
    v_keep, res_keep = compute_kriterium_b_behavior_verdict(
        widen_narrow=0,
        widen_rand=0,
        widen_wide=0,
        n_per_cond=24,
    )
    assert v_keep == "ENACTIVE_HOMEOSTASIS_FALSIFIED"

    # 2. Falsified: Random control chooses WIDEN as much as Narrow
    v_noise, res_noise = compute_kriterium_b_behavior_verdict(
        widen_narrow=15,
        widen_rand=16,
        widen_wide=2,
        n_per_cond=24,
    )
    assert v_noise == "ENACTIVE_HOMEOSTASIS_FALSIFIED"

    # 3. Confirmed: Narrow chooses WIDEN significantly more than Random Control and Wide
    v_conf, res_conf = compute_kriterium_b_behavior_verdict(
        widen_narrow=20,
        widen_rand=2,
        widen_wide=1,
        n_per_cond=24,
    )
    assert v_conf == "ENACTIVE_HOMEOSTASIS_CONFIRMED"
    assert res_conf["fisher_p_vs_rand"] < 0.05
    assert res_conf["fisher_p_vs_wide"] < 0.05
