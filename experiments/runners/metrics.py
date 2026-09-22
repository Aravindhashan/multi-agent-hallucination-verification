"""Statistical and calibration metrics calculation for the pre-Module-5 experiment.

Provides ECE, Brier score, Kish effective sample size, classification metrics,
bootstrap confidence intervals, and Wilcoxon signed-rank tests with Holm correction.
"""

from __future__ import annotations

import math
import numpy as np
from scipy import stats
from typing import Any


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    """Compute Expected Calibration Error (ECE) and reliability diagram bin statistics (§8.2).

    Bins are equal-width in [0.0, 1.0].
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    bin_stats: list[dict[str, float]] = []

    if n == 0:
        return 0.0, []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Points in bin: [lower, upper) except last bin which is [lower, upper]
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = float((bin_lower + bin_upper) / 2.0)

        bin_stats.append({
            "bin_lower": float(bin_lower),
            "bin_upper": float(bin_upper),
            "bin_center": float((bin_lower + bin_upper) / 2.0),
            "count": bin_count,
            "accuracy": bin_acc,
            "confidence": bin_conf,
        })

    return float(ece), bin_stats


def compute_brier_score(confidences: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier calibration score: mean squared error between probability and binary label."""
    if len(confidences) == 0:
        return 0.0
    return float(np.mean((confidences - labels) ** 2))


def compute_kish_effective_count(cluster_sizes: list[int]) -> float:
    """Compute Kish effective sample size: n_eff = (sum w_i)^2 / sum(w_i^2) (§8.2).

    Weight policy: an item in a redundancy cluster of size k receives weight w_i = 1/k.
    Independent items (cluster size 1) receive w_i = 1.0.
    """
    weights: list[float] = []
    for k in cluster_sizes:
        if k <= 1:
            weights.append(1.0)
        else:
            # For each of the k items in the cluster, weight is 1/k
            for _ in range(k):
                weights.append(1.0 / k)

    if not weights:
        return 0.0

    sum_w = sum(weights)
    sum_sq_w = sum(w ** 2 for w in weights)
    if sum_sq_w == 0.0:
        return 0.0
    return float((sum_w ** 2) / sum_sq_w)


def bootstrap_ci(
    values: np.ndarray,
    num_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and 95% bootstrap confidence interval (mean, lower_95, upper_95)."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(num_iterations)
    for i in range(num_iterations):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return float(np.mean(values)), lower, upper


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * m

    cumulative_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        k = m - rank
        adj_p = min(1.0, p_val * k)
        cumulative_max = max(cumulative_max, adj_p)
        corrected[orig_idx] = float(cumulative_max)

    return corrected


def paired_wilcoxon_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Perform paired Wilcoxon signed-rank test on differences (x - y).

    Returns (statistic, p_value).
    """
    diffs = x - y
    # Filter exact zeros for standard Wilcoxon
    non_zero = diffs[diffs != 0]
    if len(non_zero) < 5:
        return 0.0, 1.0
    stat, p_val = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p_val)
