"""Local: paired-seed inference. Global: separate model-training uncertainty from KID subset Monte Carlo variability.

Sources: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html;
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html.
"""

import itertools  # Enumerates all paired sign assignments for the reproducible planning simulation.

import numpy as np  # Represents the pre-aligned seed populations as finite numerical arrays.
from scipy import stats  # Uses maintained implementations of exact permutations and Student-t quantiles.


def paired_comparison(oscillator, transformer) -> dict:
    """Local: estimate oscillator-minus-transformer effects. Global: report all pairs without optimizing the stopping rule."""
    left, right = np.asarray(oscillator, dtype=float), np.asarray(transformer, dtype=float)  # Normalize caller inputs.
    if left.ndim != 1 or left.shape != right.shape or len(left) == 0:  # Reject ambiguous or empty pairing.
        raise ValueError("Expected nonempty, equally sized one-dimensional paired observations")  # No implicit seed dropping.
    difference = left - right  # Negative values favor oscillators for lower-is-better KID.
    if not np.isfinite(difference).all():  # A failed metric cannot disappear through nanmean.
        raise ValueError("Paired observations must be finite")  # Preserve failed runs in reporting instead.
    mean, count = float(difference.mean()), len(difference)  # Seeds are the independent replication unit.
    result = {"n_pairs": count, "differences": difference.tolist(), "mean_difference": mean, "ci95": None, "p_value": None}
    if count > 1:  # Estimating seed variance requires at least two independent pairs.
        radius = float(stats.t.ppf(0.975, count - 1) * stats.sem(difference))  # Approximate Gaussian-difference interval.
        result["ci95"] = [mean - radius, mean + radius]  # Does not mix in torch-fidelity's subset standard deviation.
        test = stats.permutation_test((difference,), np.mean, permutation_type="samples", vectorized=True,
                                      n_resamples=np.inf, alternative="two-sided")  # Enumerates every paired sign flip.
        result["p_value"] = float(test.pvalue)  # Exact under the sign-exchangeability null, not assumption-free.
    return result  # Interpretation belongs in reports alongside completeness and distributional assumptions.


def planning_power() -> dict:
    """Local: reproduce the pre-training Gaussian simulation. Global: state the study's low power without post-hoc sample expansion."""
    trials, pairs = 20000, 6  # Frozen before observing any trained-model comparison.
    base = np.random.default_rng(20261016).normal(size=(trials, pairs))  # Common Gaussian draws reduce noise between effect scenarios.
    signs = np.asarray(list(itertools.product([-1, 1], repeat=pairs)))  # All 64 equally weighted paired sign assignments.
    results = []  # Retain every planned effect scenario, including the null calibration.
    for effect in [0, 0.5, 0.8, 1, 1.5, 2]:  # Standardized effect means mean paired difference divided by its population SD.
        differences = base + effect  # Unit-variance Gaussian planning model, not fitted to observed seed variation.
        null_means = differences @ signs.T / pairs  # Exact conditional randomization distribution for each simulated study.
        pvalues = (np.abs(null_means) >= np.abs(differences.mean(axis=1))[:, None] - 1e-12).mean(axis=1)  # Inclusive extreme tails.
        power = float((pvalues <= 0.05).mean())  # Uses the same fixed two-sided alpha as the experimental analysis.
        results.append({"standardized_effect": effect, "power": power, "monte_carlo_se": float(np.sqrt(power * (1 - power) / trials))})
    return {"seed": 20261016, "trials": trials, "pairs": pairs, "alpha": 0.05, "model": "independent Gaussian paired differences",
            "common_random_numbers_across_effects": True, "results": results}  # Fully specifies how to reproduce the planning evidence.
