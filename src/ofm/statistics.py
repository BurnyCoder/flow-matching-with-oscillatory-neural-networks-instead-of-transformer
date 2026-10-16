"""Local: paired-seed inference. Global: separate model-training uncertainty from KID subset Monte Carlo variability.

Sources: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html;
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html.
"""

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
