"""Local: metric-contract checks. Global: prevent preprocessing or inferential mistakes from invalidating the study.

Sources: https://torch-fidelity.readthedocs.io/en/latest/usage_api.html;
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html.
"""

import numpy as np  # Supplies exact paired fixtures without model-training randomness.
import pytest  # Checks numerical tolerances and explicit invalid-input failures.
import torch  # Represents generated image tensors in their native layout.

from ofm.evaluation import ImagePopulation, to_uint8  # The public metric boundary must have one shared conversion.
from ofm.statistics import paired_comparison  # Inference operates on seeds, never on KID subsets.


def test_metric_images_share_fixed_quantization():
    """Local: check clipping and rounding. Global: every generator receives identical Inception preprocessing."""
    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).reshape(1, 1, 1, 5)  # Includes out-of-range samples.
    result = to_uint8(values)  # Uses the production metric conversion rather than a test-only approximation.
    assert result.flatten().tolist() == [0, 0, 128, 255, 255]  # Rounding is fixed and clipping is explicit.
    with pytest.raises(ValueError, match="three"):
        ImagePopulation(result)  # ImageNet Inception cannot silently reinterpret grayscale MNIST.


def test_exact_sign_flip_and_seed_interval():
    """Local: enumerate a known six-pair case. Global: a two-sided test has minimum p=2/64, not zero."""
    result = paired_comparison(np.zeros(6), np.ones(6))  # Every paired difference is exactly minus one.
    assert result["mean_difference"] == -1.0  # Defines the reported direction as oscillator minus transformer.
    assert result["p_value"] == pytest.approx(2 / 64)  # Both equally extreme sign assignments count.
    assert result["ci95"] == [-1.0, -1.0]  # No between-seed variation gives a degenerate descriptive interval.
    assert result["n_pairs"] == 6  # Subset replicates must never inflate the sample size.


def test_invalid_or_single_pair_cannot_claim_significance():
    """Local: reject missing/non-finite pairs. Global: partial studies remain visibly incomplete."""
    with pytest.raises(ValueError):
        paired_comparison([1.0, float("nan")], [1.0, 2.0])  # NaN seeds must not be silently dropped.
    with pytest.raises(ValueError):
        paired_comparison([1.0], [1.0, 2.0])  # Pairing requires equal, explicitly aligned populations.
    result = paired_comparison([0.1], [0.2])  # A partial result is still useful as a descriptive observation.
    assert result["ci95"] is None and result["p_value"] is None  # No invented uncertainty for one seed.
