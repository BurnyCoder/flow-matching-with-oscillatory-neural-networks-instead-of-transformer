"""Local: library-backed noise-to-data targets. Global: use an identical FM objective for every architecture.

Source: https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/conditional_flow_matching.py.
"""

import torch  # Generates explicitly seeded source noise and time samples.
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher  # Reuses the established independent CFM path.

MATCHER = ConditionalFlowMatcher(sigma=0.0)  # The straight bridge has no extra conditional path noise.


def conditional_path(noise: torch.Tensor, images: torch.Tensor, times: torch.Tensor):
    """Local: query the affine path and target. Global: preserve the exact TorchCFM training convention."""
    points = MATCHER.compute_mu_t(noise, images, times)  # At sigma=0 the conditional mean is the sampled state.
    target = MATCHER.compute_conditional_flow(noise, images, times, points)  # Computes the noise-to-data velocity.
    return points, target  # Both arrays have the original image shape.


def flow_batch(images: torch.Tensor, generator: torch.Generator):
    """Local: draw a reproducible supervised batch. Global: a model's internal random draws cannot change its targets."""
    noise = torch.randn(images.shape, device=images.device, generator=generator)  # Explicit stream instead of global RNG.
    times = torch.rand(images.shape[0], device=images.device, generator=generator)  # Uniform continuous flow time.
    points, target = conditional_path(noise, images, times)  # Delegates the path mathematics to TorchCFM.
    return points, times, target  # Provides inputs and ground-truth conditional velocities for MSE regression.
