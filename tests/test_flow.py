"""Local: test path targets and ODE integration. Global: neural architecture changes must not alter flow matching.

Sources: https://github.com/atong01/conditional-flow-matching; https://github.com/rtqichen/torchdiffeq.
"""

import pytest  # Verifies rejection of invalid NFE settings.
import torch  # Builds exact symbolic fixtures for the numerical solver.

from ofm.flow import conditional_path  # Calls the production library-backed path implementation.
from ofm.sampling import integrate  # Calls the same solver adapter used for generated images.


def test_linear_path_targets_and_endpoints():
    """Local: check a known affine bridge. Global: noise-to-data direction and velocity targets must agree."""
    noise = torch.zeros(3, 1, 2, 2)  # A simple source makes the exact expected path transparent.
    data = torch.full_like(noise, 2)  # Constant target gives an analytic velocity of two.
    times = torch.tensor([0.0, 0.5, 1.0])  # Covers the two endpoints and their midpoint.
    points, velocities = conditional_path(noise, data, times)  # Delegates path semantics to TorchCFM.
    torch.testing.assert_close(points[:, 0, 0, 0], torch.tensor([0.0, 1.0, 2.0]))  # Confirms interpolation convention.
    torch.testing.assert_close(velocities, data - noise)  # Checks the full tensor target, including broadcasting.


class ConstantVelocity(torch.nn.Module):
    """Local: exact derivative fixture. Global: integration can be checked without a trained network."""

    def forward(self, images, times):
        """Local: return a constant derivative. Global: x(1)=x(0)+1 is known independently of the implementation."""
        return torch.ones_like(images)  # The true solution does not depend on image state or time.


def test_midpoint_direction_and_nfe():
    """Local: integrate an analytic field. Global: verify outer flow direction and actual network evaluation count."""
    initial = torch.zeros(2, 1, 4, 4)  # Starts exactly at the origin.
    final, actual_nfe = integrate(ConstantVelocity(), initial, nfe=8)  # Four midpoint steps should call the field eight times.
    torch.testing.assert_close(final, torch.ones_like(initial))  # Rejects reverse integration or an incorrect final time.
    assert actual_nfe == 8  # Distinguishes solver steps from velocity evaluations.
    with pytest.raises(ValueError):  # Midpoint needs two field evaluations per step.
        integrate(ConstantVelocity(), initial, nfe=7)  # Odd requested NFE cannot be silently rounded.
