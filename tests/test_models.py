"""Local: numerical contracts. Global: prevent invalid geometry or hidden-state-dependent flows.

Sources: AKOrN equations 4–7, https://arxiv.org/html/2410.13821v2;
PyTorch testing: https://docs.pytorch.org/docs/stable/testing.html.
"""

import pytest  # Parametrizes the same contract over every backbone.
import torch  # Supplies tensors and automatic differentiation.

from ofm.config import ModelConfig  # Centralizes architecture settings.
from ofm.models.factory import build_model, parameter_count  # Exercises the public factory.
from ofm.models.oscillator import normalize_state, tangent_projection  # Tests the geometric primitives.


def test_projection_and_normalization():
    """Local: check tangent vectors. Global: rotations must stay on the oscillator sphere."""
    state = normalize_state(torch.randn(2, 8, 4, 3, 3))  # Creates unit vectors per oscillator.
    update = tangent_projection(state, torch.randn_like(state))  # Removes radial motion.
    torch.testing.assert_close(state.square().sum(2), torch.ones(2, 8, 3, 3))  # Confirms unit norms.
    assert (state * update).sum(2).abs().max() < 1e-5  # Tangent updates are orthogonal to the state.


@pytest.mark.parametrize("family", ["onn", "dit", "conv", "no_coupling", "no_frequency"])
@pytest.mark.parametrize("channels,size", [(1, 28), (3, 32)])
def test_velocity_contract(family, channels, size):
    """Local: check shapes and gradients. Global: all arms must plug into the same FM objective."""
    config = ModelConfig(family=family, channels=channels, size=size, width=16, depth=2, inner_steps=2)  # Cheap contract.
    model = build_model(config)  # Uses the same construction as training.
    images = torch.randn(2, channels, size, size)  # Covers actual image resolutions.
    times = torch.tensor([0.0, 1.0])  # Exercises both path endpoints.
    first = model(images, times)  # Predicts a velocity with no external state.
    second = model(images, times)  # Repeating input must repeat output, including in train mode.
    assert first.shape == images.shape  # ODE state and derivative dimensions must agree.
    torch.testing.assert_close(first, second, rtol=0, atol=0)  # Rejects stochastic oscillator initialization.
    first.square().mean().backward()  # Verifies differentiable unrolled dynamics.
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]  # Active gradients.
    assert gradients and all(torch.isfinite(g).all() for g in gradients)  # Rejects numerical divergence.


def test_parameter_matching_and_no_attention():
    """Local: inspect models. Global: preserve the advertised attention-free, matched-capacity comparison."""
    oscillator = build_model(ModelConfig())  # Counts the actual default architecture.
    transformer = build_model(ModelConfig(family="dit"))  # Factory chooses a matching feed-forward width.
    control = build_model(ModelConfig(family="conv"))  # Scaffold is shared with the oscillator network.
    target = parameter_count(oscillator)  # Includes only trainable parameters.
    assert abs(parameter_count(transformer) / target - 1) <= 0.05  # Prevents an undersized baseline.
    assert abs(parameter_count(control) / target - 1) <= 0.05  # Prevents capacity confounding in the control.
    assert not any(isinstance(m, torch.nn.MultiheadAttention) for m in oscillator.modules())  # No attention in ONN.


def test_no_frequency_has_zero_rotation():
    """Local: disable the rotation term. Global: make the trained ablation mechanistically exact."""
    model = build_model(ModelConfig(family="no_frequency", width=16, depth=1))  # Small ablation fixture.
    assert all(block.rotation is None for block in model.blocks)  # No unused frequency parameters or identity term.
