"""Local: projected coupled-oscillator blocks. Global: replace attention with genuine internal dynamical neurons.

Independent implementation of AKOrN equations 4–7: https://arxiv.org/html/2410.13821v2.
The recurrent control shares the scaffold but uses a conventional nonlinear update.
"""

import torch  # Differentiates through the finite sequence of hidden updates.
from torch import nn  # Supplies documented convolutional and normalization primitives.
from torch.nn import functional as F  # Reuses stable vector normalization and SiLU implementations.

from ofm.config import ModelConfig  # Keeps defaults and ablations explicit in saved configurations.


def normalize_state(state: torch.Tensor) -> torch.Tensor:
    """Local: normalize each rotating vector in FP32. Global: retain the unit-sphere state constraint."""
    return F.normalize(state.float(), p=2, dim=2, eps=1e-8)  # Dimension two is the rotating coordinate, not channel or space.


def tangent_projection(state: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Local: remove the radial component. Global: implement the Kuramoto coupling on the sphere's tangent space."""
    return direction.float() - state * (state * direction.float()).sum(dim=2, keepdim=True)  # Computes y - x<x,y>.


class OscillatorBlock(nn.Module):
    """Local: evolve states and read their mixed orientations. Global: propagate image features without attention."""

    def __init__(self, config: ModelConfig):
        """Local: construct one block. Global: ablations change the named mechanism while preserving the scaffold."""
        super().__init__()  # Registers child modules and parameters for optimization/checkpointing.
        self.config = config  # Holds the fixed number and size of internal steps.
        width, rotating = config.width, config.rotating_dim  # Separates scalar width from vector geometry.
        self.condition_norm = nn.GroupNorm(1, width)  # Normalizes each image without mixing batch members.
        self.time_modulation = nn.Linear(width, 2 * width)  # Provides per-channel scale and shift from continuous time.
        self.coupling = None if config.family == "no_coupling" else nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.rotation = nn.Parameter(0.05 * torch.randn(width // rotating, rotating, rotating)) if config.family in {"onn", "no_coupling"} else None
        self.recurrent_norm = nn.GroupNorm(1, width) if config.family == "conv" else None  # Stabilizes the ordinary recurrence.
        self.mix = nn.Conv2d(width, width * rotating, 1, bias=False)  # Mixes different oscillators before norm extraction.
        self.readout_norm = nn.GroupNorm(1, width)  # Conditions the scale of the extracted orientation features.
        self.readout = nn.Sequential(nn.Conv2d(width, 2 * width, 1), nn.SiLU(), nn.Conv2d(2 * width, width, 1))  # Residual readout.

    def evolve(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Local: run the inner recurrence. Global: these steps are hidden computation, not advances in flow time."""
        batch, groups, rotating, height, width = state.shape  # Exposes vector groups for the geometric operation.
        for _ in range(self.config.inner_steps):  # Unrolls a fixed, documented amount of computation.
            flattened = state.reshape(batch, groups * rotating, height, width)  # Applies ordinary convolutions to vector channels.
            mixed = self.coupling(flattened) if self.coupling is not None else torch.zeros_like(flattened)  # Exact coupling ablation.
            drive = mixed.float() + condition.float()  # Holds stimulus fixed during this block's internal updates.
            if self.recurrent_norm is not None:  # Conventional control keeps the same connectivity and iteration count.
                updated = self.recurrent_norm(flattened + self.config.step_size * F.silu(drive))  # Ordinary residual recurrence.
                state = updated.reshape_as(state)  # Restores the common readout shape without imposing spherical dynamics.
            else:  # Oscillatory arms implement the published tangent update.
                with torch.autocast(device_type=state.device.type, enabled=False):  # Geometry is always evaluated in FP32.
                    direction = tangent_projection(state, drive.reshape_as(state))  # Removes only radial coupling/stimulus motion.
                    if self.rotation is not None:  # No-frequency means zero, not an identity mapping.
                        omega = 0.5 * (self.rotation - self.rotation.transpose(-1, -2))  # Enforces skew symmetry structurally.
                        direction = direction + torch.einsum("gij,bgjhw->bgihw", omega, state)  # Adds natural angular motion.
                    state = normalize_state(state + self.config.step_size * direction)  # Retracts the Euler update to the sphere.
        return state  # The state can initialize the next block but never the next outer ODE call.

    def forward(self, features: torch.Tensor, state: torch.Tensor, time_embedding: torch.Tensor):
        """Local: condition, evolve, and read out. Global: a complete oscillator block produces residual image features."""
        scale, shift = self.time_modulation(time_embedding).chunk(2, dim=1)  # Splits time-dependent affine parameters.
        condition = self.condition_norm(features) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]  # Adds flow time.
        state = self.evolve(state, condition)  # Uses the same condition throughout this block's finite dynamics.
        batch, groups, rotating, height, width = state.shape  # Recovers the flattened scalar-channel width.
        mixed = self.mix(state.reshape(batch, groups * rotating, height, width))  # Readout must mix states before taking norms.
        vectors = mixed.float().reshape(batch, self.config.width, rotating, height, width)  # Each output feature has a vector.
        magnitudes = (vectors.square().sum(dim=2) + 1e-8).sqrt()  # Bare state norms would be constant; mixed norms carry signal.
        return features + 0.1 * self.readout(self.readout_norm(magnitudes)), state  # Residual path aids optimization.
