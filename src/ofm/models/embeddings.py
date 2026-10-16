"""Local: shared time/position encodings. Global: backbone comparisons use the same input information.

Sources: https://arxiv.org/abs/2212.09748; https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.
"""

import math  # Defines the frequency spacing without learned or hidden randomness.

import torch  # Vectorizes continuous sinusoidal encodings.


def sinusoidal(values: torch.Tensor, width: int) -> torch.Tensor:
    """Local: encode continuous scalars. Global: flow time is not discretized into oscillator iterations."""
    frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2, device=values.device) / max(width // 2 - 1, 1))
    angles = values.float().reshape(-1, 1) * frequencies[None]  # Broadcasts each scalar against a fixed frequency bank.
    return torch.cat([angles.sin(), angles.cos()], dim=-1)  # Preserves smooth dependence on the continuous input.


def positional_grid(side: int, width: int) -> torch.Tensor:
    """Local: encode two patch coordinates. Global: every architecture can distinguish spatial locations."""
    rows, columns = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")  # Uses image row/column order.
    encoding = torch.cat([sinusoidal(rows.flatten(), width // 2), sinusoidal(columns.flatten(), width // 2)], dim=-1)
    return encoding.T.reshape(1, width, side, side)  # Broadcasts one fixed position field across the minibatch.


def unpatchify(patches: torch.Tensor, channels: int, side: int, patch: int) -> torch.Tensor:
    """Local: reconstruct a dense image derivative. Global: the flow remains defined in pixel space."""
    batch = patches.shape[0]  # Retains arbitrary sampling batch sizes.
    pixels = patches.reshape(batch, side, side, patch, patch, channels)  # Separates patch grid and within-patch coordinates.
    return pixels.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, side * patch, side * patch)  # Interleaves both grids.
