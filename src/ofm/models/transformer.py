"""Local: small DiT-style baseline. Global: compare against time-conditioned attention at matched capacity.

Source: adaptive-layer-normalization design in https://arxiv.org/abs/2212.09748;
attention primitive: https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html.
"""

from torch import nn  # Reuses PyTorch attention instead of implementing an attention kernel.


class TransformerBlock(nn.Module):
    """Local: adaptive-normalized attention and MLP. Global: a credible transformer velocity-network baseline."""

    def __init__(self, width: int, hidden: int):
        """Local: define one transformer block. Global: hidden width permits explicit parameter matching."""
        super().__init__()  # Registers trainable baseline components.
        self.norm_attention = nn.LayerNorm(width, elementwise_affine=False)  # Time modulation supplies the affine parameters.
        self.norm_mlp = nn.LayerNorm(width, elementwise_affine=False)  # Separates attention and MLP normalization.
        self.attention = nn.MultiheadAttention(width, 4, dropout=0, batch_first=True)  # No dropout-dependent vector field.
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, width))  # DiT-style MLP.
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))  # Produces shifts, scales, and residual gates.
        nn.init.zeros_(self.modulation[-1].weight)  # AdaLN-Zero starts each block near the identity transformation.
        nn.init.zeros_(self.modulation[-1].bias)  # Time-dependent gates learn before large internal residuals appear.

    def forward(self, tokens, time_embedding):
        """Local: apply adaptive residual branches. Global: output remains a deterministic function of input and flow time."""
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(time_embedding).chunk(6, dim=-1)  # Two branches.
        normalized = self.norm_attention(tokens) * (1 + scale_a[:, None]) + shift_a[:, None]  # Conditions attention on time.
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]  # Avoids materializing attention maps.
        tokens = tokens + gate_a[:, None] * attended  # Uses the learned attention residual gate.
        normalized = self.norm_mlp(tokens) * (1 + scale_m[:, None]) + shift_m[:, None]  # Conditions the feed-forward branch.
        return tokens + gate_m[:, None] * self.mlp(normalized)  # Returns the next token representation.
