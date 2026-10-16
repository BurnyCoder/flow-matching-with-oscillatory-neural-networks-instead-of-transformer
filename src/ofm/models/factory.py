"""Local: reconstruct and size-match backbones. Global: use actual trainable counts, not nominal architecture labels.

Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters.
"""

from dataclasses import replace  # Adjusts a copied configuration without changing the caller's settings.

import torch  # Isolates sizing constructions from the caller's initialization RNG.

from ofm.config import ModelConfig  # Central validated model description.
from ofm.models.backbone import VelocityNetwork  # One wrapper builds every architecture.


def parameter_count(model) -> int:
    """Local: count optimized scalars. Global: expose capacity matching and ablation differences honestly."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)  # Excludes fixed buffers.


def build_model(config: ModelConfig) -> VelocityNetwork:
    """Local: resolve the transformer MLP width. Global: compare attention and oscillators within five percent capacity."""
    if config.family == "dit" and config.ffn_hidden == 0:  # Automatic matching is performed only for an unspecified baseline.
        with torch.random.fork_rng(devices=[]):  # Counting reference models must not change the actual initialization stream.
            target = parameter_count(VelocityNetwork(replace(config, family="onn")))  # Target the corresponding ONN scaffold.
            minimum = parameter_count(VelocityNetwork(replace(config, ffn_hidden=8)))  # Get baseline count at a known MLP width.
        slope = config.depth * (2 * config.width + 1)  # Two linear weights and one hidden bias grow per hidden MLP unit.
        hidden = max(8, round((8 + (target - minimum) / slope) / 8) * 8)  # Round to eight-wide hardware-friendly dimensions.
        config = replace(config, ffn_hidden=hidden)  # Records the realized width in the built model's configuration.
        model = VelocityNetwork(config)  # Initialize the actual baseline once, outside the isolated counting context.
        if abs(parameter_count(model) / target - 1) > 0.05:  # Fail instead of silently accepting a capacity mismatch.
            raise ValueError("Could not match transformer parameters within 5%; inspect the architecture configuration")
        return model  # The checkpoint stores this resolved configuration.
    return VelocityNetwork(config)  # Other arms and already-resolved checkpoints need no sizing pass.
