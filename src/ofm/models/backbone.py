"""Local: shared image-to-velocity wrapper. Global: abstract encoding, processing, and decoding across all arms.

Sources: https://arxiv.org/abs/2212.09748; https://arxiv.org/html/2410.13821v2.
"""

import torch  # Provides deterministic buffers and the input/output tensor contract.
from torch import nn  # Shares the same stem and final prediction head between model families.

from ofm.config import ModelConfig  # Carries the serialized architecture description.
from ofm.models.embeddings import positional_grid, sinusoidal, unpatchify  # Named encoding/decoding phases.
from ofm.models.oscillator import OscillatorBlock, normalize_state  # Named oscillatory feature-processing phase.
from ofm.models.transformer import TransformerBlock  # Attention exists only in the explicit baseline.


class VelocityNetwork(nn.Module):
    """Local: map an image/time pair to its velocity. Global: interchangeable stateless field for training and integration."""

    def __init__(self, config: ModelConfig):
        """Local: assemble named modules. Global: keep architecture decisions reconstructible from configuration."""
        super().__init__()  # Enables standard state_dict and device movement.
        self.config, self.intervention = config, "none"  # Interventions are opt-in evaluation diagnostics.
        width, side = config.width, config.size // config.patch  # Shared patch grid and feature size.
        self.stem = nn.Conv2d(config.channels, width, config.patch, stride=config.patch)  # Encodes each input patch once.
        self.register_buffer("positions", positional_grid(side, width))  # Fixes identical positional information across arms.
        self.time_encoder = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))  # Shared time embedding.
        self.output_norm = nn.LayerNorm(width)  # Stabilizes the common dense pixel-velocity head.
        self.output = nn.Linear(width, config.patch**2 * config.channels)  # Predicts real, unbounded velocities per patch.
        if config.family == "dit":  # Only this comparison arm instantiates attention modules.
            self.blocks = nn.ModuleList([TransformerBlock(width, config.ffn_hidden) for _ in range(config.depth)])  # DiT stack.
            self.initial_state = None  # The transformer has no oscillator state to carry.
        else:  # Every non-transformer arm shares the oscillator/readout scaffold.
            self.blocks = nn.ModuleList([OscillatorBlock(config) for _ in range(config.depth)])  # Independent block parameters.
            self.initial_state = nn.Parameter(torch.randn(1, width // config.rotating_dim, config.rotating_dim, 1, 1))  # Learned reset.
        generator = torch.Generator().manual_seed(20261016)  # Intervention randomness cannot perturb training streams.
        self.register_buffer("state_permutation", torch.randperm(side * side, generator=generator))  # Norm-preserving spatial swap.

    def process_features(self, features, time_embedding):
        """Local: dispatch the backbone phase. Global: the wrapper hides attention or oscillator implementation details."""
        if self.initial_state is None:  # DiT consumes a sequence of spatial patch tokens.
            tokens = features.flatten(2).transpose(1, 2)  # Converts NCHW into batch-first token layout.
            for block in self.blocks:  # Applies the fixed-depth time-conditioned transformer.
                tokens = block(tokens, time_embedding)  # Each block preserves token shape.
            return tokens  # The decoder accepts the same layout for every family.
        state = normalize_state(self.initial_state).expand(features.shape[0], -1, -1, *features.shape[-2:])  # Reset per evaluation.
        for index, block in enumerate(self.blocks):  # State persists only within this one network evaluation.
            features, state = block(features, state, time_embedding)  # Evolves states and extracts the next residual features.
            if index == 0 and self.intervention == "shuffle":  # Intervene after input-dependent states have formed.
                state = state.flatten(-2).index_select(-1, self.state_permutation).reshape_as(state)  # Permutes locations, keeps norms.
        return features.flatten(2).transpose(1, 2)  # Presents the common output-head interface.

    def forward(self, images: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Local: encode, process, decode. Global: define v_theta(x,t) with no persistent recurrent sampler state."""
        if images.shape[1:] != (self.config.channels, self.config.size, self.config.size):  # Refuse incompatible checkpoints.
            raise ValueError("Image shape does not match the model configuration")  # Prevents accidental resize/channel mistakes.
        times = times.to(device=images.device, dtype=torch.float32).reshape(-1)  # Accepts scalar or per-example flow time.
        if times.numel() == 1:  # ODE solvers supply a scalar shared across the batch.
            times = times.expand(images.shape[0])  # Broadcasts time without adding state or randomness.
        if times.numel() != images.shape[0]:  # Each example needs exactly one flow time.
            raise ValueError("Time must be scalar or have one entry per image")  # Exposes malformed training batches.
        encoded = self.stem(images) + self.positions  # Combines image patches and fixed spatial information.
        time_embedding = self.time_encoder(sinusoidal(times * 1000, self.config.width))  # Resolves the continuous [0,1] interval.
        tokens = self.process_features(encoded, time_embedding)  # Calls the interchangeable feature-processing phase.
        patches = self.output(self.output_norm(tokens))  # Leaves velocities unclipped for the continuous ODE.
        return unpatchify(patches, self.config.channels, self.config.size // self.config.patch, self.config.patch)  # Dense derivative.
