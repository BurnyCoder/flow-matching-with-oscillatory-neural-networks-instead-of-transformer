"""Local: resume a tiny controlled optimization. Global: interruptions must not change the learned trajectory.

Source: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html.
These synthetic CPU fixtures are software checks, not image-generation evidence.
"""

from dataclasses import replace  # Changes only the termination step between controlled runs.

import torch  # Compares exact saved model, optimizer, and RNG tensors.

from ofm.config import ModelConfig, TrainConfig  # Uses production settings and validation.
from ofm.data import PreparedData  # Avoids network downloads in software tests.
from ofm.training import train_model  # Exercises the actual training/checkpoint path.


def test_interrupted_training_matches_uninterrupted(tmp_path):
    """Local: compare two versus four updates. Global: prove checkpoint restoration preserves optimizer and RNG state."""
    pixels = torch.randint(0, 256, (20, 1, 8, 8), dtype=torch.uint8, generator=torch.Generator().manual_seed(9))  # Fixed fixture.
    data = PreparedData(pixels, pixels[:4], torch.arange(16), torch.arange(16, 20), {"cache_sha256": "synthetic-test"})  # Split fixture.
    config = TrainConfig(model=ModelConfig(channels=1, size=8, width=8, depth=1, inner_steps=1), dataset="mnist",
                         device="cpu", batch_size=4, microbatch=2, minutes=1, max_steps=2, runs_root=str(tmp_path))  # Tiny check.
    resumed = tmp_path / "resumed"  # One run is deliberately stopped after two complete optimizer updates.
    train_model(config, data, resumed)  # Writes model, optimizer, EMA, and generator state.
    train_model(replace(config, max_steps=4), data, resumed, resume=True)  # Continues the exact earlier stream.
    train_model(replace(config, max_steps=4), data, tmp_path / "whole")  # Independent uninterrupted reference execution.
    split = torch.load(resumed / "checkpoint.pt", weights_only=True)  # Opens only tensor/primitive artifacts.
    whole = torch.load(tmp_path / "whole" / "checkpoint.pt", weights_only=True)  # Reads the reference checkpoint.
    assert split["step"] == whole["step"] == 4  # Both trajectories must represent the same number of updates.
    for key in whole["model"]:  # Includes all model parameters and deterministic buffers.
        torch.testing.assert_close(split["model"][key], whole["model"][key], rtol=0, atol=0)  # Requires exact CPU equality.
    for key in whole["ema_model"]:  # EMA state is part of the evaluated generator, not an optional training detail.
        torch.testing.assert_close(split["ema_model"][key], whole["ema_model"][key], rtol=0, atol=0)  # Requires identical exports.
    torch.testing.assert_close(split["flow_rng"], whole["flow_rng"], rtol=0, atol=0)  # Future supervision must also agree.
