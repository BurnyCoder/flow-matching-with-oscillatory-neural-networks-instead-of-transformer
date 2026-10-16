"""Local: resume a tiny controlled optimization. Global: interruptions must not change the learned trajectory.

Source: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html.
These synthetic CPU fixtures are software checks, not image-generation evidence.
"""

import json  # Creates explicit interrupted-time accounting fixtures.
from dataclasses import replace  # Changes only the termination step between controlled runs.

import torch  # Compares exact saved model, optimizer, and RNG tensors.

from ofm.config import ModelConfig, TrainConfig  # Uses production settings and validation.
from ofm.data import PreparedData  # Avoids network downloads in software tests.
from ofm.training import train_model  # Exercises the actual training/checkpoint path.
from ofm.training_state import resume_timing  # Recovery must count discarded computation as spent budget.


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


def test_resume_does_not_refund_discarded_training_time(tmp_path):
    """Local: model state lags a recorded failure. Global: lost updates cannot buy a fresh fixed-time training allowance."""
    config = TrainConfig(device="cpu", runs_root=str(tmp_path))  # No GPU access is needed to test recovery policy.
    prior = {"train_seconds": 30, "config": config.to_dict()}  # Last periodic checkpoint precedes the failed segment.
    (tmp_path / "training.json").write_text(json.dumps({"status": "failed", "train_seconds": 120}))  # Actual occupied training time.
    elapsed, exhausted = resume_timing(prior, config, tmp_path)  # Read both learning state and resource-accounting evidence.
    assert elapsed == 120 and not exhausted  # Resume weights at the safe boundary while retaining all known compute cost.


def test_unclosed_gpu_reservation_cannot_extend_a_run(tmp_path):
    """Local: simulate a hard-killed worker. Global: conservative crash charging applies to per-run time as well as the total ledger."""
    config = TrainConfig(device="cuda", runs_root=str(tmp_path))  # Metadata only; the test never allocates CUDA tensors.
    prior = {"train_seconds": 30, "config": config.to_dict()}  # The last checkpoint alone cannot reveal when the worker stopped.
    (tmp_path / "budget.json").write_text(json.dumps({"history": [], "active": {"label": str(tmp_path)}}))  # Unrefunded reservation.
    elapsed, exhausted = resume_timing(prior, config, tmp_path)  # Unknown time is charged conservatively, never inferred as free.
    assert elapsed == 2700 and exhausted  # The original 45-minute arm cannot silently restart its unused checkpoint clock.
