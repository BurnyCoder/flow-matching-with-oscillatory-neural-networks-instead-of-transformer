"""Local: pre-specified evaluation schedule and controls. Global: assess quality, cost, and sensitivity without selecting favorable runs.

Sources: https://torch-fidelity.readthedocs.io/en/latest/usage_api.html;
https://arxiv.org/abs/2108.02497 (separate development decisions from final evaluation).
"""

import json  # Inspects frozen training outcomes before scheduling final metrics.
from pathlib import Path  # Groups controls separately from primary experimental arms.

import torch  # Seeds real-population partitions without using labels or model outputs.
from safetensors.torch import load_file  # Compares full identity-intervention outputs after deterministic sampling.

from ofm.artifacts import (  # Shared provenance, checkpoints, and complete progress logs.
    event,  # Emits control outcomes in real time.
    git_revision,  # Records the source of untrained baselines.
    save_checkpoint,  # Makes untrained controls independently reconstructable.
    setup_logging,  # Persists control progress.
    write_json,  # Saves complete control settings and metrics.
)
from ofm.budget import BudgetSession  # Controls count toward the same cumulative allocation.
from ofm.config import ModelConfig, TrainConfig  # Reuses the exact model family defaults.
from ofm.data import prepare_data  # Reuses the frozen, duplicate-audited dataset.
from ofm.evaluation import distribution_metrics  # One metric implementation prevents control-specific preprocessing.
from ofm.models.factory import parameter_count  # Reports the actual untrained model capacity too.
from ofm.study import evaluate_entry  # Reuses immutable evaluation attempts and subprocess isolation.
from ofm.training_state import configure_runtime, initialize_model  # Constructs the original seeded initialization.


def untrained_control(directory: Path, family: str, device: str, *, split: str = "test", count: int = 5000) -> None:
    """Local: save an unoptimized generator. Global: establish whether learned models improve over their initial image distribution."""
    if not (directory / "checkpoint.pt").exists():  # Reuse the same initial weights on resume.
        config = TrainConfig(model=ModelConfig(family=family), device=device)  # Same default architecture and seed as the first pair.
        model, config = initialize_model(config)  # Entire construction stays on CPU and consumes no GPU allocation.
        directory.mkdir(parents=True, exist_ok=True)  # Creates an explicit control artifact group.
        save_checkpoint(directory / "checkpoint.pt", {"config": config.to_dict(), "ema_model": model.state_dict()})  # Safe load contract.
        write_json(directory / "config.json", config.to_dict())  # Resolved transformer width is preserved.
        write_json(directory / "training.json", {"status": "untrained", "dataset": "cifar10", "family": family, "seed": 101,
                                                "step": 0, "train_seconds": 0, "parameters": parameter_count(model),
                                                "source_revisions": [git_revision()]})  # Cannot qualify for the primary analysis.
    evaluate_entry(directory, directory / f"evaluations/{split}", device, split=split, count=count)  # Validation supports preflight checks.


def real_control(root: Path, device: str) -> None:
    """Local: compare disjoint real-image populations twice. Global: expose finite-sample metric scale and reproducibility."""
    directory = root / "controls/real-versus-real"  # Clearly separates the metric sanity check from learned generators.
    if (directory / "control.json").exists():  # Never reroll the partition based on the resulting score.
        return  # Reuse the originally committed diagnostic.
    setup_logging(directory)  # Retains any feature-download or numerical failures.
    configure_runtime(device)  # Match all other metric computations.
    config, data = TrainConfig(), prepare_data("cifar10", "data")  # Same dataset policy and cumulative budget.
    order = torch.randperm(len(data.test_images), generator=torch.Generator().manual_seed(20261016))  # Predeclared random split.
    left, right = data.test_images[order[:5000]], data.test_images[order[5000:]]  # Disjoint official-test halves.
    with BudgetSession(config.budget_path, 24 * 3600, 1800, str(directory), enabled=device.startswith("cuda")) as budget:
        budget.check(30)  # Avoid starting a metric when the allocation is already nearly exhausted.
        first = distribution_metrics(left.float().div(127.5).sub(1), right, "cifar10", device, 20261016,
                                     Path("data/metric-cache"), "real-control-" + data.manifest["cache_sha256"])  # Shared renderer.
        budget.check(30)  # Repetition must also fit within the same authorized budget.
        second = distribution_metrics(left.float().div(127.5).sub(1), right, "cifar10", device, 20261016,
                                      Path("data/metric-cache"), "real-control-" + data.manifest["cache_sha256"])  # Identical settings.
        result = {"status": "completed", "kind": "real-versus-real", "seed": 20261016, "left_count": len(left),
                  "right_count": len(right), "left_indices": order[:5000].tolist(), "right_indices": order[5000:].tolist(),
                  "data_sha256": data.manifest["cache_sha256"], "metrics": first, "repeat_metrics": second,
                  "repeat_max_absolute_difference": max(abs(first[key] - second[key]) for key in first)}  # Full partition evidence.
        write_json(directory / "control.json", result)  # Preserve unbiased negative KID estimates if they occur.
        event("real_control_complete", metrics=first, repeat_max_absolute_difference=result["repeat_max_absolute_difference"])


def evaluate_study(root: Path, schedule: list[dict], device: str) -> None:
    """Local: execute all declared assessments. Global: prioritize complete primary seed pairs before secondary diagnostics."""
    for entry in [item for item in schedule if item["phase"] in {"main", "ablations"}]:  # Includes every planned trained seed.
        directory = root / entry["relative_path"]  # Connect the final metric directly to its fixed-budget training run.
        summary = directory / "training.json"  # Missing training cannot be silently replaced by a random model.
        if not summary.exists() or json.loads(summary.read_text())["status"] != "completed":  # Require the final budget checkpoint.
            raise ValueError(f"Final evaluation requires completed training: {directory}")  # Keep incompleteness visible.
        evaluate_entry(directory, directory / "evaluations/test-nfe32", device)  # Primary population and cost for every seed.
    for family in ["onn", "dit", "conv"]:  # Cost curves use only the predeclared first seed, not the best seed.
        directory = root / f"main/cifar10-{family}-s101"  # Same trained weights at every sampling cost.
        for nfe in [8, 16, 64]:  # NFE32 was already measured above, preventing duplicate work.
            evaluate_entry(directory, directory / f"evaluations/test-nfe{nfe}", device, nfe=nfe)  # Same noise stream at each cost.
    for seed in [101, 102]:  # Match the descriptive trained-ablation seed population.
        directory = root / f"main/cifar10-onn-s{seed}"  # Sensitivity tests share exactly the original ONN checkpoint.
        for intervention in ["identity", "shuffle"]:  # Identity is the positive implementation control for the intervention path.
            evaluate_entry(directory, directory / f"evaluations/{intervention}", device, intervention=intervention)  # Full 5,000 samples.
        original = load_file(directory / "evaluations/test-nfe32/samples.safetensors")["images"]  # Complete ordinary outputs.
        identity = load_file(directory / "evaluations/identity/samples.safetensors")["images"]  # Complete identity outputs.
        equal = torch.equal(original, identity)  # Requires equality across all values, not a visually similar grid.
        write_json(directory / "evaluations/identity/control.json", {"identity_exact_equal": equal})  # Persist the causal-path control.
        if not equal:  # Divergence here invalidates interpretation of the sensitivity comparison.
            raise RuntimeError("Identity intervention changed deterministic samples")  # Stop and diagnose rather than hiding it.
    for family in ["onn", "dit", "conv"]:  # Untrained controls expose metric behavior across all primary architectures.
        untrained_control(root / f"controls/untrained-{family}", family, device)  # Never mix these controls into seed inference.
    real_control(root, device)  # Establishes the finite-population metric floor and reproducibility after primary work is secured.
