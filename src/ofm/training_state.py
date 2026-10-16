"""Local: construct and persist training state. Global: make interruptions reproducible rather than fresh experiments.

Sources: https://docs.pytorch.org/docs/2.11/notes/randomness.html;
https://docs.pytorch.org/docs/2.11/generated/torch.optim.swa_utils.AveragedModel.html.
"""

import importlib.metadata  # Records actual installed dependencies rather than intended version ranges.
import json  # Reconciles checkpoint progress with failed-run and conservative reservation records.
import os  # Sets documented deterministic CUDA configuration before kernels run.
import sys  # Records the actual Python runtime in provenance.
from dataclasses import replace  # Constructs the matched unablated initialization without mutating the experiment.
from pathlib import Path  # Canonicalizes ledger labels and run directories for recovery matching.

import torch  # Provides model, optimizer, and RNG serialization.
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn  # Reuses PyTorch's maintained EMA implementation.

from ofm.artifacts import git_revision, seed_for  # Stable stream identities and source provenance.
from ofm.models.factory import build_model  # Shares checkpoint and training construction.


def resume_timing(prior: dict, config, directory: Path) -> tuple[float, bool]:
    """Local: reconcile saved progress with spent time. Global: failed/discarded updates cannot extend a fixed-budget comparison."""
    elapsed = float(prior["train_seconds"])  # Model recovery begins at the last complete checkpoint's recorded clock.
    summary_path = directory / "training.json"  # A graceful failure may have recorded work after that checkpoint.
    if summary_path.exists():  # Preserve known compute cost even when its corresponding updates were discarded.
        elapsed = max(elapsed, json.loads(summary_path.read_text()).get("train_seconds", 0))  # Never refund lost optimizer work.
    interrupted = False  # CPU runs and clean GPU exits have no unknown outstanding reservation.
    if config.device.startswith("cuda") and config.budget_path.exists():  # Only GPU work has a persistent reservation ledger.
        ledger = json.loads(config.budget_path.read_text())  # Atomic ledger replacement gives a complete accounting snapshot.
        records = ledger.get("history", []) + ([ledger["active"]] if ledger.get("active") else [])  # Most recent reservation is last.
        matching = [record for record in records if Path(record["label"]).resolve() == directory.resolve()]  # Ignore other runs/evaluations.
        interrupted = bool(matching and matching[-1].get("status", "active") in {"active", "interrupted_full_charge"})
        if interrupted:  # Exact stop time is unknown, so the original reserved training horizon is conservatively spent.
            elapsed = max(elapsed, prior["config"]["minutes"] * 60)  # Do not restart the remaining duration from an older checkpoint.
    return elapsed, interrupted and elapsed >= config.minutes * 60  # Explicit user extensions remain separate, non-primary configurations.


def configure_runtime(device: str) -> None:
    """Local: configure reproducible kernels. Global: neither CPU fallback nor nondeterministic settings stay hidden."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Required by deterministic CUDA matrix multiplication.
    torch.set_num_threads(int(os.environ.get("OFM_CPU_THREADS", "4")))  # Bounds small-operation CPU overhead.
    torch.use_deterministic_algorithms(True)  # Unsupported nondeterministic operations raise rather than silently drift.
    torch.backends.cudnn.benchmark = False  # Avoids timing-dependent convolution algorithm selection.
    torch.set_float32_matmul_precision("high")  # Applies the same documented TF32 policy to all arms.
    if device.startswith("cuda") and not torch.cuda.is_available():  # Explicit CUDA requests must run on the GPU.
        raise RuntimeError("CUDA was requested but is unavailable; select --device cpu explicitly for CPU execution")


def initialize_model(config):
    """Local: seed architecture initialization. Global: data/noise streams stay paired and ablations share retained weights."""
    torch.manual_seed(seed_for(config.seed, f"initialization:{config.model.family}"))  # Independent model initialization stream.
    model = build_model(config.model)  # Builds on CPU so sizing does not spend GPU allocation.
    if config.model.family in {"no_coupling", "no_frequency"}:  # Mechanism ablations start with matching surviving weights.
        with torch.random.fork_rng(devices=[]):  # Reference construction cannot perturb the eventual training RNG.
            torch.manual_seed(seed_for(config.seed, "initialization:onn"))  # Uses the corresponding unablated ONN seed.
            reference = build_model(replace(config.model, family="onn")).state_dict()  # Original scaffold initialization.
        model.load_state_dict({key: reference[key] for key in model.state_dict()}, strict=True)  # Excludes only removed parameters.
    return model, replace(config, model=model.config)  # Saves the resolved transformer feed-forward width.


def optimizer_and_ema(model, config):
    """Local: create shared optimization machinery. Global: the backbone is the intended primary treatment difference."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999),
                                  weight_decay=config.weight_decay)  # Uses identical objective optimizer settings.
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay))  # Averages all trainable parameters.
    return optimizer, ema  # Both objects must be checkpointed to resume evaluated model behavior.


def snapshot(config, model, optimizer, ema, stream, flow_generator, step, train_seconds, data_hash, revisions):
    """Local: capture one optimizer boundary. Global: future updates reproduce the uninterrupted data/optimizer trajectory."""
    return {"format_version": 1, "config": config.to_dict(), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "ema_model": ema.module.state_dict(), "ema_updates": int(ema.n_averaged),
            "batch_stream": stream.state_dict(), "flow_rng": flow_generator.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if config.device.startswith("cuda") else [],
            "step": step, "train_seconds": train_seconds, "data_hash": data_hash, "source_revisions": revisions}


def restore(state, config, model, optimizer, ema, stream, flow_generator, data_hash):
    """Local: restore all state with compatibility checks. Global: a resume cannot silently change the experimental treatment."""
    previous, current = dict(state["config"]), config.to_dict()  # Compare complete saved settings, including resolved architecture.
    for key in ["minutes", "max_steps", "log_every", "checkpoint_seconds"]:  # Explicit development termination changes are allowed.
        previous.pop(key, None), current.pop(key, None)  # Do not permit changes to optimizer, data, precision, or device.
    if previous != current or state["data_hash"] != data_hash:  # Dataset and scientific-setting changes require a new run.
        raise ValueError("Resume configuration or dataset differs from the checkpoint")  # Prevents mislabeled continued experiments.
    model.load_state_dict(state["model"], strict=True)  # Restores trained parameters and deterministic buffers.
    optimizer.load_state_dict(state["optimizer"])  # Restores Adam moments and update counters on the model device.
    ema.module.load_state_dict(state["ema_model"], strict=True)  # Restores the actual evaluated generator.
    ema.n_averaged.fill_(state["ema_updates"])  # Preserves the first-update versus subsequent-update EMA behavior.
    stream.load_state_dict(state["batch_stream"])  # Restores permutation, cursor, and next-epoch RNG.
    flow_generator.set_state(state["flow_rng"])  # Restores future noise/time supervision exactly.
    torch.set_rng_state(state["torch_rng"])  # Restores any library use of the default CPU random stream.
    if config.device.startswith("cuda"):  # CPU-only checks never initialize a CUDA context.
        torch.cuda.set_rng_state_all(state["cuda_rng"])  # Restores default device streams as well as the dedicated one.


def environment_record(device: str) -> dict:
    """Local: record the actual runtime. Global: distinguish results produced by different libraries or hardware."""
    packages = {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions()}
    return {"python": sys.version, "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu",
            "git_revision": git_revision(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "packages": dict(sorted(packages.items()))}
