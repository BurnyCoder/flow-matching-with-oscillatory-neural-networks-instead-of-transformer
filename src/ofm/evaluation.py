"""Local: generation and held-out distribution metrics. Global: one preprocessing path serves every model and control.

Sources: https://torch-fidelity.readthedocs.io/en/latest/usage_api.html;
https://pythonot.github.io/gen_modules/ot.sliced.html; https://arxiv.org/abs/1801.01401.
"""

import json  # Loads the final training status used to identify primary-eligible checkpoints.
import os  # Reads the configured CPU thread limit for NumPy/SciPy metric reductions.
import time  # Measures metric overhead separately from generation latency.
from pathlib import Path  # Groups one complete evaluation attempt.

import ot  # Reuses POT for the explicitly non-Inception MNIST pixel-space diagnostic.
import torch  # Supplies typed populations and held-out tensor indexing.
import torch_fidelity  # Reuses the reference-compatible Inception preprocessing and unbiased KID estimator.
from threadpoolctl import threadpool_limits  # Controls native BLAS pools: https://github.com/joblib/threadpoolctl.
from torch.utils.data import Dataset  # Implements the documented torch-fidelity image-population contract.

from ofm.artifacts import (  # Shared provenance and logging.
    event,  # Streams complete measurements.
    file_hash,  # Identifies the exact evaluated weights.
    git_revision,  # Records evaluator provenance.
    setup_logging,  # Creates per-attempt logs.
    timestamp,  # Names independent attempts.
    write_json,  # Persists complete settings and results.
)
from ofm.budget import BudgetSession  # Evaluation consumes the same cumulative GPU allocation as training.
from ofm.checkpoints import load_generator  # Evaluates EMA weights rather than an accidentally different model state.
from ofm.data import prepare_data  # Reuses the audited split manifest.
from ofm.sampling import (  # Shares exact sampling and timing with the standalone command.
    generate,  # Uses the common noise stream and solver.
    save_samples,  # Retains every raw output.
    synchronize,  # Makes asynchronous timing meaningful.
)
from ofm.training import validation_loss  # Reports held-out velocity error using an independent RNG stream.
from ofm.training_state import configure_runtime  # Applies the same deterministic numerical settings.


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Local: clip, scale, and round generated pixels. Global: fixes one metric conversion for every arm."""
    return images.clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)  # Full raw floats are retained separately.


class ImagePopulation(Dataset):
    """Local: expose uint8 RGB images. Global: torch-fidelity owns resizing and feature normalization identically."""

    def __init__(self, images: torch.Tensor):
        """Local: validate metric input. Global: reject accidental grayscale or already-normalized Inception tensors."""
        if images.ndim != 4 or images.shape[1] != 3 or images.dtype != torch.uint8:  # Enforce the documented input contract.
            raise ValueError("Inception input must be NCHW uint8 with three channels")  # Avoid silent preprocessing differences.
        self.images = images  # Keeps the complete unselected population available to the library's DataLoader.

    def __len__(self):
        """Local: expose sample count. Global: KID/FID retain their explicitly reported population sizes."""
        return len(self.images)  # No resampling or filtering occurs in the adapter.

    def __getitem__(self, index):
        """Local: return one RGB image. Global: keep class labels out of unconditional distribution metrics."""
        return self.images[index]  # Returns native-resolution pixels for torch-fidelity's shared preprocessing.


@threadpool_limits.wrap(limits=int(os.environ.get("OFM_CPU_THREADS", "4")), user_api="blas")  # Prevent metric CPU oversubscription.
def distribution_metrics(generated, real, dataset: str, device: str, seed: int, cache_root: Path,
                         real_cache: str | None = None, batch_size: int = 32) -> dict:
    """Local: compute appropriate distribution diagnostics. Global: use Inception KID only for the CIFAR primary comparison."""
    if dataset == "mnist":  # ImageNet features are not a defensible primary quality measure for grayscale digit pilots.
        value = ot.sliced_wasserstein_distance(generated.flatten(1).numpy(), real.float().div(127.5).sub(1).flatten(1).numpy(),
                                               n_projections=128, p=2, seed=seed)  # Fixed random projections in pixel space.
        return {"pixel_sliced_wasserstein": float(value), "projections": 128}  # Label this diagnostic rather than calling it FID.
    metrics = torch_fidelity.calculate_metrics(input1=ImagePopulation(to_uint8(generated)), input2=ImagePopulation(real),
                                               cuda=device.startswith("cuda"), batch_size=batch_size, isc=False, fid=True,
                                               kid=True, kid_subsets=100, kid_subset_size=min(1000, len(generated), len(real)),
                                               rng_seed=seed, verbose=False, cache_root=str(cache_root),
                                               input2_cache_name=real_cache)  # Only real features are cached under split hashes.
    return {key: float(value) for key, value in metrics.items()}  # Convert NumPy scalars into strict JSON values.


def evaluate_run(source: Path, output: Path | None = None, count: int = 5000, seed: int = 20261016,
                 nfe: int = 32, batch_size: int = 32, device: str = "cuda", split: str = "validation",
                 intervention: str = "none") -> dict:
    """Local: load, generate, measure, persist. Global: a completed evaluation is reproducible from its saved checkpoint and seeds."""
    source = Path(source)  # Accepts both CLI and programmatic paths.
    output = Path(output) if output else source / "evaluations" / timestamp()  # Never silently replaces another evaluation.
    if (output / "evaluation.json").exists():  # Results are immutable once completed.
        raise FileExistsError(f"Evaluation already exists: {output}")  # Callers explicitly reuse or choose a new directory.
    setup_logging(output)  # Persists progress even if a metric download or computation fails.
    configure_runtime(device)  # Fails on unavailable CUDA instead of silently switching benchmark hardware.
    model, config = load_generator(source, "cpu")  # CPU loading precedes GPU budget reservation.
    if split not in {"validation", "test"} or intervention not in {"none", "identity", "shuffle"}:  # Validate API callers too.
        raise ValueError("Unsupported evaluation split or intervention")  # Avoid misleading metadata from misspelled options.
    if intervention != "none" and config.model.family not in {"onn", "no_coupling", "no_frequency"}:  # Sensitivity is oscillator-specific.
        raise ValueError("State interventions require an oscillator model")  # A transformer must not appear to receive a no-op shuffle.
    data = prepare_data(config.dataset, config.data_root, config.validation_size)  # Reuses frozen audited data.
    real = data.test_images if split == "test" else data.images[data.validation_indices]  # Explicit held-out metric population.
    checkpoint = source / "checkpoint.pt" if source.is_dir() else source  # Identifies training checkpoint provenance.
    if not checkpoint.exists():  # A portable export stores safe inference tensors instead of optimizer state.
        checkpoint = source / "model.safetensors"  # Both representations remain content-addressable.
    training = json.loads((source / "training.json").read_text()) if (source / "training.json").exists() else {}  # Status evidence.
    primary = (config.dataset == "cifar10" and split == "test" and count == 5000 and nfe == 32 and intervention == "none"
               and training.get("status") == "completed" and config.minutes == 45 and config.max_steps == 0)  # Eligibility is explicit.
    settings = {"dataset": config.dataset, "family": config.model.family, "training_seed": config.seed, "split": split,
                "intervention": intervention, "generated_count": count, "real_count": len(real), "sampling_seed": seed,
                "metric_seed": seed, "nfe": nfe, "kid_subsets": 100, "kid_subset_size": min(1000, count, len(real)),
                "checkpoint_sha256": file_hash(checkpoint), "data_sha256": data.manifest["cache_sha256"],
                "source_revision": git_revision(), "primary_eligible": primary, "device": device,
                "blas_threads": int(os.environ.get("OFM_CPU_THREADS", "4"))}  # Full measurement provenance and native thread policy.
    write_json(output / "evaluation_config.json", settings)  # Preserve an attempted experiment before any GPU operation.
    event("evaluation_start", **settings)  # Logs all settings rather than only a favorable final score.
    try:  # Retain an explicit failure artifact for incomplete evaluations.
        with BudgetSession(config.budget_path, config.budget_hours * 3600, 1800, str(output),
                           enabled=device.startswith("cuda")) as budget:  # All measured work shares the persistent cap.
            model = model.to(device).eval()  # The reported generator is exactly the saved EMA model.
            model.intervention = intervention  # Deterministic identity or norm-preserving spatial state shuffle.
            images, sampling = generate(model, count, seed, nfe, batch_size, device, budget)  # Saves every seed's source-noise prefix.
            save_samples(output, images, sampling)  # Full outputs survive a later metric failure.
            velocity_mse = validation_loss(model, data, config.seed, device, batch_size)  # Always validation, never fitting to test.
            budget.check(30)  # Refuse an expensive metric when the reservation is already nearly exhausted.
            started = time.perf_counter()  # Separate feature/metric cost from sample latency.
            metrics = distribution_metrics(images, real, config.dataset, device, seed, Path(config.data_root) / "metric-cache",
                                           f"{config.dataset}-{split}-{data.manifest['cache_sha256']}", batch_size)  # Stable real cache.
            synchronize(device)  # Include completed feature extraction in elapsed timing.
            result = {**settings, "status": "completed", "sampling": sampling, "metrics": metrics,
                      "validation_velocity_mse": velocity_mse, "metric_seconds": time.perf_counter() - started}  # Full evidence.
            write_json(output / "evaluation.json", result)  # A final result exists only after every requested metric succeeds.
            event("evaluation_complete", **result)  # Mirrors complete scalar results into timestamped logs.
    except BaseException as error:  # Failed attempts remain visible without pretending missing metrics are zero.
        write_json(output / "failure.json", {**settings, "error": repr(error)})  # Retain the precise failure reason.
        raise  # Orchestration decides whether the remaining pre-specified arms can proceed.
    return result  # Returned values and saved JSON have the same scientific meaning.
