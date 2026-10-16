"""Local: solve the outer flow and save complete samples. Global: separate generated-data time from oscillator time.

Source: https://github.com/rtqichen/torchdiffeq; https://docs.pytorch.org/vision/stable/utils.html.
"""

import time  # Measures synchronized end-to-end generation latency.
from pathlib import Path  # Keeps images and raw tensors together in one evidence directory.

import torch  # Supplies initial Gaussian states and GPU synchronization.
from safetensors.torch import save_file  # Preserves all unclipped generated tensors without printing/truncating them.
from torchdiffeq import odeint  # Reuses the maintained midpoint integrator.
from torchvision.utils import save_image  # Renders fixed, unselected image grids.

from ofm.artifacts import event, seed_for, write_json  # Shares logging and reproducible seed derivation.


@torch.inference_mode()
def integrate(model, initial: torch.Tensor, nfe: int = 32, budget=None):
    """Local: integrate dx/dt=v(x,t). Global: count actual field calls rather than confusing them with solver steps."""
    if nfe < 2 or nfe % 2:  # The chosen midpoint method evaluates the field exactly twice per step.
        raise ValueError("Midpoint sampling requires an even nfe >= 2")  # Never silently change the comparison budget.
    evaluations = 0  # Counts actual callbacks from the library solver.

    def velocity(times, images):
        """Local: adapt the solver's argument order. Global: every callback resets the backbone's internal state."""
        nonlocal evaluations  # Updates the counter owned by this integration call.
        if budget is not None:  # Study generation shares the same persisted GPU allocation.
            budget.check(2.0)  # Leaves time to finish one bounded field evaluation and release the reservation.
        evaluations += 1  # Counts one network invocation regardless of batch size.
        return model(images, times)  # Uses the model's public image-first contract.

    grid = torch.tensor([0.0, 1.0], device=initial.device)  # Noise is at zero and data at one.
    final = odeint(velocity, initial, grid, method="midpoint", options={"step_size": 2.0 / nfe})[-1]  # Fixed solver settings.
    if not torch.isfinite(final).all():  # A NaN image must not become an apparently valid clipped PNG.
        raise FloatingPointError("Non-finite values produced by the flow ODE")  # Exposes numerical divergence.
    return final, evaluations  # Reports actual evaluations for auditing and quality-cost comparisons.


def synchronize(device: str) -> None:
    """Local: wait for outstanding CUDA kernels. Global: latency measurements include completed GPU work."""
    if str(device).startswith("cuda"):  # CPU operations are synchronous and do not require a CUDA runtime.
        torch.cuda.synchronize(device)  # Makes wall-clock timings meaningful for asynchronous CUDA execution.


@torch.inference_mode()
def generate(model, count: int, seed: int, nfe: int, batch_size: int, device: str, budget=None):
    """Local: sample a fixed stream in bounded batches. Global: compare all arms on identical source noise prefixes."""
    if count < 1 or batch_size < 1:  # Rejects empty artifacts and undefined per-image timing.
        raise ValueError("count and batch_size must be positive")  # Makes input errors explicit.
    generator = torch.Generator(device=device).manual_seed(seed_for(seed, "sampling"))  # Distinct from optimization noise.
    shape = (model.config.channels, model.config.size, model.config.size)  # Derives image shape from the saved model.
    warmup = torch.zeros(min(batch_size, count), *shape, device=device)  # Warmup must not consume the evaluation RNG stream.
    integrate(model, warmup, nfe=2, budget=budget)  # Initializes kernels before the measured sampling interval.
    synchronize(device)  # Excludes asynchronous warmup work from generation timing.
    if str(device).startswith("cuda"):  # GPU-only memory telemetry has a clear device-specific meaning.
        torch.cuda.reset_peak_memory_stats(device)  # Measures this sampler rather than preceding training peaks.
    start, outputs, actual_nfe = time.perf_counter(), [], 0  # Keeps full float outputs for later auditing.
    for offset in range(0, count, batch_size):  # Bounds activation memory on the 8 GB laptop GPU.
        initial = torch.randn(min(batch_size, count - offset), *shape, device=device, generator=generator)  # Paired source noise.
        samples, calls = integrate(model, initial, nfe=nfe, budget=budget)  # The outer solver is identical for all backbones.
        if calls != nfe:  # Solver implementation changes cannot silently alter the reported comparison.
            raise RuntimeError(f"Expected {nfe} evaluations but solver used {calls}")  # Fail rather than mislabel sample cost.
        actual_nfe = calls  # All complete batches use the same per-image evaluation count.
        outputs.append(samples.cpu())  # Transfers every sample, preserving unclipped floating-point values.
        if offset == 0 or (offset // batch_size) % 20 == 0:  # Reports useful progress without one log per ODE step.
            event("sampling_progress", completed=offset + len(samples), total=count, nfe=calls)  # Full structured message.
    synchronize(device)  # Includes the last batch's completed GPU work.
    elapsed = time.perf_counter() - start  # Counts generation and CPU output transfer, excluding model load and warmup.
    images = torch.cat(outputs)  # Reassembles the complete, unselected sample population.
    metadata = {"count": count, "seed": seed, "nfe": actual_nfe, "batch_size": batch_size, "seconds": elapsed,
                "seconds_per_image": elapsed / count, "precision": "float32", "timing_excludes": "model load and warmup",
                "clipped_fraction": float((images.abs() > 1).float().mean()),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None}
    return images, metadata  # Metric evaluation and publication consume the same complete sample set.


def save_samples(directory: Path, images: torch.Tensor, metadata: dict) -> None:
    """Local: persist raw outputs and a fixed grid. Global: visual evidence cannot be selected after observing quality."""
    directory.mkdir(parents=True, exist_ok=True)  # Creates an explicit sampling artifact group.
    save_file({"images": images.contiguous()}, directory / "samples.safetensors")  # Saves all generated values without clipping.
    save_image((images[:64].clamp(-1, 1) + 1) / 2, directory / "grid.png", nrow=8)  # Always displays the first 64 samples.
    write_json(directory / "sampling.json", metadata)  # Records rendering-independent latency, seeds, and clipping statistics.
    event("sampling_complete", directory=str(directory), **metadata)  # Streams full metadata to terminal and persistent log.
