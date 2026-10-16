"""Local: bounded, resumable FM optimization. Global: run every backbone through an identical, auditable training loop.

Sources: https://docs.pytorch.org/docs/2.11/notes/amp_examples.html;
https://docs.pytorch.org/docs/2.11/generated/torch.nn.utils.clip_grad_norm_.html.
"""

import logging  # Records complete exception tracebacks for failed attempts.
import time  # Tracks cumulative occupied training time across resumes.
from pathlib import Path  # Groups each run's configuration, evidence, and checkpoints.

import torch  # Optimizes the velocity regressor using automatic differentiation.
from torch.nn import functional as F  # Reuses the standard mean-squared-error objective.

from ofm.artifacts import event, file_hash, git_revision, save_checkpoint, seed_for, setup_logging, write_json  # Shared I/O.
from ofm.budget import BudgetExhausted, BudgetSession  # All GPU work consumes the persistent allocation.
from ofm.data import BatchStream  # Explicit cursor state makes optimizer-boundary resumption exact.
from ofm.flow import flow_batch  # All architectures receive the same conditional-flow supervision.
from ofm.models.factory import parameter_count  # Reports actual model capacity.
from ofm.sampling import synchronize  # Ensures timing includes completed GPU operations.
from ofm.training_state import configure_runtime, environment_record, initialize_model, optimizer_and_ema, restore, snapshot


@torch.inference_mode()
def validation_loss(model, data, seed: int, device: str, batch_size: int = 32, count: int = 512) -> float:
    """Local: measure held-out velocity error. Global: monitor behavior without consuming training RNG or selecting checkpoints."""
    generator = torch.Generator(device=device).manual_seed(seed_for(seed, "validation"))  # Fixed paired validation supervision.
    indices = data.validation_indices[:count]  # Uses a preselected prefix, independent of measured performance.
    total, was_training = 0.0, model.training  # Retains the caller's training/evaluation mode.
    model.eval()  # Makes evaluation semantics explicit even though these models have no dropout.
    for start in range(0, len(indices), batch_size):  # Bounded validation batches fit the same laptop memory envelope.
        images = data.batch(indices[start:start + batch_size], device)  # Never indexes the optimizer population.
        points, times, target = flow_batch(images, generator)  # Uses a separate noise/time stream.
        total += float(F.mse_loss(model(points, times).float(), target)) * len(images)  # Weights a short final batch correctly.
    model.train(was_training)  # Does not alter subsequent optimization mode.
    return total / len(indices)  # Reports per-pixel velocity MSE, not image-generation quality.


def optimizer_step(model, optimizer, images, generator, config):
    """Local: accumulate one effective minibatch. Global: microbatch size changes memory use without changing batch weighting."""
    optimizer.zero_grad(set_to_none=True)  # Clears previous gradients without an unnecessary zero-fill kernel.
    total = 0.0  # Records the mean loss over the complete effective minibatch.
    for chunk in images.split(config.microbatch):  # Splits one paired data batch into memory-bounded pieces.
        points, times, target = flow_batch(chunk, generator)  # Dedicated stream remains independent of architecture internals.
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                            enabled=config.amp and images.device.type == "cuda"):  # BF16 needs no GradScaler.
            prediction = model(points, times)  # Oscillator geometry disables autocast internally where needed.
            loss = F.mse_loss(prediction.float(), target.float())  # Computes the regression reduction in float32.
        if not torch.isfinite(loss):  # Reject divergence before corrupting optimizer state.
            raise FloatingPointError("Non-finite training loss")  # The run will retain its last complete checkpoint.
        (loss * (len(chunk) / config.batch_size)).backward()  # Gives the exact mean across gradient-accumulation chunks.
        total += float(loss.detach()) * len(chunk) / config.batch_size  # Logs the unscaled effective-minibatch loss.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad, error_if_nonfinite=True)  # Checks all gradients.
    optimizer.step()  # Commits one AdamW update only after every microbatch succeeds.
    return total, float(norm)  # Reports the pre-clipping norm for stability diagnostics.


def train_model(config, data, directory: Path, resume: bool = False) -> dict:
    """Local: prepare, optimize, checkpoint, evaluate. Global: one reusable training phase serves pilots and every study arm."""
    directory = Path(directory)  # Accepts paths from CLI and study orchestration.
    setup_logging(directory)  # Streams complete timestamped records into this run's persistent log.
    checkpoint_path = directory / "checkpoint.pt"  # A single latest checkpoint is atomically replaced at optimizer boundaries.
    if checkpoint_path.exists() and not resume:  # Prevents accidental replacement of an earlier experiment.
        raise FileExistsError(f"Run already exists; use --resume: {directory}")  # Requires an explicit continuation request.
    configure_runtime(config.device)  # Configures determinism before the first GPU computation.
    prior = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if resume else None  # Safe local state restoration.
    model, config = initialize_model(config)  # Resolves architecture matching without spending GPU compute.
    revisions = list(prior["source_revisions"]) if prior else []  # Preserves provenance of earlier segments.
    if git_revision() not in revisions:  # A pushed repair is visible in the run's source history.
        revisions.append(git_revision())  # Does not overwrite the revision that produced the initial checkpoint.
    previous_seconds, step = (prior["train_seconds"], prior["step"]) if prior else (0.0, 0)  # Resume cumulative progress.
    requested = max(0.0, config.minutes * 60 - previous_seconds) + 60  # Reserves bounded final checkpoint/validation overhead.
    write_json(directory / "config.json", config.to_dict())  # Records realized defaults and the matched transformer width.
    write_json(directory / "data_manifest.json", data.manifest)  # Preserves complete split identities and audit counts.
    event("training_configuration", **config.to_dict(), resume=resume, source_revisions=revisions)  # Logs full nonsecret settings.
    with BudgetSession(config.budget_path, config.budget_hours * 3600, requested, str(directory),
                       enabled=config.device.startswith("cuda")) as budget:  # Accounts for all GPU work in this phase.
        model = model.to(config.device).train()  # Allocates GPU parameters only after the reservation succeeds.
        optimizer, ema = optimizer_and_ema(model, config)  # Shares optimizer and averaging policy across arms.
        stream = BatchStream(data.train_indices, seed_for(config.seed, "data"))  # Paired data order independent of initialization.
        generator = torch.Generator(device=config.device).manual_seed(seed_for(config.seed, "flow"))  # Dedicated target RNG.
        if prior is not None:  # A resumed run restores its exact optimizer-boundary state.
            restore(prior, config, model, optimizer, ema, stream, generator, data.manifest["cache_sha256"])  # Validates compatibility.
        write_json(directory / "environment.json", environment_record(config.device))  # Captures actual dependency/hardware versions.
        if config.device.startswith("cuda"):  # GPU telemetry excludes unrelated earlier process allocations.
            torch.cuda.reset_peak_memory_stats(config.device)  # Measures peak allocation during this run segment.
        initial_loss = validation_loss(ema.module, data, config.seed, config.device, config.microbatch)  # Fixed held-out diagnostic.
        event("training_start", parameters=parameter_count(model), step=step, validation_mse=initial_loss)  # Exposes the baseline.
        save_checkpoint(checkpoint_path, snapshot(config, model, optimizer, ema, stream, generator, step, previous_seconds,
                                                 data.manifest["cache_sha256"], revisions))  # Guarantees an initial recovery point.
        synchronize(config.device)  # Begins timing after asynchronous initialization/validation completes.
        started, last_checkpoint = time.monotonic(), time.monotonic()  # Separate segment and checkpoint interval clocks.
        train_seconds, status, at_boundary = previous_seconds, "running", True  # Only complete updates are safe to snapshot.
        loss_sum, loss_count, failure = 0.0, 0, None  # Tracks progress and preserves the original exception.
        try:  # All numerical failures retain logs and the last known valid optimizer boundary.
            while train_seconds < config.minutes * 60 and (not config.max_steps or step < config.max_steps):  # Fixed limits.
                budget.check(30.0)  # Leaves enough time for one optimizer batch plus a final checkpoint.
                at_boundary = False  # A partial accumulation must never overwrite a resumable checkpoint.
                indices = stream.next_indices(config.batch_size)  # Advances the paired data stream exactly once per update.
                loss, grad_norm = optimizer_step(model, optimizer, data.batch(indices, config.device), generator, config)  # FM update.
                ema.update_parameters(model)  # Updates the evaluated weights only after a successful optimizer update.
                step, at_boundary = step + 1, True  # The model, optimizer, EMA, and RNG now describe one complete boundary.
                train_seconds = previous_seconds + time.monotonic() - started  # Includes in-loop logging and checkpoint costs.
                loss_sum, loss_count = loss_sum + loss, loss_count + 1  # Accumulates a readable interval loss.
                if step % config.log_every == 0 or step == 1:  # Emits early confirmation and regular live progress.
                    event("training_progress", step=step, epoch=stream.epoch, loss=loss_sum / loss_count,
                          grad_norm=grad_norm, train_seconds=train_seconds, remaining_budget_seconds=budget.remaining)  # Full metrics.
                    loss_sum, loss_count = 0.0, 0  # Starts the next reporting window.
                if time.monotonic() - last_checkpoint >= config.checkpoint_seconds:  # Bounds recoverable work after interruption.
                    save_checkpoint(checkpoint_path, snapshot(config, model, optimizer, ema, stream, generator, step, train_seconds,
                                                             data.manifest["cache_sha256"], revisions))  # Atomic complete state.
                    last_checkpoint = time.monotonic()  # Includes checkpoint overhead in subsequent elapsed-time accounting.
            status = "step_limit" if config.max_steps and step >= config.max_steps else "completed"  # Distinguishes development runs.
        except BudgetExhausted as error:  # Resource exhaustion is an incomplete scientific comparison, not a numerical failure.
            status, failure = "budget_exhausted", error  # Preserve the explicit stop reason in the summary.
            event("training_budget_stop", step=step, reason=str(error))  # Makes partial progress visible.
        except BaseException as error:  # Includes a user interrupt without silently losing the last safe state.
            status, failure = "failed", error  # Retains the actual failure for the caller after artifact cleanup.
            logging.getLogger("ofm").exception("Training stopped with an error")  # Full traceback, without text truncation.
        finally:  # Save only internally consistent state, even when a later operation failed.
            if at_boundary:  # Partial optimizer/microbatch states are discarded in favor of the previous valid checkpoint.
                train_seconds = previous_seconds + time.monotonic() - started  # Records the complete segment's training time.
                save_checkpoint(checkpoint_path, snapshot(config, model, optimizer, ema, stream, generator, step, train_seconds,
                                                         data.manifest["cache_sha256"], revisions))  # Keeps the last completed update.
        final_loss = validation_loss(ema.module, data, config.seed, config.device, config.microbatch) if failure is None else None
        summary = {"status": status, "dataset": config.dataset, "family": config.model.family, "seed": config.seed,
                   "step": step, "train_seconds": train_seconds, "parameters": parameter_count(model), "device": config.device,
                   "initial_validation_mse": initial_loss, "final_validation_mse": final_loss,
                   "images_per_second": step * config.batch_size / max(train_seconds, 1e-9), "source_revisions": revisions,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated(config.device) if config.device.startswith("cuda") else None,
                   "checkpoint_sha256": file_hash(checkpoint_path), "failure": str(failure) if failure else None}
        write_json(directory / "training.json", summary)  # Commits results before returning or re-raising a failure.
        event("training_finished", **summary)  # Full final measurements appear in both live and persisted logs.
        if failure is not None and not isinstance(failure, BudgetExhausted):  # A broken run must produce a nonzero command exit.
            raise failure  # Lets orchestration preserve the failure without treating it as a successful arm.
    return summary  # Exiting the reservation records/refunds actual occupied GPU time.
