"""Local: fixed-batch learning and numerical probes. Global: distinguish a trainable implementation from a shape-only demo.

Sources: https://docs.pytorch.org/docs/2.11/optim.html;
https://karpathy.github.io/2019/04/25/recipe/ (overfit one batch before trusting larger experiments).
"""

import argparse  # Supports an isolated process launched by the study runner.
import time  # Measures the diagnostic's occupied compute cost.
from pathlib import Path  # Groups complete diagnostic artifacts.

import torch  # Uses the same automatic differentiation and mixed precision as ordinary training.
from safetensors.torch import save_file  # Preserves full fixed inputs and predictions without text truncation.
from torch.nn import functional as F  # Shares the standard velocity MSE definition.

from ofm.artifacts import (  # Reuses durable evidence handling.
    event,  # Streams measured progress.
    save_checkpoint,  # Preserves reconstructable diagnostic weights.
    seed_for,  # Separates the fixed-batch RNG stream.
    setup_logging,  # Opens persistent progress logs.
    write_json,  # Saves complete outcomes and settings.
)
from ofm.budget import BudgetSession  # Charges development GPU work against the same 24-hour cap.
from ofm.config import ModelConfig, TrainConfig  # Reuses the actual published generator architecture.
from ofm.data import prepare_data  # Draws only from the audited training population.
from ofm.flow import flow_batch  # Fixed supervision uses the exact production TorchCFM path.
from ofm.models.oscillator import (  # Directly checks production geometry.
    normalize_state,  # Tests unit-sphere normalization.
    tangent_projection,  # Tests perpendicular forcing.
)
from ofm.training_state import (  # Shares runtime and initialization policy.
    configure_runtime,  # Enables deterministic hardware settings.
    environment_record,  # Captures actual software/hardware provenance.
    initialize_model,  # Uses the full seeded architecture.
)


def run_checks(directory: Path, device: str = "cuda", steps: int = 500) -> dict:
    """Local: fit one fixed noise/image/time batch. Global: expose gradient, geometry, and learning failures before the pilot."""
    setup_logging(directory)  # Makes diagnostics inspectable while optimization runs.
    config = TrainConfig(model=ModelConfig(channels=1, size=28), dataset="mnist", device=device, batch_size=16, microbatch=16)
    configure_runtime(device)  # Matches deterministic training settings on the selected hardware.
    data = prepare_data("mnist", config.data_root, config.validation_size)  # Never use held-out samples for this fit.
    model, config = initialize_model(config)  # Uses the full default oscillator model, not a simplified test network.
    with BudgetSession(config.budget_path, config.budget_hours * 3600, 600, str(directory),
                       enabled=device.startswith("cuda")) as budget:  # Bounds the whole development probe to ten minutes.
        model = model.to(device).train()  # Allocates accelerator state only after reservation.
        generator = torch.Generator(device=device).manual_seed(seed_for(101, "fixed_batch"))  # Isolated diagnostic RNG.
        inputs, times, targets = flow_batch(data.batch(data.train_indices[:16], device), generator)  # Keep one supervision batch fixed.
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)  # Same initial optimizer settings.
        history, started = [], time.monotonic()  # Retain every measured update, not selected loss examples.
        for step in range(steps + 1):  # Include a genuinely pre-update baseline and the final post-update loss.
            budget.check(10)  # Stop before the diagnostic can consume its entire reservation.
            optimizer.zero_grad(set_to_none=True)  # Each gradient corresponds only to this fixed objective.
            with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                prediction = model(inputs, times)  # Tests the same mixed-precision forward path as production training.
                loss = F.mse_loss(prediction.float(), targets)  # Reduces the regression objective in float32.
            history.append(float(loss.detach()))  # Preserve the full optimization trajectory for reporting.
            if step % 50 == 0:  # Regular live evidence prevents an apparently stuck diagnostic.
                event("fixed_batch_progress", step=step, loss=history[-1])  # No scalar or textual truncation.
            if step < steps:  # The final measurement must not hide another unreported optimizer update.
                loss.backward()  # Exercises gradients through all oscillator evolution steps.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)  # Reject nonfinite gradients.
                optimizer.step()  # Commits one fixed-batch AdamW update.
        model.eval()  # Check deterministic predictions without mutable hidden state.
        with torch.inference_mode():  # Numerical probes do not need gradient storage.
            first, second = model(inputs, times), model(inputs, times)  # Repeating the identical call must reset oscillators.
            states = normalize_state(torch.randn(2, 4, 4, 3, 3, device=device))  # Unit vectors in the actual state layout.
            tangent = tangent_projection(states, torch.randn_like(states))  # Arbitrary local forcing projected onto the sphere.
            deterministic = torch.equal(first, second)  # Requires exact equality on the same runtime/device.
            norm_error = float((states.norm(dim=2) - 1).abs().max())  # Quantifies sphere normalization precision.
            tangent_error = float((states * tangent).sum(2).abs().max())  # Tangent forcing must be perpendicular to state.
        result = {"status": "completed", "steps": steps, "initial_loss": history[0], "final_loss": history[-1],
                  "loss_ratio": history[-1] / history[0], "deterministic": deterministic, "norm_max_error": norm_error,
                  "tangent_max_error": tangent_error, "seconds": time.monotonic() - started, "losses": history,
                  "passed": history[-1] < 0.1 * history[0] and deterministic and norm_error < 1e-5 and tangent_error < 1e-5}
        save_file({"inputs": inputs.cpu(), "times": times.cpu(), "targets": targets.cpu(), "predictions": first.cpu()},
                  directory / "fixed_batch.safetensors")  # Stores every tensor value without printing abbreviated arrays.
        save_checkpoint(directory / "checkpoint.pt", {"config": config.to_dict(), "ema_model": model.state_dict()})  # Exportable probe.
        write_json(directory / "config.json", config.to_dict())  # Documents this diagnostic's intentionally smaller fixed batch.
        write_json(directory / "environment.json", environment_record(device))  # Records hardware and dependency versions.
        write_json(directory / "diagnostic.json", result)  # Preserves failed criteria as well as successful ones.
        event("fixed_batch_complete", **{key: value for key, value in result.items() if key != "losses"})  # Full history is above/on disk.
        if not result["passed"]:  # A failed learning probe blocks the pilot rather than silently continuing.
            raise RuntimeError("Fixed-batch or numerical diagnostic failed; inspect diagnostic.json")  # Explicit gate.
    return result  # All measurements are returned and persisted identically.


if __name__ == "__main__":  # The scheduler runs this probe in its own process to release CUDA allocations afterward.
    parser = argparse.ArgumentParser()  # Keep the diagnostic's internal invocation explicit.
    parser.add_argument("--output", type=Path, required=True)  # Stable path supports recording completed checks.
    parser.add_argument("--device", default="cuda")  # CPU remains usable for development.
    args = parser.parse_args()  # Reject misspelled options before allocating compute.
    run_checks(args.output, args.device)  # Uses exactly the programmatic diagnostic phase.
