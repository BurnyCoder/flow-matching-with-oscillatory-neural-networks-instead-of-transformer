"""Local: readable phase orchestration. Global: keep command entry points independent of implementation details.

Each phase owns one concern; training, sampling, evaluation, and publication reuse the same saved contracts.
"""

from pathlib import Path  # Groups phase artifacts within the selected run directory.

from ofm.artifacts import setup_logging, timestamp  # Provides shared timestamped progress and run naming.
from ofm.budget import BudgetSession  # Reserves GPU time before allocating generator parameters.
from ofm.checkpoints import load_generator  # Reuses EMA reconstruction for all inference operations.
from ofm.data import prepare_data  # Downloads and audits the common dataset once.
from ofm.sampling import generate, save_samples  # Separates solving the flow from persisting its outputs.
from ofm.training import train_model  # Owns optimization and resumable state.
from ofm.training_state import configure_runtime  # Shares deterministic device configuration.


def train(config, directory: Path | None = None, resume: bool = False):
    """Local: prepare data then optimize. Global: one command completes a real, reproducible training workflow."""
    directory = directory or Path(config.runs_root) / f"{timestamp()}-{config.dataset}-{config.model.family}-s{config.seed}"  # Unique run.
    setup_logging(directory)  # Includes dataset preparation messages in the run record.
    data = prepare_data(config.dataset, config.data_root, config.validation_size)  # Freezes audited train/validation membership.
    return train_model(config, data, directory, resume=resume)  # All model families enter the identical training phase.


def sample(source: Path, output: Path | None, count: int, seed: int, nfe: int, batch_size: int, device: str):
    """Local: load, generate, save. Global: checkpoint sampling works without training or an external inference service."""
    output = output or Path(source) / "samples" / timestamp()  # Keeps standalone sampling attempts separate.
    setup_logging(output)  # Captures progress in the sampling artifact group.
    configure_runtime(device)  # Applies the same numerical settings as evaluation.
    model, config = load_generator(source, "cpu")  # CPU reconstruction does not spend the GPU budget.
    with BudgetSession(config.budget_path, config.budget_hours * 3600, 1800, str(output), enabled=device.startswith("cuda")) as budget:
        model = model.to(device)  # Moves weights only after the reservation succeeds.
        images, metadata = generate(model, count, seed, nfe, batch_size, device, budget)  # Fixed-noise complete sample population.
        save_samples(output, images, metadata)  # Stores raw tensors, a preselected grid, and all measured costs.
    return {"output": str(output), **metadata}  # Provides a usable result path to CLI callers.


def evaluate(**kwargs):
    """Local: delegate the evaluation phase. Global: optional metric imports stay outside model/training modules."""
    from ofm.evaluation import evaluate_run  # Loads metric machinery only when requested.
    return evaluate_run(**kwargs)  # Keeps detailed metric and artifact policy in its own module.


def study(**kwargs):
    """Local: delegate the pre-specified schedule. Global: a single command can resume the bounded research workflow."""
    from ofm.study import run_study  # Avoids circular imports while reusing the public phase functions.
    return run_study(**kwargs)  # The schedule owns ordering, phase state, and freeze gates.


def report(**kwargs):
    """Local: delegate evidence synthesis. Global: reports are generated from recorded results, not claimed expectations."""
    from ofm.reporting import create_reports  # Typesetting dependencies are needed only for reporting.
    return create_reports(**kwargs)  # Keeps data analysis and PDF rendering out of the training loop.


def publish(**kwargs):
    """Local: delegate artifact export. Global: explicit publication is isolated from ordinary local experimentation."""
    from ofm.publishing import publish_study  # Hub authentication is needed only after artifacts are prepared.
    return publish_study(**kwargs)  # Exports an explicit allowlist, excluding credentials and downloaded source data.
