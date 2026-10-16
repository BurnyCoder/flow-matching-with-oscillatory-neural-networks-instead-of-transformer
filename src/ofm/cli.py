"""Local: command-line interface and .env loading. Global: reproduce the pipeline from a fresh checkout.

Sources: https://docs.python.org/3/library/argparse.html; https://bbc2.github.io/python-dotenv/.
"""

import argparse  # Uses standard-library argument validation rather than adding a CLI framework.
import json  # Loads saved run configuration and prints machine-readable command results.
import os  # Reads explicit environment defaults without exposing credential values.
from dataclasses import replace  # Applies requested termination overrides to a resumed run.
from pathlib import Path  # Restricts .env discovery to the current workspace.

from dotenv import load_dotenv  # Loads user-configured options without overriding already-exported variables.

from ofm.config import ModelConfig, TrainConfig  # Keeps CLI and checkpoint settings consistent.


def parser() -> argparse.ArgumentParser:
    """Local: declare supported commands. Global: expose the complete reproducible workflow through one executable."""
    root = argparse.ArgumentParser(description="Image flow matching with coupled Kuramoto oscillator backbones")  # Descriptive help.
    commands = root.add_subparsers(dest="command", required=True)  # Requires an explicit operation instead of launching training.
    training = commands.add_parser("train", help="Train or resume a velocity network")  # Main optimization command.
    training.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")  # Selects native data preprocessing.
    training.add_argument("--model", choices=["onn", "dit", "conv", "no_coupling", "no_frequency"], default="onn")  # Backbone arm.
    training.add_argument("--seed", type=int, default=101)  # Pairing key for data/noise streams.
    training.add_argument("--minutes", type=float)  # Omitted resume values preserve the saved duration.
    training.add_argument("--steps", type=int)  # Explicit development termination; primary runs remain time-limited.
    training.add_argument("--batch-size", type=int)  # Overrides the effective batch only for a new run.
    training.add_argument("--microbatch", type=int)  # Controls gradient-accumulation memory.
    training.add_argument("--width", type=int, default=128)  # Small test architectures remain explicitly configurable.
    training.add_argument("--depth", type=int, default=4)  # Defaults match the pre-specified backbone.
    training.add_argument("--inner-steps", type=int, default=4)  # Hidden dynamics are distinct from sampling NFE.
    training.add_argument("--device")  # Explicit CPU/CUDA choice overrides environment only for a new run.
    training.add_argument("--run-dir", type=Path)  # Optional stable directory for named study arms.
    training.add_argument("--resume", type=Path)  # Resumes a complete run directory and its serialized scientific settings.
    training.add_argument("--config", type=Path)  # Loads an explicit frozen configuration for a new run.
    for name in ["sample", "evaluate"]:  # Shared inference arguments avoid duplicated interface definitions.
        command = commands.add_parser(name, help=f"{name.capitalize()} a saved EMA generator")  # Both consume saved checkpoints.
        command.add_argument("--run", type=Path, required=True)  # Identifies a training run or exported model directory.
        command.add_argument("--output", type=Path)  # Explicit output grouping supports controlled repeated evaluations.
        command.add_argument("--count", type=int, default=64 if name == "sample" else 5000)  # Preview versus primary metric population.
        command.add_argument("--seed", type=int, default=20261016)  # Fixed default generation seed.
        command.add_argument("--nfe", type=int, default=32)  # Outer velocity-call budget, not hidden oscillator steps.
        command.add_argument("--batch-size", type=int, default=32)  # Sampling batching is shared across architectures.
        command.add_argument("--device", default=os.environ.get("OFM_DEVICE", "cuda"))  # Device remains explicit in output metadata.
        if name == "evaluate":  # Test access requires an explicit flag or the frozen study runner.
            command.add_argument("--split", choices=["validation", "test"], default="validation")  # Protects the held-out test phase.
            command.add_argument("--intervention", choices=["none", "identity", "shuffle"], default="none")  # Sensitivity diagnostics.
    schedule = commands.add_parser("study", help="Run or resume the frozen bounded comparison")  # Orchestrated experiment command.
    schedule.add_argument("--root", type=Path, default=Path("runs/study"))  # Stable artifacts permit automatic recovery.
    schedule.add_argument("--phase", choices=["checks", "pilot", "main", "ablations", "evaluate", "all"], default="all")  # Explicit stages.
    schedule.add_argument("--dry-run", action="store_true")  # Inspects the schedule without training or test evaluation.
    schedule.add_argument("--device", default=os.environ.get("OFM_DEVICE", "cuda"))  # Matches local hardware by default.
    reports = commands.add_parser("report", help="Build Markdown reports, figures, and PDFs from recorded artifacts")  # Evidence synthesis.
    reports.add_argument("--root", type=Path, default=Path("runs/study"))  # Selects which experiment group to describe.
    reports.add_argument("--output", type=Path, default=Path("reports"))  # Canonical experiment-report directory.
    reports.add_argument("--no-pdf", action="store_true")  # Allows lightweight text-only inspection when explicitly requested.
    publication = commands.add_parser("publish", help="Export and upload audited artifacts to a public HF collection")  # Explicit release.
    publication.add_argument("--root", type=Path, default=Path("runs/study"))  # Includes all reported runs, not just a best seed.
    publication.add_argument("--namespace", default=os.environ.get("HF_NAMESPACE", "BurnyCoder"))  # User-owned Hub destination.
    publication.add_argument("--dry-run", action="store_true")  # Builds reviewable exports before requiring write authentication.
    return root  # argparse supplies complete installation-independent help and error messages.


def training_arguments(args) -> tuple[TrainConfig, Path | None, bool]:
    """Local: resolve new-run or resume settings. Global: omitted options cannot overwrite a previous scientific configuration."""
    if args.resume and args.config:  # A run cannot simultaneously restore state and request an independent new configuration.
        raise ValueError("Use either --resume or --config, not both")  # Prevent ambiguous scientific settings.
    if args.resume:  # Saved settings are authoritative for an existing experiment.
        config = TrainConfig.from_dict(json.loads((args.resume / "config.json").read_text()))  # Reconstructs the prior run.
        overrides = {key: value for key, value in {"minutes": args.minutes, "max_steps": args.steps}.items() if value is not None}
        return replace(config, **overrides), args.resume, True  # Only explicit stopping-limit changes are accepted here.
    if args.config:  # The study runner supplies the complete frozen configuration instead of relying on current .env defaults.
        config = TrainConfig.from_dict(json.loads(args.config.read_text()))  # Validate every serialized scientific setting.
        return config, args.run_dir, False  # No other CLI defaults silently overwrite explicit configuration contents.
    channels, size = (1, 28) if args.dataset == "mnist" else (3, 32)  # Native dataset dimensions are part of the model contract.
    model = ModelConfig(family=args.model, channels=channels, size=size, width=args.width, depth=args.depth,
                        inner_steps=args.inner_steps)  # Defaults retain the full pre-specified architecture.
    config = TrainConfig(model=model, dataset=args.dataset, seed=args.seed, device=args.device or os.environ.get("OFM_DEVICE", "cuda"),
                         batch_size=args.batch_size or int(os.environ.get("OFM_BATCH_SIZE", "128")),
                         microbatch=args.microbatch or int(os.environ.get("OFM_MICROBATCH", "32")),
                         minutes=args.minutes if args.minutes is not None else 45, max_steps=args.steps or 0,
                         data_root=os.environ.get("OFM_DATA_ROOT", "data"), runs_root=os.environ.get("OFM_RUNS_ROOT", "runs"),
                         budget_hours=float(os.environ.get("OFM_BUDGET_HOURS", "24")))  # Serializes every effective environment default.
    return config, args.run_dir, False  # New runs get a timestamped directory unless explicitly named.


def dispatch() -> None:
    """Local: dispatch a configured phase. Global: every command runs inside the shared complete terminal transcript."""
    args = parser().parse_args()  # Validates syntax before downloading data or allocating compute.
    from ofm import pipeline  # Imports the heavy numerical stack only after parsing and cache configuration.
    if args.command == "train":  # Named phase calls keep implementation details below this wrapper.
        config, directory, resume = training_arguments(args)  # Resolves and validates all effective settings.
        result = pipeline.train(config, directory, resume)  # Executes the complete data-to-checkpoint workflow.
    elif args.command == "sample":  # Standalone generation reuses the same checkpoint and solver contract.
        result = pipeline.sample(args.run, args.output, args.count, args.seed, args.nfe, args.batch_size, args.device)
    elif args.command == "evaluate":  # Evaluation receives an explicit split and intervention choice.
        result = pipeline.evaluate(source=args.run, output=args.output, count=args.count, seed=args.seed, nfe=args.nfe,
                                   batch_size=args.batch_size, device=args.device, split=args.split, intervention=args.intervention)
    elif args.command == "study":  # The study runner owns frozen schedules and resumptions.
        result = pipeline.study(root=args.root, phase=args.phase, dry_run=args.dry_run, device=args.device)
    elif args.command == "report":  # Report generation never trains or selects favorable checkpoints.
        result = pipeline.report(root=args.root, output=args.output, pdf=not args.no_pdf)
    else:  # argparse guarantees the remaining command is the explicit publication action.
        result = pipeline.publish(root=args.root, namespace=args.namespace, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, allow_nan=False))  # Returns complete structured results without truncation.


def main() -> None:
    """Local: configure caches and transcript capture. Global: runtime output remains auditable after process exit."""
    load_dotenv(Path.cwd() / ".env")  # Does not search parent directories or unrelated projects.
    os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache/huggingface"))  # Keeps optional Hub authentication local.
    os.environ.setdefault("TORCH_HOME", str(Path.cwd() / ".cache/torch"))  # Keeps evaluator downloads in the checkout.
    from ofm.terminal import capture_terminal  # Imports after local environment defaults are established.
    with capture_terminal(Path(os.environ.get("OFM_RUNS_ROOT", "runs"))):  # No text truncation in the persistent command log.
        dispatch()  # Keeps CLI parsing, errors, and complete final results within one auditable invocation.


if __name__ == "__main__":  # Supports python -m ofm.cli as well as the installed ofm executable.
    main()  # Uses the same parsing and orchestration path in both entry modes.
