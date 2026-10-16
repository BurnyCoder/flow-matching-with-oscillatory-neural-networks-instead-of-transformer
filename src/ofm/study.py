"""Local: immutable run ordering and phase orchestration. Global: execute a bounded comparison without adaptive seed selection.

Sources: https://docs.python.org/3/library/subprocess.html;
https://docs.python.org/3/library/random.html; scientific decisions are pre-specified in docs/protocol.md.
"""

import json  # Restores schedules, freeze records, and completed-run summaries.
import random  # Generates run ordering from one fixed RNG unrelated to training.
import subprocess  # Isolates model processes so GPU allocations are released between arms.
import sys  # Reuses the local environment rather than an unrelated system executable.
from pathlib import Path  # Organizes phase/seed/model artifacts under one study root.

from ofm.artifacts import (  # Shared durable provenance.
    event,  # Records phase transitions.
    file_hash,  # Freezes the protocol identity.
    git_revision,  # Associates invocations with source history.
    setup_logging,  # Separates orchestration logs from training logs.
    timestamp,  # Names every command attempt.
    write_json,  # Commits schedules and results atomically.
)


def build_schedule() -> list[dict]:
    """Local: list all training arms before outcomes exist. Global: equal time budgets and randomized order survive resumes."""
    generator, schedule = random.Random(20261016), []  # Fixed execution-order stream is separate from every model RNG.
    for phase, dataset, seeds, families, minutes in [  # Explicit allocation, without outcome-dependent additions.
        ("pilot", "mnist", [101], ["onn", "dit", "conv"], 20),  # One hour leaves pilot allocation for numerical checks.
        ("main", "cifar10", range(101, 107), ["onn", "dit", "conv"], 45),  # Six complete paired-seed groups.
        ("ablations", "cifar10", [101, 102], ["no_coupling", "no_frequency"], 45),  # Descriptive two-seed causal removals.
    ]:
        for seed in seeds:  # Keep paired seeds adjacent to reduce unrelated long-term hardware changes.
            order = generator.sample(families, len(families))  # Randomizes architecture order within each seed group.
            for family in order:  # Every declared arm appears once, irrespective of its future metric.
                schedule.append({"phase": phase, "dataset": dataset, "seed": seed, "family": family, "minutes": minutes,
                                 "relative_path": f"{phase}/{dataset}-{family}-s{seed}"})  # Stable resume identity.
    return schedule  # Contains no trained outputs, validation values, or test-set information.


def run_command(arguments: list[str], directory: Path) -> None:
    """Local: execute a phase in a fresh process. Global: failed attempts remain recorded and never masquerade as completed arms."""
    directory.mkdir(parents=True, exist_ok=True)  # Retains invocation evidence even if the child fails before initialization.
    command = [sys.executable, *arguments]  # Structured arguments avoid shell expansion and credential interpolation.
    write_json(directory / f"invocation-{timestamp()}.json", {"argv": command, "revision": git_revision()})  # Complete public settings.
    result = subprocess.run(command, check=False)  # Child CLI streams and captures its complete output in real time.
    if result.returncode:  # A killed budget worker or numerical failure is not a valid experimental result.
        write_json(directory / f"failure-{timestamp()}.json", {"returncode": result.returncode, "argv": command})  # Append failure evidence.
        raise RuntimeError(f"Study command failed with exit {result.returncode}: {directory}")  # Stop for diagnosis before further spending.


def train_entry(root: Path, entry: dict, device: str) -> dict:
    """Local: complete one scheduled budget. Global: rerunning a phase resumes unfinished work and skips completed arms."""
    directory = root / entry["relative_path"]  # One immutable location per predeclared run.
    summary_path = directory / "training.json"  # Completion is determined from persisted evidence, not directory existence.
    if summary_path.exists():  # A previous invocation may have already completed this scheduled arm.
        summary = json.loads(summary_path.read_text())  # Preserve its original measurements without retraining.
        if summary["status"] == "completed":  # Step-limited development runs are deliberately not treated as complete.
            return summary  # Skipping cannot extend training based on an observed score.
    if (directory / "checkpoint.pt").exists():  # Resume only when a complete optimizer checkpoint is available.
        arguments = ["--resume", str(directory), "--minutes", str(entry["minutes"]), "--steps", "0"]  # Cumulative time limit.
    else:  # A fresh scheduled run gets explicit scientific settings.
        arguments = ["--dataset", entry["dataset"], "--model", entry["family"], "--seed", str(entry["seed"]),
                     "--minutes", str(entry["minutes"]), "--run-dir", str(directory), "--device", device]  # Defaults are protocol-fixed.
    run_command(["-m", "ofm.cli", "train", *arguments], directory)  # Releases all model/optimizer GPU memory on child exit.
    summary = json.loads(summary_path.read_text())  # The child writes its actual completion status.
    if summary["status"] != "completed":  # Budget exhaustion cannot silently become a final 45-minute comparison.
        raise RuntimeError(f"Incomplete scheduled arm: {directory}: {summary['status']}")  # Preserve and report partial results.
    return summary  # All arms use this identical orchestration path.


def evaluate_entry(directory: Path, output: Path, device: str, *, count=5000, nfe=32, split="test", intervention="none"):
    """Local: run one immutable metric attempt. Global: never regenerate samples to select a more favorable score."""
    if (output / "evaluation.json").exists():  # Completed evaluations are reused verbatim on study resume.
        return  # Repeated orchestration cannot select a different generated population.
    run_command(["-m", "ofm.cli", "evaluate", "--run", str(directory), "--output", str(output), "--count", str(count),
                 "--nfe", str(nfe), "--split", split, "--intervention", intervention, "--device", device], output)  # Same metric seed.


def run_study(root: Path, phase: str = "all", dry_run: bool = False, device: str = "cuda") -> dict:
    """Local: execute checks, pilots, frozen comparisons, and metrics. Global: users can recover the full workflow with one command."""
    root, schedule = Path(root), build_schedule()  # Deterministic ordering is independent of existing outcomes.
    if dry_run:  # Inspecting the planned allocation performs no download, training, or test evaluation.
        return {"root": str(root), "phase": phase, "schedule": schedule, "gpu_cap_hours": 24}  # Complete reviewable schedule.
    setup_logging(root)  # Records phase-level progress separately from each model's training log.
    schedule_path = root / "schedule.json"  # Frozen before any scheduled optimizer updates.
    if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule:  # Reject changes hidden behind a resume command.
        raise ValueError("Study schedule differs from the existing frozen schedule")  # Requires an explicit protocol revision.
    write_json(schedule_path, schedule)  # Atomic and idempotent when the schedule is unchanged.
    phases = ["checks", "pilot", "main", "ablations", "evaluate"] if phase == "all" else [phase]  # Preserves scientific phase order.
    for selected in phases:  # Each phase can also be resumed independently through the public CLI.
        event("study_phase_start", phase=selected, root=str(root))  # Meaningful live phase transition.
        if selected == "checks":  # Numerical and fixed-batch learning gates precede expensive comparisons.
            destination = root / "checks/fixed-batch"  # A clearly labeled development experiment.
            result_path = destination / "diagnostic.json"  # Completed checks are not rerun unnecessarily.
            if not result_path.exists() or not json.loads(result_path.read_text())["passed"]:  # Preserve failures before reattempts.
                run_command(["-m", "ofm.diagnostics", "--output", str(destination), "--device", device], destination)
        elif selected in {"pilot", "main", "ablations"}:  # Optimization phases all use the same resumable run helper.
            if selected != "pilot" and not (root / "freeze.json").exists():  # Final comparisons cannot begin before pilot completion.
                raise ValueError("Complete the MNIST pilot to freeze the study before CIFAR-10 training")  # No test-driven tuning.
            for entry in [item for item in schedule if item["phase"] == selected]:  # Preserve randomized order, including on resume.
                train_entry(root, entry, device)  # Completes only the declared cumulative duration.
                if selected == "pilot":  # Pilot decisions use validation diagnostics, never official test metrics.
                    directory = root / entry["relative_path"]  # Reuses the just-completed final pilot checkpoint.
                    evaluate_entry(directory, directory / "evaluations/validation", device, count=512, split="validation")  # Fixed pilot view.
            if selected == "pilot" and not (root / "freeze.json").exists():  # Freeze once, before any CIFAR outcome exists.
                configs = {entry["relative_path"]: json.loads((root / entry["relative_path"] / "config.json").read_text())
                           for entry in schedule if entry["phase"] == "pilot"}  # Capture actual resolved pilot settings.
                write_json(root / "freeze.json", {"timestamp": timestamp(), "revision": git_revision(), "pilot_configs": configs,
                                                 "protocol_sha256": file_hash(Path("docs/protocol.md")), "changes": []})  # No tuning assumed.
        elif selected == "evaluate":  # Official test metrics are generated only after fixed-budget training.
            if not (root / "freeze.json").exists():  # Preserve the boundary between pilot selection and final assessment.
                raise ValueError("Test study evaluation requires the frozen pilot record")  # Explicitly rejects premature test use.
            from ofm.study_evaluation import evaluate_study  # Additional controls belong in their own focused module.
            evaluate_study(root, schedule, device)  # Primary metrics, cost curves, ablations, and sensitivity controls.
        else:  # Programmatic callers receive the same invalid-phase protection as argparse users.
            raise ValueError(f"Unknown study phase: {selected}")  # Never silently skip an unsupported phase.
        event("study_phase_complete", phase=selected, root=str(root))  # Marks a completed phase rather than merely a launched process.
    return {"root": str(root), "completed_phases": phases, "schedule": str(schedule_path)}  # Concrete recovery/artifact locations.
