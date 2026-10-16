"""Local: explicit artifact export and Hub publication. Global: release every reported checkpoint without credentials or dataset caches.

Sources: https://huggingface.co/docs/huggingface_hub/en/guides/upload;
https://huggingface.co/docs/huggingface_hub/en/guides/collections; https://huggingface.co/docs/safetensors/index.
"""

import json  # Reads scientific metadata and creates auditable release manifests.
import shutil  # Copies only explicitly permitted experiment artifacts.
from pathlib import Path  # Resolves local export destinations independently from Hub repository identifiers.

from huggingface_hub import (  # Uses supported authenticated Hub operations.
    HfApi,  # Creates and uploads public model repositories.
    add_collection_item,  # Groups every released domain in one collection.
    create_collection,  # Creates or reuses the explicitly named study collection.
)
from safetensors.torch import save_file  # Exports non-executable inference weights.

from ofm.artifacts import (  # Shared provenance and hashes.
    event,  # Emits complete public release progress.
    file_hash,  # Verifies artifact identity.
    git_revision,  # Connects exports to source code.
    setup_logging,  # Captures release diagnostics without auth values.
    timestamp,  # Isolates fresh export attempts.
    write_json,  # Persists reviewable manifests.
)
from ofm.checkpoints import load_generator  # Reconstructs the same EMA generator used in reported evaluations.

REPOSITORY = "https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer"  # Public code provenance.
ARTIFACT_NAMES = {"checkpoint.pt", "config.json", "training.json", "diagnostic.json", "environment.json", "data_manifest.json",
                  "run.log", "evaluation.json", "evaluation_config.json", "failure.json", "sampling.json", "control.json",
                  "samples.safetensors", "fixed_batch.safetensors", "grid.png"}  # Positive allowlist, never a broad directory upload.


def write_manifest(directory: Path) -> dict:
    """Local: hash every exported file. Global: downloaded reports and checkpoints can be independently integrity-checked."""
    manifest = {path.relative_to(directory).as_posix(): {"sha256": file_hash(path), "bytes": path.stat().st_size}
                for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.json"}  # Exclude self-reference.
    write_json(directory / "manifest.json", manifest)  # A manifest describes actual existing bytes, not expected filenames.
    return manifest  # Callers can inspect total size and every public artifact before upload.


def export_run(source: Path, target: Path) -> dict:
    """Local: package one checkpoint and its evidence. Global: portable exports reload identically and exclude unrelated local files."""
    source, target = Path(source), Path(target)  # Accept CLI-derived and test paths consistently.
    target.mkdir(parents=True, exist_ok=True)  # The caller chooses an isolated timestamped export root.
    original_hash = file_hash(source / "checkpoint.pt")  # Guard against exporting an actively changing checkpoint.
    model, config = load_generator(source, "cpu")  # No GPU compute or inference service is needed to build a release.
    for path in sorted(source.rglob("*")):  # Inspect only files belonging to this explicitly selected experiment.
        if path.is_file() and path.name in ARTIFACT_NAMES:  # Whitelist excludes .env, tokens, source data, and miscellaneous files.
            if path.is_symlink():  # A symlink must not smuggle an unrelated credential or dataset into a public export.
                raise ValueError(f"Refusing symbolic link in export: {path}")  # Keep the intended artifact boundary explicit.
            destination = target / path.relative_to(source)  # Preserve evaluation grouping and original filenames.
            destination.parent.mkdir(parents=True, exist_ok=True)  # Create nested sampling/evaluation artifact directories.
            shutil.copy2(path, destination)  # Copy only authorized files, never caches or whole source datasets.
    if file_hash(target / "checkpoint.pt") != original_hash:  # An atomic training update during export changes file identity.
        raise RuntimeError(f"Checkpoint changed during export: {source}")  # Retry after the run is stable; never publish mismatched weights.
    weights = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}  # Safe flat tensor mapping.
    save_file(weights, target / "model.safetensors")  # Inference users need no optimizer pickle or private code objects.
    write_json(target / "config.json", config.to_dict())  # Resolved architecture exactly reconstructs the exported state.
    status_path = source / "training.json" if (source / "training.json").exists() else source / "diagnostic.json"  # Scientific role.
    result = json.loads(status_path.read_text()) if status_path.exists() else {"status": "incomplete"}  # Never infer successful training.
    card = f"""---
license: mit
pipeline_tag: unconditional-image-generation
tags:
- pytorch
- flow-matching
- oscillatory-neural-networks
---
# {config.dataset} / {config.model.family} / seed {config.seed}

This is an unconditional pixel-space velocity network from the [oscillatory flow matching study]({REPOSITORY}).
Recorded status: **{result.get('status', 'incomplete')}**. The model is a research artifact; sample quality and limitations are documented in its accompanying evaluations and the public reports. No performance advantage is assumed.

## Loading and sampling

Install the locked environment from the source repository. Download this entire run subdirectory, including `config.json` and `model.safetensors`, then use:

```bash
ofm sample --run /path/to/downloaded/run --device cpu --count 64 --nfe 32
```

The public Python loader is `ofm.checkpoints.load_generator(path, device="cpu")`; models accept `forward(images, times)` and return image-shaped velocities. The safe inference weights are the same EMA weights used for reported generation. `checkpoint.pt` additionally preserves the available training/diagnostic state; ordinary training checkpoints are resumable with `ofm train --resume /path/to/run` after downloading the official dataset.

## Training and evaluation

Dataset: {config.dataset}; architecture: {config.model.family}; seed: {config.seed}. This is a from-scratch generator, with no text conditioning or pretrained image encoder. The exact architecture, optimizer, data manifests, source revisions, and metrics are included as JSON. Generated tensors are complete and unclipped; PNG grids show the first fixed samples after display clipping. KID/FID are CIFAR diagnostics and are not classification accuracy. Untrained, fixed-batch, or step-limited checkpoints are explicitly labeled and must not be treated as final-budget primary results.

## Limitations and provenance

Small native-resolution datasets and a short local compute budget limit image quality and external generalization. Six paired seeds are exploratory; inference interventions alone do not establish an oscillator mechanism. Dataset content and evaluator biases remain relevant. See the [protocol]({REPOSITORY}/blob/main/docs/protocol.md) and [reports]({REPOSITORY}/tree/main/reports) for all results, including failures. This export contains no downloaded dataset or credentials. `manifest.json` records file hashes; the original checkpoint SHA-256 is `{original_hash}`. Export code revision: `{git_revision()}`.
"""  # Model card states its actual role and reconstruction contract without fabricated evaluation claims.
    (target / "README.md").write_text(card, encoding="utf-8")  # Each checkpoint remains understandable outside its parent collection.
    manifest = write_manifest(target)  # Hash only after every artifact and model card has been written.
    return {"source": str(source), "target": str(target), "dataset": config.dataset, "family": config.model.family,
            "seed": config.seed, "status": result.get("status"), "files": len(manifest),
            "bytes": sum(value["bytes"] for value in manifest.values())}  # Concrete reviewable export summary.


def publish_study(root: Path, namespace: str = "BurnyCoder", dry_run: bool = False) -> dict:
    """Local: export, authenticate, upload, collect. Global: publication is explicit and includes all reported models, not a best seed."""
    root, destination = Path(root), Path("exports") / timestamp()  # Fresh directory prevents stale or unrelated files entering a release.
    setup_logging(destination)  # Capture release progress without logging token values or raw authentication responses.
    exports = []  # Preserve every reportable checkpoint selected under the requested root.
    for checkpoint in sorted(root.rglob("checkpoint.pt")):  # Only the current requested experiment tree is inspected.
        source = checkpoint.parent  # Each checkpoint's adjacent records establish its scientific role.
        if not (source / "training.json").exists() and not (source / "diagnostic.json").exists():  # Active unreported runs are unstable.
            continue  # They will be included after a completed, failed, or diagnostic result is recorded.
        config = json.loads((source / "config.json").read_text())  # Determine destination without executing checkpoint contents.
        target = destination / config["dataset"] / source.relative_to(root)  # Group all seeds and controls into dataset repositories.
        exports.append(export_run(source, target))  # Complete concrete export before authentication or upload.
    if not exports:  # An empty release must not create misleading public repositories.
        raise ValueError(f"No reported checkpoints to publish under {root}")  # Exposes missing training/report artifacts.
    datasets = sorted({entry["dataset"] for entry in exports})  # All measured domains receive a public model repository.
    for dataset in datasets:  # Finalize complete per-domain bundles before requesting any network write.
        directory = destination / dataset  # Hub upload never sees files outside this explicit export tree.
        paths = [Path(entry["target"]).relative_to(directory).as_posix() for entry in exports if entry["dataset"] == dataset]
        listing = "\n".join(f"- [{path}]({path}/README.md)" for path in paths)  # List every included seed/control, without ranking.
        (directory / "README.md").write_text(f"---\nlicense: mit\npipeline_tag: unconditional-image-generation\n---\n"
            f"# Oscillatory flow matching: {dataset}\n\nAll reported checkpoints and evidence from [the source study]({REPOSITORY}). "
            "Weights are in the run subdirectories below. Each has a model card, exact config, safe inference weights, "
            "original checkpoint, logs, and artifact manifest. Consult each status before interpreting results.\n\n" + listing + "\n", encoding="utf-8")
        for name in ["schedule.json", "freeze.json", "budget.json"]:  # Include available study-level provenance without arbitrary JSON copying.
            matches = [root / name, root / "study" / name, root.parent / name]  # Supports study-only or all-runs publication roots.
            existing = next((path for path in matches if path.exists()), None)  # Prefer provenance closest to the requested root.
            if existing is not None:  # Do not fabricate a freeze or budget file for development-only exports.
                shutil.copy2(existing, directory / name)  # Public records contain settings/accounting, not credentials.
        report_root = Path("reports")  # Public reports are an explicit independent artifact source, never arbitrary user files.
        report_names = {"paper.md", "paper.pdf", "experiments.md", "experiments.pdf", "analysis.json",
                        "planning-power.md", "planning-power.pdf", "planning-power.json"}  # Complete synthesis and planning evidence.
        for path in report_root.rglob("*"):  # Include generated report sources and figures from all reported experiment groups.
            allowed = path.name in report_names or (path.parent.name in {"experiments", "figures"} and path.suffix in {".md", ".pdf", ".png"})
            if path.is_file() and not path.is_symlink() and allowed:  # Positive format/location policy excludes caches and secrets.
                report_target = directory / "reports" / path.relative_to(report_root)  # Keep report links portable within the release.
                report_target.parent.mkdir(parents=True, exist_ok=True)  # Preserve experiment and figure grouping.
                shutil.copy2(path, report_target)  # Report bundles contain no original dataset downloads.
        write_manifest(directory)  # A root manifest covers every nested artifact except self-referential manifests.
    result = {"export_root": str(destination), "exports": exports, "dry_run": dry_run,
              "repositories": [f"{namespace}/oscillatory-flow-matching-{dataset}" for dataset in datasets]}  # Reviewable plan.
    write_json(destination / "publication.json", result)  # Preserve the exact prepared release even if authentication fails.
    if dry_run:  # Preparation and verification need no Hub write scope.
        return result  # User can inspect real weights, cards, counts, and hashes rather than a hypothetical release plan.
    api = HfApi()  # Uses supported Hub authentication discovery; never prints or stores tokens in scientific artifacts.
    identity = api.whoami()  # Verify the destination account before public writes.
    if identity["name"] != namespace:  # Prevent accidental publication to an unexpected personal namespace.
        raise ValueError(f"Authenticated Hub identity must be {namespace}")  # Report only the expected identity, not auth details.
    collection = create_collection("Oscillatory flow matching — local exploratory study", namespace=namespace, private=False,
                                   description="All reported MNIST/CIFAR-10 models, paired seeds, controls, and reproducible evidence.",
                                   exists_ok=True)  # One user-owned collection groups the complete release.
    result["collection"] = collection.url  # Save the concrete public destination returned by the Hub.
    for dataset, repo_id in zip(datasets, result["repositories"], strict=True):  # No best-model selection during upload.
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)  # User explicitly authorized public publication.
        commit = api.upload_folder(repo_id=repo_id, folder_path=destination / dataset, repo_type="model",
                                   commit_message="Publish complete reported oscillator-flow study artifacts")  # Allowlisted tree only.
        add_collection_item(collection.slug, item_id=repo_id, item_type="model", exists_ok=True)  # Links every dataset repository.
        event("hub_repository_published", repository=repo_id, commit_url=commit.commit_url)  # Public provenance only, no credential values.
    write_json(destination / "publication.json", result)  # Mark actual successful collection/repository destinations.
    return result  # Final callers can link directly to usable public model artifacts.
