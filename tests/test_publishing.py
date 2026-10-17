"""Local: portable release contracts. Global: every exported reported generator must reload without private source or credentials.

Sources: https://huggingface.co/docs/safetensors/index;
https://huggingface.co/docs/huggingface_hub/en/guides/upload.
"""

import json  # Writes explicit local configuration fixtures.

import torch  # Constructs a small deterministic generator without real training or Hub access.

from ofm.checkpoints import load_generator  # Same reconstruction path used by the sampling CLI.
from ofm.config import ModelConfig, TrainConfig  # Keeps the test's model contract explicit.
from ofm.models.factory import build_model  # Production architecture constructor.
from ofm.publishing import (  # Exercise individual checkpoints and the complete release boundary.
    export_run,  # Reconstructs a single reported generator.
    publish_study,  # Preserves evidence that does not belong to a checkpoint.
)


def test_export_roundtrip_and_secret_exclusion(tmp_path):
    """Local: export/reload safe weights. Global: publish usable artifacts while excluding credentials and unrelated files."""
    source, target = tmp_path / "run", tmp_path / "export"  # Isolated paths prevent accidental publication of real secrets.
    source.mkdir()  # Create only the explicit fixture artifacts.
    config = TrainConfig(model=ModelConfig(channels=1, size=8, width=8, depth=1, inner_steps=1), device="cpu")
    model = build_model(config.model).eval()  # A tiny but genuine oscillator generator.
    torch.save({"config": config.to_dict(), "ema_model": model.state_dict()}, source / "checkpoint.pt")  # Safe tensor-only state.
    (source / "config.json").write_text(json.dumps(config.to_dict()))  # Portable reconstruction metadata.
    (source / ".env").write_text("HF_TOKEN=FAKE_TEST_SECRET")  # The public exporter must never copy this fixture.
    (source / "unrelated.bin").write_bytes(b"not an experiment artifact")  # Broad recursive copying would be unsafe.
    export_run(source, target)  # Build the same allowlisted artifact package used for real reported checkpoints.
    restored, _ = load_generator(target)  # Must load using only the portable directory.
    images, times = torch.randn(2, 1, 8, 8), torch.tensor([0.25, 0.75])  # Check actual predictions, not only state key names.
    torch.testing.assert_close(model(images, times), restored(images, times), rtol=0, atol=0)  # Exact CPU prediction equivalence.
    assert not (target / ".env").exists() and not (target / "unrelated.bin").exists()  # Explicit exclusion, not a warning alone.
    assert (target / "manifest.json").exists() and (target / "README.md").exists()  # Reviewable hashes and model card.


def test_study_export_retains_controls_without_checkpoints(tmp_path, monkeypatch):
    """Local: export standalone control/failure evidence. Global: a complete release must preserve experiments that have no model."""
    monkeypatch.chdir(tmp_path)  # Dry-run exports and logs stay inside this isolated fixture.
    source, control = tmp_path / "runs/model", tmp_path / "runs/controls/real-versus-real"  # Distinct experimental roles.
    source.mkdir(parents=True)  # One reported generator makes this a valid study release.
    control.mkdir(parents=True)  # The real-image control deliberately has no checkpoint.
    config = TrainConfig(model=ModelConfig(channels=1, size=8, width=8, depth=1, inner_steps=1), device="cpu")
    torch.save({"config": config.to_dict(), "ema_model": build_model(config.model).state_dict()}, source / "checkpoint.pt")
    (source / "config.json").write_text(json.dumps(config.to_dict()))  # Complete safe reconstruction metadata.
    (source / "training.json").write_text(json.dumps({"status": "untrained"}))  # Explicitly report the fixture's actual role.
    evidence = {"left_indices": [2, 0], "right_indices": [1, 3], "repeat_max_absolute_difference": 0}  # Full partition evidence.
    (control / "control.json").write_text(json.dumps(evidence))  # Rendering omits large index arrays, so preserve the raw record.
    (control / "failure-first.json").write_text(json.dumps({"error": "fixture failure"}))  # Preserve earlier failed attempts too.
    (control / "run.log").write_text("2026-10-17T00:00:00Z control completed\n")  # Keep the associated timestamped trajectory.
    (control / ".env").write_text("HF_TOKEN=FAKE_TEST_SECRET")  # Sibling credentials must remain outside the allowlist.
    result = publish_study(tmp_path / "runs", dry_run=True)  # Exercise actual release orchestration without Hub credentials.
    target = tmp_path / result["export_root"] / config.dataset  # Inspect the same directory that would be uploaded.
    copied = target / "study-records/controls/real-versus-real"  # Standalone records have an explicit shared evidence location.
    assert json.loads((copied / "control.json").read_text()) == evidence  # No partition indices may be silently dropped.
    assert (copied / "failure-first.json").exists() and (copied / "run.log").exists()  # Retain both outcome and failure provenance.
    assert not (copied / ".env").exists()  # Confirm the repair did not broaden export to arbitrary directory contents.
    manifest = json.loads((target / "manifest.json").read_text())  # Root integrity coverage must include the added evidence.
    assert "study-records/controls/real-versus-real/control.json" in manifest  # Exported controls are hash-verifiable.
