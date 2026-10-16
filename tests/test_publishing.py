"""Local: portable release contracts. Global: every exported reported generator must reload without private source or credentials.

Sources: https://huggingface.co/docs/safetensors/index;
https://huggingface.co/docs/huggingface_hub/en/guides/upload.
"""

import json  # Writes explicit local configuration fixtures.

import torch  # Constructs a small deterministic generator without real training or Hub access.

from ofm.checkpoints import load_generator  # Same reconstruction path used by the sampling CLI.
from ofm.config import ModelConfig, TrainConfig  # Keeps the test's model contract explicit.
from ofm.models.factory import build_model  # Production architecture constructor.
from ofm.publishing import export_run  # The export implementation must enforce its own file allowlist.


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
