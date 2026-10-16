"""Local: safe checkpoint loading. Global: training, exported models, and CPU sampling share reconstruction semantics.

Sources: https://docs.pytorch.org/docs/stable/notes/serialization.html;
https://huggingface.co/docs/safetensors/index.
"""

import json  # Loads portable model descriptions without executing source code.
from pathlib import Path  # Supports either a run directory or a direct checkpoint path.

import torch  # Loads only tensor/primitive training state.
from safetensors.torch import load_file  # Portable exports contain no executable pickle objects.

from ofm.config import TrainConfig  # Rebuilds the complete validated experiment configuration.
from ofm.models.factory import build_model  # Reuses the same architecture constructor as training.


def load_generator(source: Path, device: str = "cpu"):
    """Local: reconstruct EMA weights. Global: generation uses the same model after local resume or Hub download."""
    source = Path(source)  # Accepts pathlib or ordinary CLI strings.
    if source.is_dir() and (source / "model.safetensors").exists():  # Published export format is executable-code-free.
        config = TrainConfig.from_dict(json.loads((source / "config.json").read_text()))  # Restores all recorded settings.
        weights = load_file(source / "model.safetensors", device="cpu")  # Delays GPU allocation until explicitly requested.
    else:  # Local training checkpoints additionally contain optimizer and RNG state.
        checkpoint = source / "checkpoint.pt" if source.is_dir() else source  # Resolves the ordinary run directory contract.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)  # Rejects arbitrary unpickled Python objects.
        config, weights = TrainConfig.from_dict(state["config"]), state["ema_model"]  # Evaluation always uses EMA weights.
    model = build_model(config.model)  # Recreates the resolved baseline width as well as oscillator parameters.
    model.load_state_dict(weights, strict=True)  # Missing or extra tensors are incompatible, not silently ignored.
    return model.to(device).eval(), config  # No dropout or mutable oscillator state is retained for sampling.
