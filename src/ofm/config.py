"""Local: validated experiment settings. Global: serialize every choice needed to reproduce a run.

Sources: https://docs.python.org/3/library/dataclasses.html;
https://bbc2.github.io/python-dotenv/; numerical defaults are pre-specified in docs/protocol.md.
"""

from dataclasses import asdict, dataclass, field  # Standard-library immutable-description helpers avoid a config framework.
from pathlib import Path  # Keeps paths portable within the checkout.


@dataclass
class ModelConfig:
    """Local: architecture parameters. Global: the same description reconstructs every checkpoint."""

    family: str = "onn"  # Selects the backbone or a pre-specified trained ablation.
    channels: int = 3  # CIFAR-10 is RGB; MNIST overrides this to one.
    size: int = 32  # Native image side length, without a learned image encoder.
    width: int = 128  # Total scalar feature width, partitioned into rotating vectors.
    depth: int = 4  # Counts independently parameterized network blocks.
    patch: int = 4  # Reduces spatial cost while predicting every original pixel.
    rotating_dim: int = 4  # Each oscillator lies on the unit sphere in four dimensions.
    inner_steps: int = 4  # Counts hidden updates per block, distinct from sampling NFE.
    step_size: float = 0.1  # Fixed projected-Euler step, frozen for the initial comparison.
    ffn_hidden: int = 0  # Zero requests deterministic transformer parameter matching.

    def __post_init__(self):
        """Local: reject invalid shapes. Global: prevent silent changes to the intended architecture."""
        if self.family not in {"onn", "dit", "conv", "no_coupling", "no_frequency"}:  # Only implemented arms are accepted.
            raise ValueError(f"Unknown backbone: {self.family}")  # Fail before allocating model/data resources.
        if self.width < 8 or self.width % 4 or self.width % self.rotating_dim:  # Attention and vector groups must divide width.
            raise ValueError("width must be >=8 and divisible by four and rotating_dim")  # Explains the shape contract.
        if self.rotating_dim < 2 or self.size % self.patch or min(self.depth, self.inner_steps) < 1:  # Validate recurrence/grid.
            raise ValueError("Invalid rotating_dim, patch grid, depth, or inner_steps")  # Refuse an invalid experiment.
        if self.step_size <= 0:  # Internal evolution must advance by a positive step.
            raise ValueError("step_size must be positive")  # Avoid an accidentally disabled network.


@dataclass
class TrainConfig:
    """Local: one training run's settings. Global: frozen metadata supports paired, resumable experiments."""

    model: ModelConfig = field(default_factory=ModelConfig)  # Avoids sharing a mutable architecture instance.
    dataset: str = "cifar10"  # Chooses an official torchvision dataset.
    seed: int = 101  # Pairing key, distinct from internally derived RNG streams.
    device: str = "cuda"  # Explicitly requested device; unavailable CUDA is an error, not a silent CPU run.
    batch_size: int = 128  # Number of examples contributing to each optimizer update.
    microbatch: int = 32  # Bounds activation memory while preserving the effective batch.
    learning_rate: float = 3e-4  # Constant, pre-specified AdamW learning rate.
    weight_decay: float = 0.01  # Shared regularization across compared arms.
    clip_grad: float = 1.0  # Prevents single updates with unbounded gradient norm.
    ema_decay: float = 0.999  # Sampling uses this exponential average of model weights.
    minutes: float = 45.0  # Cumulative training duration, including checkpoints/validation within the loop.
    max_steps: int = 0  # Nonzero enables explicit development checks, never the primary comparison.
    validation_size: int = 5000  # The pre-specified holdout from the official training split.
    data_root: str = "data"  # Keeps downloaded datasets inside the checkout.
    runs_root: str = "runs"  # Stores local logs, state, and generated outputs.
    budget_hours: float = 24.0  # Cumulative cap shared by all GPU operations.
    log_every: int = 50  # Flushes progress frequently without dumping tensors as text.
    checkpoint_seconds: float = 300.0  # Protects long runs against interruptions.
    amp: bool = True  # BF16 is enabled only on CUDA; oscillator geometry stays FP32.

    def __post_init__(self):
        """Local: check run invariants. Global: preserve meaningful pairing and bounded execution."""
        if self.dataset not in {"mnist", "cifar10"}:  # The initial experiment has exactly two datasets.
            raise ValueError("dataset must be mnist or cifar10")  # Reject unsupported implicit preprocessing.
        if self.batch_size < 1 or self.microbatch < 1 or self.batch_size % self.microbatch:  # Exact gradient accumulation.
            raise ValueError("batch_size must be a positive multiple of microbatch")  # Keeps batch size honest.
        if self.minutes <= 0 or self.budget_hours <= 0 or self.max_steps < 0:  # Training and reservation limits are positive.
            raise ValueError("Invalid training duration, step count, or budget")  # Avoids unbounded accidental runs.
        if not 0 <= self.ema_decay < 1 or self.learning_rate <= 0:  # Validate optimizer/EMA behavior.
            raise ValueError("Invalid EMA decay or learning rate")  # Makes errors visible before training.

    @property
    def budget_path(self) -> Path:
        """Local: locate the shared ledger. Global: restarts and different runs cannot reset compute use."""
        return Path(self.runs_root) / "budget.json"  # One ledger per explicitly configured experiment root.

    def to_dict(self) -> dict:
        """Local: flatten nested dataclasses. Global: emit complete JSON-compatible configuration."""
        return asdict(self)  # Preserves every default and override in the artifact.

    @classmethod
    def from_dict(cls, value: dict):
        """Local: rebuild a serialized config. Global: resume with the original scientific settings."""
        values = dict(value)  # Avoids mutating the loaded checkpoint dictionary.
        values["model"] = ModelConfig(**values["model"])  # Restores architecture validation as well as its fields.
        return cls(**values)  # Runs training-setting validation on deserialization.
