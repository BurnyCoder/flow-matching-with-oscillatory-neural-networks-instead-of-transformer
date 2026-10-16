"""Local: frozen-schedule contracts. Global: prevent accidental extra seeds, unequal training budgets, or adaptive order.

Source: https://docs.python.org/3/library/random.html; design rationale is in docs/protocol.md.
"""

import json  # Writes an explicit pilot-freeze fixture without any model training.

from ofm.config import ModelConfig, TrainConfig  # Uses the actual serialized scientific configuration contract.
from ofm.study import build_schedule, scheduled_config  # Tests production scheduling and frozen-setting reconstruction.


def test_schedule_preserves_all_predeclared_arms():
    """Local: inspect every planned run. Global: the study cannot silently replace missing seeds with favorable results."""
    schedule = build_schedule()  # No metrics or training results are available when order is generated.
    assert schedule == build_schedule()  # A resumed invocation reconstructs the same randomized order.
    main = [entry for entry in schedule if entry["phase"] == "main"]  # Primary experiment population.
    assert len(main) == 18 and sum(entry["minutes"] for entry in main) == 810  # Exactly 13.5 training hours.
    for seed in range(101, 107):  # Pairing uses six explicitly named independent initialization groups.
        assert {entry["family"] for entry in main if entry["seed"] == seed} == {"onn", "dit", "conv"}  # Complete blocks.
    ablations = [entry for entry in schedule if entry["phase"] == "ablations"]  # Trained controls have a separate allocation.
    assert len(ablations) == 4 and sum(entry["minutes"] for entry in ablations) == 180  # Exactly three training hours.


def test_comparison_inherits_frozen_pilot_settings(tmp_path, monkeypatch):
    """Local: change environment after freeze. Global: comparison settings cannot drift through a later .env edit."""
    config = TrainConfig(model=ModelConfig(channels=1, size=28), dataset="mnist", device="cpu")  # Recorded pilot reference.
    (tmp_path / "freeze.json").write_text(json.dumps({"pilot_configs": {"pilot/mnist-onn-s101": config.to_dict()}}))
    monkeypatch.setenv("OFM_BATCH_SIZE", "16")  # This later environment change must not alter the frozen study.
    entry = {"phase": "ablations", "dataset": "cifar10", "seed": 102, "family": "no_frequency", "minutes": 45}
    result = scheduled_config(tmp_path, entry, "cpu")  # Reconstruct the comparison's actual saved configuration.
    assert result.batch_size == 128 and result.microbatch == 32  # Preserves the observed pilot optimization settings.
    assert result.model.channels == 3 and result.model.size == 32 and result.seed == 102  # Only declared domain/seed changes.
    assert result.model.family == "no_frequency" and result.minutes == 45  # Applies the scheduled experimental intervention.
