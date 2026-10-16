"""Local: frozen-schedule contracts. Global: prevent accidental extra seeds, unequal training budgets, or adaptive order.

Source: https://docs.python.org/3/library/random.html; design rationale is in docs/protocol.md.
"""

from ofm.study import build_schedule  # Tests the same schedule consumed by real subprocess orchestration.


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
