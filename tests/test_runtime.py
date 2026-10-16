"""Local: persistence contracts. Global: budgets and paired data streams survive interruptions.

Sources: https://docs.python.org/3/library/fcntl.html; https://docs.pytorch.org/docs/stable/random.html.
"""

import json  # Inspects persisted, reviewable ledger state.
import subprocess  # Exercises the hard deadline in an isolated process rather than terminating pytest.
import sys  # Selects the active test environment for the isolated worker.

import pytest  # Checks rejection and intentional failure paths.
import torch  # Supplies deterministic sampling fixtures.

from ofm.budget import BudgetExhausted, BudgetSession  # Public cumulative-budget interface.
from ofm.data import BatchStream, split_indices  # Public reproducible data preparation interfaces.


def test_budget_refunds_and_persists(tmp_path):
    """Local: inspect a completed reservation. Global: future runs inherit actual consumed time."""
    path = tmp_path / "budget.json"  # Isolates the test from the real GPU budget.
    with BudgetSession(path, limit_seconds=100, requested_seconds=30, label="test") as session:  # Reserves atomically.
        session.check()  # Confirms there is time for one small operation.
        assert json.loads(path.read_text())["charged_seconds"] == 30  # An abrupt crash cannot erase reserved use.
    ledger = json.loads(path.read_text())  # Reads the finalized, refunded ledger.
    assert 0 <= ledger["charged_seconds"] < 5  # Only elapsed time remains charged on clean exit.
    assert ledger["active"] is None  # Makes subsequent jobs eligible for their own reservation.
    assert len(ledger["history"]) == 1  # Keeps an audit entry rather than overwriting the past.


def test_budget_cannot_be_reset_or_overbooked(tmp_path):
    """Local: refuse unsafe reconfiguration. Global: the hard cap cannot silently increase on resume."""
    path = tmp_path / "budget.json"  # Separate ledger for the failure scenario.
    path.write_text(json.dumps({"limit_seconds": 10, "charged_seconds": 10, "active": None, "history": []}))  # Exhausted.
    with pytest.raises(BudgetExhausted), BudgetSession(path, 10, 2, "exhausted"):  # No allocation remains.
        pytest.fail("exhausted budget was accepted")  # The body must remain unreachable.
    with pytest.raises(ValueError), BudgetSession(path, 20, 2, "reset"):  # Rejects silently increasing the existing cap.
        pytest.fail("budget limit was silently changed")  # Guard against hidden budget renewal.


def test_budget_deadline_terminates_a_blocked_worker(tmp_path):
    """Local: simulate a call without cooperative checks. Global: native metrics cannot ignore the persistent GPU cap."""
    script = ("import time; from pathlib import Path; from ofm.budget import BudgetSession; "
              f"session=BudgetSession(Path({str(tmp_path / 'budget.json')!r}), 1, 1, 'blocked'); "
              "session.__enter__(); time.sleep(20)")  # A sleeping child stands in for an unresponsive numerical call.
    result = subprocess.run([sys.executable, "-c", script], timeout=15, check=False)  # Bounds the test independently too.
    assert result.returncode == -9  # The separate guard delivers SIGKILL rather than relying on Python checkpoints.
    ledger = json.loads((tmp_path / "budget.json").read_text())  # Abrupt exit must retain conservative accounting.
    assert ledger["charged_seconds"] == 1 and ledger["active"] is not None  # No refund for an interrupted reservation.


def test_stream_resume_matches_uninterrupted():
    """Local: checkpoint permutation and cursor. Global: resumed comparisons use identical minibatch prefixes."""
    stream = BatchStream(torch.arange(19), seed=42)  # Small size forces an epoch boundary.
    stream.next_indices(13)  # Leaves six examples before reshuffling.
    state = stream.state_dict()  # Captures both cursor and future random permutations.
    expected = stream.next_indices(12)  # Crosses into the next epoch.
    restored = BatchStream(torch.arange(19), seed=999)  # Deliberately starts from a different state.
    restored.load_state_dict(state)  # Restores the original random stream fully.
    torch.testing.assert_close(restored.next_indices(12), expected)  # Requires exact resume equivalence.


def test_split_is_stratified_disjoint_and_reproducible():
    """Local: partition examples. Global: validation cannot leak into optimizer minibatches."""
    labels = torch.arange(100) % 10  # Ten balanced classes with identifiable rows.
    train, validation = split_indices(labels, validation_size=20, seed=42)  # Reserves two examples per class.
    assert set(train.tolist()).isdisjoint(validation.tolist())  # Prevents index leakage.
    assert len(train) == 80 and len(validation) == 20  # Prevents unintended data loss.
    assert torch.bincount(labels[validation]).tolist() == [2] * 10  # Checks stratification directly.
    again = split_indices(labels, 20, 42)  # Repeats preparation with the same seed.
    torch.testing.assert_close(validation, again[1])  # Ensures reproducible manifests.
