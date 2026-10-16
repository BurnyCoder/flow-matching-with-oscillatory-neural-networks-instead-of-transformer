"""Local: report completeness checks. Global: absent seed results must remain incomplete rather than imply scientific success.

Source: https://neurips.cc/public/guides/PaperChecklist (claims must match available evidence).
"""

import json  # Writes small explicit result fixtures without running model training.

from ofm.reporting import create_reports  # Tests the same report command used after the real experiment.


def test_empty_study_reports_no_empirical_claim(tmp_path):
    """Local: render an unrun study. Global: generated papers distinguish plans from completed experiments."""
    root, output = tmp_path / "runs", tmp_path / "reports"  # Isolated fixtures contain no measured seed population.
    root.mkdir()  # An empty artifact directory is valid during implementation.
    result = create_reports(root, output, pdf=False)  # Text-only testing avoids unnecessary typesetting work.
    analysis = json.loads((output / "analysis.json").read_text())  # Inspect the machine-readable scientific conclusion.
    assert analysis["primary_complete"] is False and analysis["paired_comparison"] is None  # No invented inferential result.
    assert "incomplete" in (output / "paper.md").read_text().lower()  # The paper must prominently state evidence limits.
    assert (output / "experiments.md").exists() and result["paper"] == str(output / "paper.md")  # Usable artifact links.


def test_repeated_failures_keep_distinct_experiment_records(tmp_path):
    """Local: record two failures in one run. Global: later retries must not overwrite earlier failure reports."""
    root, output = tmp_path / "runs", tmp_path / "reports"  # Independent synthetic evidence tree.
    run = root / "main/failed-run"  # Both attempts belong to the same intended experiment.
    run.mkdir(parents=True)  # Create only the failure artifacts needed for this regression.
    for attempt in [1, 2]:  # Distinct errors must remain separately reviewable.
        (run / f"failure-{attempt}.json").write_text(json.dumps({"returncode": attempt}))  # Preserve original attempt identity.
    create_reports(root, output, pdf=False)  # Exercise discovery and final filenames, not a helper-only imitation.
    assert len(list((output / "experiments").glob("failure-*.md"))) == 2  # Prevent same-parent filename collisions.
