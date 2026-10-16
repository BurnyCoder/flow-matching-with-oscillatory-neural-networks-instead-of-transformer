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
