"""Local: report completeness checks. Global: absent seed results must remain incomplete rather than imply scientific success.

Source: https://neurips.cc/public/guides/PaperChecklist (claims must match available evidence).
"""

import json  # Writes small explicit result fixtures without running model training.

import pytest  # Compares numerical summaries with explicitly calculated expected values.

from ofm.report_summary import (  # Complete primary and descriptive secondary populations.
    summarize_primary_cost,  # All six declared training seeds at the primary solver cost.
    summarize_secondary_pairs,  # Both declared seeds for descriptive mechanism checks.
)
from ofm.report_text import number  # Parameter counts must remain exact in the readable evidence tables.
from ofm.reporting import analyze, create_reports  # Exercises production aggregation and rendering.


def test_report_preserves_integer_counts_and_boolean_outcomes():
    """Local: format exact counts without rounding. Global: the paper must retain actual capacity and clear verification outcomes."""
    assert number(1297712) == "1297712"  # Six-significant-digit float formatting would silently round this parameter count.
    assert number(True) == "True" and number(None) == "—"  # A passed check and an absent measurement have distinct meanings.


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


def test_all_runs_root_preserves_primary_pair_identity():
    """Local: aggregate a nested study alongside development runs. Global: one consolidated report must retain primary inference."""
    rows = [{"id": f"study/main/cifar10-{family}-s{seed}/evaluations/test-nfe32", "primary_eligible": True,
             "family": family, "training_seed": seed, "metrics": {"kernel_inception_distance_mean": value}}
            for seed in range(101, 107) for family, value in [("onn", 0.01), ("dit", 0.02)]]  # Six explicit complete pairs.
    result = analyze(rows)  # Exercise path-independent phase identification, not a manually rearranged record list.
    assert result["primary_complete"] and result["paired_seeds"] == list(range(101, 107))  # Root selection cannot hide measured pairs.


def test_cost_summary_uses_all_six_primary_seeds_only():
    """Local: exclude secondary costs and incomplete seed sets. Global: summary tables cannot silently change the estimand."""
    runs, evaluations = [], []  # Construct six known final-budget observations without any real computation.
    for seed in range(101, 107):  # The pre-specified independent experimental units.
        identifier = f"study/main/cifar10-onn-s{seed}"  # Match production artifact identities.
        runs.append({"id": identifier, "parameters": 100, "images_per_second": 200,
                     "peak_memory_bytes": 2 * 1024**2})  # Known capacity, throughput, and memory in bytes.
        evaluations.append({"id": identifier + "/evaluations/test-nfe32", "family": "onn", "training_seed": seed,
                            "primary_eligible": True, "metrics": {"kernel_inception_distance_mean": 0.2 if seed == 106 else 0.1,
                                                                  "frechet_inception_distance": 100},
                            "sampling": {"seconds_per_image": 0.01, "peak_memory_bytes": 1024**2}})  # Fixed measured costs.
    secondary = {**evaluations[0], "id": runs[0]["id"] + "/evaluations/test-nfe64", "primary_eligible": False,
                 "metrics": {"kernel_inception_distance_mean": 0, "frechet_inception_distance": 0}}  # Deliberately attractive alternate cost.
    summary = summarize_primary_cost(runs, evaluations + [secondary])  # Secondary sampling must not affect primary means.
    assert len(summary) == 1 and summary[0]["kid_mean"] == pytest.approx(0.7 / 6)  # Include the worse sixth seed, exclude the alternate NFE.
    assert summary[0]["milliseconds_per_image"] == 10 and summary[0]["training_peak_mib"] == 2  # Verify displayed unit conversion.
    assert summarize_primary_cost(runs, evaluations[:-1] + [secondary]) == []  # Five seeds cannot masquerade as the full population.


def test_secondary_summary_pairs_both_declared_seeds_and_excludes_other_costs():
    """Local: align two-seed mechanism checks. Global: preserve paired differences without expanding exploratory evidence into inference."""
    rows = []  # Known scores make pairing and missing-seed behavior directly checkable.
    for seed, baseline, ablated, shuffled in [(101, 0.1, 0.15, 0.3), (102, 0.2, 0.25, 0.2)]:
        for phase, family, intervention, score in [("main", "onn", "none", baseline),
                ("ablations", "no_frequency", "none", ablated), ("main", "onn", "shuffle", shuffled)]:
            rows.append({"id": f"study/{phase}/{family}-{seed}/evaluations/{intervention}", "family": family,
                         "training_seed": seed, "intervention": intervention, "primary_eligible": intervention == "none",
                         "nfe": 32, "split": "test", "generated_count": 5000, "real_count": 10000,
                         "metrics": {"kernel_inception_distance_mean": score}})  # Match the declared metric populations.
    alternate = {**rows[1], "nfe": 64, "metrics": {"kernel_inception_distance_mean": 0}}  # Attractive but wrong solver cost.
    summaries = summarize_secondary_pairs(rows + [alternate])  # Both seeds must be retained regardless of their direction.
    indexed = {row["condition"]: row for row in summaries}  # Compare known paired quantities, not table formatting.
    assert indexed["no_frequency"]["mean_difference"] == pytest.approx(0.05)  # Paired ablation-minus-original contrast.
    assert indexed["shuffle"]["differences"] == pytest.approx([0.2, 0])  # A null second seed must not disappear.
    assert not any(row["condition"] == "no_frequency" for row in summarize_secondary_pairs(rows[:4]))  # Missing second ablation stays incomplete.
