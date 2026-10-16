"""Local: synthesize recorded experimental evidence. Global: reports follow immutable artifacts rather than desired conclusions.

Sources: https://neurips.cc/public/guides/PaperChecklist;
https://arxiv.org/abs/1807.03341 (separate empirical findings from speculative explanations).
"""

import json  # Reads complete machine-readable experiment records.
import shutil  # Copies fixed sample grids into a portable report bundle.
from pathlib import Path  # Discovers only artifacts under the explicitly requested study root.

from ofm.artifacts import git_revision, timestamp, write_json  # Shares provenance and atomic analysis output.
from ofm.report_figures import make_figures, render_pdf  # Keeps visualization/typesetting details below orchestration.
from ofm.report_text import (  # Reuses readable report composition functions.
    experiment_index,  # Consolidates every model and technical term.
    experiment_record,  # Documents each individual attempt.
    paper_text,  # Synthesizes methods and measured claims.
)
from ofm.statistics import (  # Shared inferential and pre-training planning calculations.
    paired_comparison,  # Compares complete trained seed pairs.
    planning_power,  # Reproduces the frozen Gaussian planning simulation.
)


def collect_records(root: Path, filename: str) -> list[dict]:
    """Local: read every matching artifact. Global: missing/failed experiments remain distinguishable from favorable completed ones."""
    return [{**json.loads(path.read_text()), "id": path.parent.relative_to(root).as_posix(), "artifact": str(path)}
            for path in sorted(root.rglob(filename))]  # Stable path ordering avoids ranking records by observed performance.


def analyze(evaluations: list[dict]) -> dict:
    """Local: align declared seed pairs. Global: incomplete results cannot produce a complete-study significance claim."""
    candidates = [row for row in evaluations if row.get("primary_eligible") and row["id"].startswith("main/")
                  and row["family"] in {"onn", "dit"}]  # Reject controls, secondary NFE values, and partial training checkpoints.
    indexed = {}  # One score per architecture/seed is required to avoid implicit best-evaluation selection.
    for row in candidates:  # Validate every observed primary score before calculating inference.
        key = row["family"], row["training_seed"]  # Pair identity is explicit rather than positional file ordering.
        if key in indexed:  # Multiple primary evaluations require an explicit provenance decision, never silent selection.
            raise ValueError(f"Duplicate primary evaluation for {key}")  # Refuse to pick the lowest metric.
        indexed[key] = row["metrics"]["kernel_inception_distance_mean"]  # Native unscaled KID.
    seeds = [seed for seed in range(101, 107) if ("onn", seed) in indexed and ("dit", seed) in indexed]  # Declared complete pairs.
    complete = len(seeds) == 6  # Six independent pairs were specified before training.
    paired = paired_comparison([indexed["onn", seed] for seed in seeds], [indexed["dit", seed] for seed in seeds]) if complete else None
    conclusion = "Primary comparison incomplete; no claim about an oscillator advantage is supported."  # Default for missing evidence.
    if paired is not None:  # Interpret direction and uncertainty without stopping or extending experiments based on p-values.
        low, high = paired["ci95"]  # Approximate seed interval is distinct from per-evaluation KID subset standard deviation.
        if high < 0 and paired["p_value"] <= 0.05:  # Both predeclared summaries support the proposed direction in this regime.
            conclusion = "Exploratory evidence favors oscillator KID at this fixed training budget; broader superiority is untested."
        elif low > 0 and paired["p_value"] <= 0.05:  # A reliably positive difference contradicts the directional hypothesis.
            conclusion = "The observed positive KID difference contradicts the proposed oscillator improvement in this regime."
        else:  # Uncertain or discordant summaries must not be called equivalence or success.
            conclusion = "The primary result is inconclusive under the pre-specified interval and sign-flip analysis."
    return {"primary_complete": complete, "paired_seeds": seeds, "paired_comparison": paired, "conclusion": conclusion,
            "expected_seeds": list(range(101, 107)), "generated_at": timestamp(), "report_revision": git_revision()}  # Auditable synthesis.


def create_reports(root: Path, output: Path, pdf: bool = True) -> dict:
    """Local: collect, analyze, render. Global: one command rebuilds every experiment record and the technical paper from artifacts."""
    root, output = Path(root), Path(output)  # Normalize API callers as well as CLI paths.
    output.mkdir(parents=True, exist_ok=True)  # A fresh checkout can produce a report without manual directories.
    runs, evaluations = collect_records(root, "training.json"), collect_records(root, "evaluation.json")  # All measured arms.
    diagnostics, controls = collect_records(root, "diagnostic.json"), collect_records(root, "control.json")  # Explicit sanity checks.
    failures = collect_records(root, "failure*.json")  # Includes numerical, command, metric, and process-deadline failures.
    analysis = analyze(evaluations)  # Inferential claims depend only on the complete fixed primary population.
    analysis.update({"run_count": len(runs), "evaluation_count": len(evaluations), "failure_count": len(failures)})  # Evidence completeness.
    budget_path = root.parent / "budget.json"  # Default layout shares one ledger across study and development checks.
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {}  # Missing accounting is never invented.
    write_json(output / "analysis.json", analysis)  # Machine-readable results support independent audit and reanalysis.
    figures = make_figures(evaluations, analysis, output)  # Figures exist only when corresponding observations exist.
    records = [("training", row) for row in runs] + [("evaluation", row) for row in evaluations]  # Every reported experiment gets a record.
    records += [("diagnostic", row) for row in diagnostics] + [("control", row) for row in controls] + [("failure", row) for row in failures]
    documents = []  # Tracks actual Markdown files requiring optional PDF counterparts.
    for kind, record in records:  # Do not filter failures, untrained controls, or weak-performing seeds.
        name = (kind + "-" + record["id"].replace("/", "--"))  # Stable descriptive filename, independent of metric magnitude.
        destination = output / "experiments" / f"{name}.md"  # Separates individual evidence from the consolidated paper.
        destination.parent.mkdir(parents=True, exist_ok=True)  # Create the per-experiment report directory.
        image_path = Path(record["artifact"]).parent / "grid.png"  # Generated grids always contain the first fixed 64 outputs.
        image_reference = None  # Not every diagnostic produces an image grid.
        if image_path.exists():  # Copy only existing evidence, never a placeholder image.
            image_target = output / "figures" / f"{name}.png"  # Keeps the Markdown/PDF bundle portable.
            shutil.copy2(image_path, image_target)  # Preserve the original unselected image grid exactly.
            image_reference = "../figures/" + image_target.name  # Relative to the per-experiment Markdown directory.
        destination.write_text(experiment_record(kind, record, image_reference), encoding="utf-8")  # Complete settings and outputs.
        documents.append(destination)  # Compile every experiment, not only the primary paper.
    index = output / "experiments.md"  # One lay-readable summary of all measured networks and technical terms.
    index.write_text(experiment_index(runs, evaluations, diagnostics, failures, analysis), encoding="utf-8")  # No favorable-seed selection.
    paper = output / "paper.md"  # Standalone technical synthesis with methods, assumptions, results, and limitations.
    paper.write_text(paper_text(runs, evaluations, diagnostics, controls, failures, analysis, budget, figures), encoding="utf-8")
    documents.extend([index, paper])  # Consolidated sources receive the same PDF rendering as individual experiments.
    power = planning_power()  # Reproduces the frozen planning simulation without using any observed training result.
    write_json(output / "planning-power.json", power)  # Preserves every planned effect and Monte Carlo standard error.
    power_document = output / "planning-power.md"  # Separate experiment record for the pre-training planning analysis.
    power_document.write_text("# Planning power analysis\n\nThis reproduces the pre-training simulation in the protocol. "
        "It is not a post-hoc estimate from observed KID differences. Six pairs were retained despite low moderate-effect power "
        "because the 24-hour budget defines an exploratory study. Common Gaussian draws are shifted across effect scenarios; "
        "all 64 sign assignments are enumerated for every simulated study. The null rejection rate is below 0.05 because "
        "the exact test is discrete.\n\n```json\n" + json.dumps(power, indent=2) + "\n```\n\n"
        "Source: [SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).\n",
        encoding="utf-8")  # Complete assumptions and outcomes without an implied empirical advantage.
    documents.append(power_document)  # The planning experiment also receives a reproducible PDF.
    if pdf:  # Users may explicitly request text-only generation for quick inspection.
        for document in documents:  # Sequential CPU rendering avoids competing with GPU-run timing unnecessarily.
            render_pdf(document, output)  # Pandoc and Typst errors propagate rather than claiming a missing PDF exists.
    return {"paper": str(paper), "pdf": str(paper.with_suffix(".pdf")) if pdf else None,
            "experiments": len(records), "analysis": analysis}  # Concrete artifacts and appropriately limited conclusions.
