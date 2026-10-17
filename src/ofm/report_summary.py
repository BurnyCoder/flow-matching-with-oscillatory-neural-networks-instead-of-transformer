"""Local: concise summaries of complete primary populations. Global: compress measured evidence without selecting seeds or costs.

Source: https://docs.python.org/3.12/library/statistics.html#statistics.fmean.
Inferential intervals remain in ofm.statistics; these summaries are descriptive arithmetic means.
"""

from pathlib import Path  # Joins each evaluation to the exact training artifact that produced its checkpoint.
from statistics import fmean  # Standard-library arithmetic means preserve equal weight per training seed.


def summarize_primary_cost(runs: list[dict], evaluations: list[dict]) -> list[dict]:
    """Local: join six final checkpoints per architecture. Global: show quality, throughput, latency, and memory at the fixed primary cost."""
    training = {row["id"]: row for row in runs}  # Keep artifact identities rather than joining only on a potentially repeated seed.
    summaries = []  # Incomplete architectures deliberately have no six-seed mean.
    for family in ("onn", "dit", "conv"):  # Fixed display order is independent of the measured ranking.
        rows = [row for row in evaluations if row.get("primary_eligible") and row["family"] == family
                and "main" in Path(row["id"]).parts]  # Exclude ablations, interventions, and secondary solver costs.
        seeds = [row["training_seed"] for row in rows]  # The experimental unit is the trained seed, not a KID subset.
        if len(seeds) != len(set(seeds)):  # Repeated evaluations cannot silently give one seed extra weight.
            raise ValueError(f"Duplicate primary cost observation for {family}")  # Require an explicit provenance repair.
        if set(seeds) != set(range(101, 107)):  # Preserve the pre-specified six-seed population.
            continue  # Individual measured rows remain available in the full evidence tables.
        models = [training[Path(row["id"]).parent.parent.as_posix()] for row in rows]  # Same exact run for quality and training cost.
        capacities = {row["parameters"] for row in models}  # A single displayed capacity must describe every summarized model.
        if len(capacities) != 1:  # Different architectures within a family would violate the frozen comparison.
            raise ValueError(f"Parameter counts differ within {family}")  # Do not hide configuration drift behind a mean.
        summaries.append({"family": family, "parameters": models[0]["parameters"],
                          "kid_mean": fmean(row["metrics"]["kernel_inception_distance_mean"] for row in rows),
                          "fid_mean": fmean(row["metrics"]["frechet_inception_distance"] for row in rows),
                          "images_per_second": fmean(row["images_per_second"] for row in models),
                          "milliseconds_per_image": 1000 * fmean(row["sampling"]["seconds_per_image"] for row in rows),
                          "training_peak_mib": max(row["peak_memory_bytes"] for row in models) / 1024**2,
                          "sampling_peak_mib": max(row["sampling"]["peak_memory_bytes"] for row in rows) / 1024**2})
    return summaries  # Memory is a maximum; means and units are explicitly named for rendering and independent inspection.


def summarize_secondary_pairs(evaluations: list[dict]) -> list[dict]:
    """Local: pair declared two-seed removals and interventions. Global: compress descriptive evidence without a new significance test."""
    scores = {}  # Each condition has exactly one recorded KID per declared seed.
    for row in evaluations:  # Select the pre-specified population, not the most favorable metric or sampling cost.
        if (row.get("training_seed") not in {101, 102} or row.get("nfe") != 32 or row.get("split") != "test"
                or row.get("generated_count") != 5000 or row.get("real_count") != 10000):
            continue  # Partial populations and alternate solver costs cannot replace a declared sensitivity assessment.
        parts, condition = Path(row["id"]).parts, None  # Artifact phase separates trained removals from inference perturbations.
        if "ablations" in parts and row.get("primary_eligible") and row["family"] in {"no_coupling", "no_frequency"}:
            condition = row["family"]  # These are separately trained final-budget models.
        elif "main" in parts and row["family"] == "onn":  # Inference diagnostics share the original oscillator run.
            if row.get("intervention") in {"identity", "shuffle"}:
                condition = row["intervention"]  # Keep identity as an explicit observed control.
            elif row.get("primary_eligible"):
                condition = "original"  # Reference is the original 32-NFE final-budget assessment.
        if condition is None:  # Transformer, recurrent-conv, untrained, and development records are unrelated here.
            continue  # They remain visible in the full measurement tables.
        key = condition, row["training_seed"]  # Explicit pairing prevents filename order from changing the contrast.
        if key in scores:  # Never pick one of multiple apparent replicates by score or discovery order.
            raise ValueError(f"Duplicate secondary observation for {key}")  # Require an explicit evidence repair.
        scores[key] = row["metrics"]["kernel_inception_distance_mean"]  # Preserve unscaled KID at full stored precision.
    summaries = []  # Missing pairs remain absent from this complete-pair summary.
    for condition in ("no_coupling", "no_frequency", "identity", "shuffle"):  # Predeclared order, independent of effect direction.
        if not all((arm, seed) in scores for arm in ("original", condition) for seed in (101, 102)):
            continue  # One seed cannot stand in for the planned two-seed diagnostic.
        differences = [scores[condition, seed] - scores["original", seed] for seed in (101, 102)]  # Ablated/intervened minus original.
        summaries.append({"condition": condition, "kid_mean": fmean(scores[condition, seed] for seed in (101, 102)),
                          "differences": differences, "mean_difference": fmean(differences)})  # Descriptive statistics only.
    return summaries  # No confidence interval or p-value is inferred from these two exploratory seeds.
