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
