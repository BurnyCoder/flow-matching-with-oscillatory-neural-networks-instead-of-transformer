"""Local: evidence figures and PDF rendering. Global: plots preserve all seeds and distinguish variation from uncertainty.

Sources: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html;
https://pandoc.org/MANUAL.html; https://github.com/messense/typst-py.
"""

from pathlib import Path  # Resolves figure and PDF assets within the report directory.

import matplotlib.pyplot as plt  # Uses the standard scientific plotting stack for standalone exportable figures.
import numpy as np  # Computes clearly labeled seed summaries without suppressing observations.
import pypandoc  # Supplies the bundled Pandoc Markdown-to-Typst converter.
import typst  # Compiles the generated Typst document into a standalone PDF.

plt.switch_backend("Agg")  # Selects a headless backend before any figures are created.


def make_figures(evaluations: list[dict], analysis: dict, output: Path) -> list[str]:
    """Local: draw observed seeds and cost curves. Global: never invent missing data or plot KID subsets as independent models."""
    directory, figures = output / "figures", []  # Keep artifacts portable inside the report bundle.
    directory.mkdir(parents=True, exist_ok=True)  # Empty studies legitimately produce no data figures.
    primary = [row for row in evaluations if row.get("primary_eligible") and "main" in Path(row["id"]).parts]
    if primary:  # Draw only measured trained-seed results.
        figure, axis = plt.subplots(figsize=(7, 4))  # One consistent axis for direct architecture comparison.
        for offset, family in enumerate(["onn", "dit", "conv"]):  # Fixed ordering avoids outcome-based visual emphasis.
            rows = sorted([row for row in primary if row["family"] == family], key=lambda row: row["training_seed"])
            values = [row["metrics"]["kernel_inception_distance_mean"] for row in rows]  # Preserve KID's native scale.
            axis.scatter([offset] * len(values), values, label=family, alpha=0.8)  # Show every seed, including outliers.
            if values:  # Draw the mean as a distinct marker, without pretending it is another replicate.
                axis.scatter(offset, np.mean(values), marker="_", s=400, color="black")  # Descriptive architecture mean.
        axis.set(xticks=range(3), xticklabels=["Oscillator", "Transformer", "Recurrent conv"], ylabel="KID (lower is better)",
                 title="Final 45-minute checkpoints; each dot is one seed")  # Units and experimental budget are explicit.
        axis.grid(axis="y", alpha=0.2)  # A light grid helps read small differences without overstating precision.
        figure.tight_layout()  # Prevent clipped axis labels in exported PDFs and PNGs.
        figure.savefig(directory / "seed-kid.png", dpi=180)  # Standalone report asset, not an interactive screenshot.
        plt.close(figure)  # Avoid accumulating figures during repeated report generation.
        figures.append("figures/seed-kid.png")  # Returned paths are relative for portable Markdown rendering.
    paired = analysis.get("paired_comparison")  # Only complete predeclared pair populations receive inferential intervals.
    if paired is not None:  # Partial studies remain descriptive instead of producing an attractive significance plot.
        figure, axis = plt.subplots(figsize=(7, 3))  # A difference plot makes the sign of the hypothesis direct.
        axis.scatter(analysis["paired_seeds"], paired["differences"], label="Observed paired seeds")  # Preserve all paired effects.
        axis.axhline(0, color="black", linewidth=1)  # Zero is the relevant null reference, not an arbitrary truncated baseline.
        axis.axhline(paired["mean_difference"], color="tab:blue", label="Paired mean")  # Separate central estimate from observations.
        axis.axhspan(*paired["ci95"], alpha=0.15, color="tab:blue", label="Approx. 95% t interval for mean")  # Seed uncertainty.
        axis.set(xlabel="Training seed", ylabel="KID(ONN) − KID(DiT)", title="Negative favors oscillators")  # Predeclared direction.
        axis.legend(fontsize=8)  # Explain the shaded interval rather than leaving ambiguous error bars.
        figure.tight_layout()  # Preserve labels in export.
        figure.savefig(directory / "paired-difference.png", dpi=180)  # Save a standalone inference figure.
        plt.close(figure)  # Release CPU plotting resources promptly.
        figures.append("figures/paired-difference.png")  # Include only figures backed by measured data.
    curves = [row for row in evaluations if row.get("training_seed") == 101 and "main" in Path(row["id"]).parts
              and row.get("intervention") == "none" and row.get("split") == "test"]  # Preselected cost-curve seed only.
    if curves:  # The first available checkpoint can provide a visibly partial curve.
        figure, axis = plt.subplots(figsize=(7, 4))  # Shared axes make cost/quality tradeoffs comparable.
        for family in ["onn", "dit", "conv"]:  # Fixed family order is independent of performance.
            rows = sorted([row for row in curves if row["family"] == family], key=lambda row: row["nfe"])
            axis.plot([row["sampling"]["seconds_per_image"] for row in rows],
                      [row["metrics"]["kernel_inception_distance_mean"] for row in rows], marker="o", label=family)
            for row in rows:  # Label actual solver cost as well as measured latency.
                axis.annotate(str(row["nfe"]), (row["sampling"]["seconds_per_image"], row["metrics"]["kernel_inception_distance_mean"]))
        axis.set(xlabel="Measured seconds per generated image", ylabel="KID (lower is better)",
                 title="Seed 101 only; labels are velocity evaluations")  # No seed error bars are possible for a one-seed curve.
        axis.legend()  # Architecture colors have an explicit mapping.
        figure.tight_layout()  # Ensure complete export.
        figure.savefig(directory / "quality-cost.png", dpi=180)  # Preserve the raw-axis, unselected cost comparison.
        plt.close(figure)  # Release the plot after saving.
        figures.append("figures/quality-cost.png")  # Paper embeds this fixed-seed diagnostic with its limitation.
    return figures  # Empty input yields an honest empty figure population.


def render_pdf(markdown: Path, root: Path) -> Path:
    """Local: convert Markdown through Typst. Global: every paper/report has a reproducible standalone PDF representation."""
    source = markdown.with_suffix(".typ")  # Preserve the intermediate typesetting source for debugging and replication.
    pypandoc.convert_file(str(markdown), "typst", outputfile=str(source),
                         extra_args=["--standalone", "--resource-path=" + str(root.resolve()), "-V", "papersize=a4"])
    destination = markdown.with_suffix(".pdf")  # Keep human-readable source and rendered output adjacent.
    typst.compile(str(source.resolve()), output=str(destination.resolve()), root=str(root.resolve()))  # No external LaTeX needed.
    return destination  # A concrete generated file, never a claimed PDF that was not compiled.
