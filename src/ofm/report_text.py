"""Local: readable report composition. Global: generated prose states measured evidence, assumptions, and limitations explicitly.

Scientific sources are linked in the generated paper's related-work section; numeric claims come from saved artifacts.
"""

import json  # Presents complete scalar/configuration evidence without inventing narrative results.
from pathlib import Path  # Loads the exact configuration accompanying an experiment.

from ofm.report_summary import summarize_primary_cost  # Aggregates complete primary populations.


def number(value) -> str:
    """Local: format a displayed scalar. Global: distinguish missing results from measured zero."""
    return "—" if value is None else f"{value:.6g}" if isinstance(value, (float, int)) else str(value)  # Keep useful precision.


def table(headers: list[str], rows: list[list]) -> str:
    """Local: compose one Markdown table. Global: reports share consistent missing-value and numeric formatting."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]  # Standard GFM table header.
    lines.extend("| " + " | ".join(number(value).replace("|", "/") for value in row) + " |" for row in rows)  # Escape table delimiters.
    return "\n".join(lines) if rows else "No measurements are available yet."  # Do not imply an empty study measured zero.


def evidence_link(kind: str, record: dict) -> str:
    """Local: link a table row to its full record. Global: every displayed empirical value has inspectable primary evidence."""
    filename = kind + "-" + record["id"].replace("/", "--") + ".md"  # Matches the report writer's stable artifact-derived naming.
    return f"[{record['id']}](<experiments/{filename}>)"  # Relative links work in GitHub, local Markdown, and the Hub report bundle.


def training_table(runs) -> str:
    """Local: summarize every trained or untrained arm. Global: capacity and achieved compute remain visible beside quality."""
    return table(["Run", "Status", "Parameters", "Updates", "Train seconds", "Images/s", "Validation MSE"],
                 [[evidence_link("training", row), row["status"], row.get("parameters"), row.get("step"), row.get("train_seconds"),
                   row.get("images_per_second"), row.get("final_validation_mse")] for row in runs])  # All recorded arms, fixed path order.


def quality_table(evaluations) -> str:
    """Local: summarize every metric attempt that completed. Global: sample counts and metric variability are never implicit."""
    return table(["Evaluation", "Generated/real N", "NFE", "KID mean", "KID subset SD", "FID", "Pixel SWD", "ms/image"],
                 [[evidence_link("evaluation", row), f"{row['generated_count']}/{row['real_count']}", row["nfe"],
                   row["metrics"].get("kernel_inception_distance_mean"), row["metrics"].get("kernel_inception_distance_std"),
                   row["metrics"].get("frechet_inception_distance"), row["metrics"].get("pixel_sliced_wasserstein"),
                   1000 * row["sampling"]["seconds_per_image"]] for row in evaluations])  # No rounding before statistical analysis.


def experiment_record(kind: str, record: dict, image_reference: str | None) -> str:
    """Local: document one actual experiment. Global: retain motivation, settings, evidence, and appropriately limited interpretation."""
    notes = {  # Interpret by experimental role rather than by whether its result looks favorable.
        "training": "This run fits the pre-specified velocity MSE objective. Completion and held-out error establish execution; generation quality requires its separate evaluation. No checkpoint was selected by best validation or test score.",
        "evaluation": "This evaluation measures a fixed checkpoint and fixed noise population. KID subset SD describes metric subsampling variability; it is not a confidence interval across trained seeds. The first 64 images are shown without selection.",
        "diagnostic": "This fixed-batch probe checks optimization, finite geometry, and deterministic predictions. It intentionally memorizes 16 training examples with fixed noise and times. Passing does not establish generalization or generative quality.",
        "control": "This diagnostic checks metric scale/repetition or identity-intervention behavior. It is excluded from primary paired-seed inference and cannot establish an oscillator mechanism by itself.",
        "failure": "This attempt did not complete normally. Its existence is retained; missing metrics are not replaced by zero or silently removed from the planned study. Diagnose the recorded error before any authorized retry.",
    }
    visible = {key: value for key, value in record.items() if key not in {"losses", "left_indices", "right_indices"}}  # Large arrays stay intact in artifacts.
    config_path = Path(record["artifact"]).parent / "config.json"  # Training/diagnostic settings live next to their results.
    configuration = json.loads(config_path.read_text()) if config_path.exists() else None  # Evaluation records already include settings.
    text = f"# {kind.capitalize()}: {record['id']}\n\n## Purpose and interpretation\n\n{notes[kind]}\n\n"  # State role before numbers.
    text += "## Recorded evidence\n\n```json\n" + json.dumps(visible, indent=2, allow_nan=False) + "\n```\n\n"  # Complete scalar record.
    if configuration is not None:  # Do not invent settings for artifacts that do not contain a training configuration.
        text += "## Exact configuration\n\n```json\n" + json.dumps(configuration, indent=2) + "\n```\n\n"  # Preserve resolved defaults.
    if image_reference:  # Only existing images can support visual claims.
        text += f"## Fixed sample grid\n\n![First 64 generated samples]({image_reference})\n\n"  # No best-sample curation.
    text += ("## Method choices and learning\n\nThe shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison "
             "and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. "
             "Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. "
             "These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; "
             "causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, "
             "and logs remain in the referenced artifact bundle. Sources: "
             "[protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), "
             "[TorchCFM](https://github.com/atong01/conditional-flow-matching), "
             "[AKOrN](https://arxiv.org/html/2410.13821v2).\n")  # Explicit provenance and limits, without invented failure explanations.
    return text  # Returned Markdown is also the source of this experiment's PDF.


def experiment_index(runs, evaluations, diagnostics, failures, analysis) -> str:
    """Local: consolidate all evidence and terminology. Global: a reader can compare networks without assuming classification metrics."""
    text = f"# Experiment ledger and glossary\n\n{analysis['conclusion']}\n\n"  # Main scientific status appears first.
    text += ("All networks generate unconditional images in pixel space. They share independent conditional flow matching, "
             "AdamW, continuous time conditioning, 4×4 patches, and an EMA sampling checkpoint. MNIST is development data; "
             "CIFAR-10 is the final comparison domain. There is no curriculum or augmentation. Classification accuracy, text fill rate, "
             "and task-completion accuracy are not applicable; run completion is reported separately from image quality.\n\n")
    text += table(["Network", "Feature computation", "Training objective", "Role"], [  # Fixed architecture mapping for lay readers.
        ["onn", "Coupled 4D unit-vector rotations; four recurrent blocks", "Velocity MSE", "Proposed generator"],
        ["dit", "Four time-conditioned attention/MLP blocks", "Velocity MSE", "Parameter-matched transformer"],
        ["conv", "Convolution + SiLU + GroupNorm recurrence", "Velocity MSE", "Conventional recurrent control"],
        ["no_coupling", "Oscillators with coupling convolution removed", "Velocity MSE", "Trained removal; lower capacity"],
        ["no_frequency", "Oscillators with natural-frequency term removed", "Velocity MSE", "Trained removal; slightly lower capacity"],
    ]) + "\n\n## Training outcomes\n\n" + training_table(runs)  # Architecture explanations precede empirical tables.
    text += "\n\n## Distribution and cost measurements\n\n" + quality_table(evaluations)  # Every completed assessment is included.
    text += (f"\n\n## Completeness\n\nRecorded diagnostics: {len(diagnostics)}. Recorded failed attempts: {len(failures)}. "
             f"Completed primary pairs: {len(analysis['paired_seeds'])}/6. Missing values mean unmeasured, not zero. "
             "Individual records and PDFs are in `experiments/`; raw records remain in the run directories and published artifact manifests.\n\n")
    text += "## Glossary\n\n" + table(["Term", "Meaning"], [  # Definitions distinguish objectives, metrics, and uncertainty.
        ["Flow matching", "Learn a velocity that transports Gaussian noise toward the image distribution."],
        ["Oscillator", "A hidden vector whose direction changes under natural rotation and learned interactions."],
        ["Coupling", "A learned convolution that lets hidden oscillators influence one another."],
        ["Flow time / internal time", "Noise-to-image position / recurrent computation inside one velocity prediction."],
        ["NFE", "Number of velocity-network evaluations used to generate each image."],
        ["KID", "Unbiased squared kernel discrepancy between fixed Inception features; smaller is better and estimates may be negative."],
        ["KID subset SD", "Variation across random metric subsets of one fixed pair of image populations, not across trained models."],
        ["FID", "Gaussian feature-distribution distance; finite-sample biased, so sample count must accompany it."],
        ["Pixel SWD", "Wasserstein distance averaged over random pixel-space projections; a MNIST diagnostic, not semantic accuracy."],
        ["Seed pair", "Two independently initialized architectures using paired data/noise/time stream prefixes."],
        ["Confidence interval", "A model-based repeated-study procedure; here an approximate t interval across six seed differences."],
        ["Sign-flip test", "Enumerate all signs of paired differences; exact only under the specified exchangeability/symmetry null."],
        ["EMA", "Exponential moving average of trainable weights, used for all reported samples."],
        ["Ablation / intervention", "Retrain with a mechanism removed / perturb a trained model during inference."],
        ["Power", "Probability of detecting an assumed effect under a planning model; six pairs have about 24% power at effect 0.8."],
    ]) + "\n"  # One centralized glossary avoids duplicating these definitions throughout individual records.
    return text  # This index is the requested consolidated experiment summary.


def paper_text(runs, evaluations, diagnostics, controls, failures, analysis, budget, figures) -> str:
    """Local: assemble a technical paper from fixed methods and actual outputs. Global: strength of claims follows completed evidence."""
    paired = analysis["paired_comparison"]  # Missing primary outcomes must remain visibly missing throughout the paper.
    result_text = json.dumps(paired, indent=2) if paired else "Primary paired inference is unavailable because the study is incomplete."
    primary_sentence = (f"The mean oscillator-minus-transformer KID difference is {number(paired['mean_difference'])} "
                        f"(approximate 95% t interval [{number(paired['ci95'][0])}, {number(paired['ci95'][1])}]; "
                        f"exact two-sided sign-flip p = {number(paired['p_value'])})." if paired else "The primary comparison is incomplete.")
    summaries = summarize_primary_cost(runs, evaluations)  # Uses only complete six-seed primary populations at NFE32.
    overview = table(["Model", "Parameters", "Mean KID", "Mean FID", "Train images/s", "Sample ms/image", "Peak train/sample MiB"],
                     [[row["family"], row["parameters"], row["kid_mean"], row["fid_mean"], row["images_per_second"],
                       row["milliseconds_per_image"], f"{number(row['training_peak_mib'])}/{number(row['sampling_peak_mib'])}"]
                      for row in summaries])  # Memory is a maximum across seeds, not a confidence interval.
    text = rf"""# Oscillatory velocity networks for budget-constrained image flow matching

**Libor Burian**  
Independent researcher

## Abstract

We implement an attention-free, AKOrN-inspired velocity network for unconditional image generation and a reproducible comparison with a small DiT-style transformer and recurrent convolutional control. The design isolates backbone computation while sharing a straight conditional flow-matching path, data preparation, optimizer, and numerical sampler. The planned primary comparison uses six paired CIFAR-10 seeds, 45 minutes per model, 5,000 generated images per seed, and 32 velocity evaluations. {primary_sentence} **{analysis['conclusion']}** At report generation, {len(runs)} training/control records and {len(evaluations)} completed evaluations were available. The [recorded analysis](analysis.json) and linked experiment tables below provide the empirical evidence. This is an exploratory local-compute study, not a claim of state-of-the-art performance.

## Context, motivation, and introduction

Flow matching specifies a regression target for a continuous generative transport; it does not require self-attention. This experiment asks whether a coupled-oscillator parameterization gives useful sample quality for its measured cost on a small laptop GPU. The proposed advantage is falsifiable: an oscillator-minus-transformer KID difference reliably above zero contradicts improvement in this regime. A wide interval spanning zero is inconclusive. Implementation feasibility and comparative quality are separate claims.

## Background and related work

[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) motivates velocity regression along conditional probability paths. [TorchCFM](https://github.com/atong01/conditional-flow-matching) supplies the independent linear conditional path used here. [Artificial Kuramoto Oscillatory Neurons](https://arxiv.org/html/2410.13821v2), equations 4–7, motivates vector-valued spherical dynamics and linear mixing before a norm readout. This is an independent adaptation of those published equations to an image-velocity network, not copied AKOrN repository code or a reproduction of its original experiments. [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) motivates the time-conditioned transformer control. [Kuramoto Orientation Diffusion Models](https://arxiv.org/abs/2509.15328) is related work on a different generative process; this study changes the velocity backbone while retaining the ordinary image-space transport. We make no first-in-literature claim. KID follows the kernel-based estimator described in [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401), implemented with [torch-fidelity](https://torch-fidelity.readthedocs.io/en/latest/).

## Method

### Domain, data, and leakage control

MNIST (28×28, one channel) is the pilot domain; CIFAR-10 (32×32, RGB) is the comparison domain. Official torchvision training/test sources are downloaded with their supported archive integrity checks. A fixed stratified 5,000-example validation reservation uses split seed 20261016. SHA-256 hashes of decoded image bytes identify exact overlap; validation images overlapping test are excluded, and training examples overlapping either reserved validation or test are removed before manifests are frozen. This audit detects exact duplicates, not near-duplicates or feature-extractor pretraining contamination. Every model uses the same surviving indices, native pixel resolution, and uint8/127.5−1 normalization. There is no augmentation, label conditioning, pretrained generator, or curriculum. Test images never enter the optimizer; final metric results do not select hyperparameters or checkpoints.

### Objective and transport

For independent Gaussian noise $x_0$, an image $x_1$, and uniform $t$ on $[0,1]$, train

$$x_t=(1-t)x_0+t x_1,\qquad \mathcal{{L}}=\mathbb{{E}}\|v_\theta(x_t,t)-(x_1-x_0)\|_2^2.$$

The implementation uses a mean over pixels and minibatch elements. TorchCFM with sigma zero computes the path mean and conditional velocity; explicit generators supply noise and time so architecture initialization cannot consume these streams. Generation integrates $dx/dt=v_\theta(x,t)$ from Gaussian noise at zero to an image at one, using torchdiffeq's fixed midpoint solver. Sixteen outer steps require 32 actual velocity calls. Oscillator updates occur inside each call and never advance the outer flow time. The contract is `forward(images, times) -> velocities` with identical image shape and an unbounded velocity output.

### Oscillator backbone

A learned stride-four patch convolution maps pixels to width 128, with fixed two-dimensional sinusoidal positions. Continuous sinusoidal flow-time features feed a two-layer SiLU time encoder. Four blocks each evolve 32 four-dimensional hidden vectors per patch for four internal Euler updates of fixed size 0.1. A learned initial embedding is normalized and expanded deterministically on every forward call; no state is carried between ODE evaluations. Within a block, let $s_i$ be a unit vector, $c_i$ a time-conditioned stimulus, $J$ a learned 3×3 convolution, and $\Omega_i=(A_i-A_i^T)/2$ a learned skew-symmetric frequency matrix. Each internal update is

$$d_i=\Omega_i s_i + (J s+c)_i - s_i\langle s_i,(J s+c)_i\rangle,\qquad s_i\leftarrow \frac{{s_i+0.1d_i}}{{\|s_i+0.1d_i\|_2}}.$$

State normalization uses epsilon $10^{{-8}}$. The conditioning stimulus uses per-image GroupNorm and a learned time-dependent affine transformation. Natural frequencies are unconstrained parameters projected to skew symmetry at each use; no symmetry assumption is imposed on the learned convolution, so no energy-minimization guarantee is claimed. States carry between the four blocks within a call. A 1×1 convolution mixes all state channels into 128 four-dimensional vectors; their stabilized norms become 128 features. Mixing before norms is essential because the individual normalized state norms would otherwise be constant. A GroupNorm and two-layer SiLU convolutional readout add a residual scaled by 0.1. A shared LayerNorm/linear patch head reconstructs dense pixel velocities. Geometry and reductions remain float32 during mixed-precision training.

### Baselines, ablations, and sensitivity interventions

The transformer shares patch encoding, positional features, time encoding, and output prediction. Four blocks use four-head self-attention, a GELU MLP, and AdaLN-Zero time-conditioned residual gates. Its feed-forward width is deterministically adjusted in multiples of eight to match the oscillator trainable count within 5%; the default RGB width resolves to 576. The recurrent convolutional control retains the oscillator scaffold, convolution, iteration count, and mixed-state readout, replacing tangent dynamics with GroupNorm of a SiLU residual update. Counts, throughput, and memory are measured rather than assumed equal.

Separate no-coupling and no-natural-frequency models are trained for seeds 101 and 102. Retained parameters start from the corresponding ONN initialization. Removing coupling changes both parameter count and spatial information exchange, so its effects are not an isolated capacity-matched causal estimate; image-wide GroupNorm statistics remain. Zero frequency removes natural rotation, not all dynamics. Inference sensitivity diagnostics permute spatial hidden states after block one while preserving each vector's norm, and compare with an identity intervention on the same checkpoints and source noise. Residual image features are retained. They test sensitivity to correspondence, not whether synchronization caused quality. Untrained architectures and disjoint real-versus-real image populations provide additional controls. The spatial permutation uses seed 20261016 and is fixed before evaluation; it is not selected to maximize degradation. These operations are specified in the [backbone implementation](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/src/ofm/models/backbone.py) and [oscillator blocks](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/src/ofm/models/oscillator.py).

### Optimization, pairing, and budget

All trainable generators start from scratch with AdamW, learning rate 0.0003, betas (0.9,0.999), weight decay 0.01, effective batch 128, microbatch 32, gradient clipping at 1.0, and EMA decay 0.999. The initial rate is a conservative Adam starting point, not a validated optimum. Accumulation weights each microbatch by its fraction of the effective batch. BF16 autocast is used on CUDA while the MSE and oscillator geometry remain float32. Sampling and held-out velocity diagnostics use float32. CPU sampling is supported. No learning-rate search or adaptive quality-based stopping is used.

Seed-derived streams separate model initialization, data ordering, flow targets, validation, and generation. Data/noise/time stream prefixes are paired across architectures, while model initialization streams are independent. Unequal throughput implies different numbers of updates and examples at equal wall time; this is part of the quality-versus-cost estimand. Model execution order is randomized within each seed group before training. Checkpoints save optimizer, EMA, data permutation/cursor, CPU/CUDA RNG states, elapsed training time, dataset hash, resolved configuration, and source history. The final budget checkpoint is evaluated, not the best intermediate checkpoint.

The allocation is at most 1.5 GPU-hours for checks/pilot, 13.5 training hours for three CIFAR architectures × six seeds, three training hours for two ablations × two seeds, and six hours for sampling/evaluation/contingency. A shared file-locked ledger reserves time before GPU allocation, refunds unused reservations only on orderly exit, and charges crashed reservations conservatively. Cooperative loop checks leave cleanup time; an independent process deadline terminates a blocked worker near reservation expiry. The total ceiling is 24 occupied GPU-hours across resumes. Initialization, validation, checkpoints, downloads within reservations, and stalled wall time are conservatively charged; training-overhead accounting can therefore exceed the nominal optimizer-time allocation and consumes contingency. We do not measure energy, GPU utilization integrals, or hardware-independent FLOPs. The local target is an RTX 5070 Laptop GPU with 8 GB VRAM; actual software/hardware records accompany every run.

### Measurement and pre-specified analysis

The primary outcome is unscaled KID between 5,000 generated CIFAR images and all 10,000 official test images, with 100 subsets of size 1,000 and identical torch-fidelity Inception preprocessing and metric RNG. Generation and metric seed 20261016 is fixed across arms and training seeds. Generated floats are clamped to [-1,1], mapped to [0,255], rounded, and converted to uint8; complete unclipped tensors and clipping fractions remain available. Real images retain their original uint8 values. torch-fidelity owns resizing and feature normalization for both populations. Negative unbiased KID estimates are retained. FID is secondary and its generated/real sample counts accompany every value. MNIST uses 128-projection pixel-space sliced Wasserstein distance only as a pilot diagnostic; it is not mislabeled as semantic image quality. Velocity MSE uses the first fixed 512 surviving validation indices with a separate, training-seed-derived supervision stream, paired across architectures. Native BLAS metric threads are bounded by the recorded CPU-thread setting (default four).

For six primary seed pairs, report each KID difference, its mean, an approximate two-sided 95% Student-t interval with five degrees of freedom, and an exact two-sided paired sign-flip p-value at alpha 0.05. The t interval assumes approximately normal independent seed differences; sign-flip inference requires sign exchangeability/symmetry under the null. Six seeds cannot establish these assumptions. KID subset SD concerns one fixed image population and is not a standard error across trained models. Intervals condition on the fixed data split, test population, generation noise, metric RNG, and hardware. The shared dataset/evaluator limits external generalization. If any declared primary pair is missing, report available observations and incompleteness without complete-study inference. Ablations, fixed-seed cost curves, and interventions are descriptive secondary analyses without multiplicity-adjusted confirmatory claims.

The pre-training power calculation used 20,000 simulated Gaussian six-difference samples, all 64 sign assignments, and seed 20261016. At standardized effect 0.8, simulated rejection probability was 0.2406 (Monte Carlo standard error 0.0030). The first study is therefore exploratory and underpowered for moderate effects. Neither sample count nor training duration is extended to obtain significance. Measured throughput, synchronized generation latency, peak allocated GPU memory, parameter counts, and validation velocity MSE supplement distribution metrics. The seed-101 quality-cost curve uses 8,16,32,64 evaluations; no best seed is selected.

## Results and claims

{analysis['conclusion']}

{primary_sentence}

### Fixed-budget comparison

{overview}

Each available row summarizes all six declared seeds at 45 training minutes and 32 velocity evaluations. KID, FID, throughput, and latency are arithmetic means across seeds. Each quality evaluation uses 5,000 generated and 10,000 real images. Memory is the maximum allocated peak across those six runs, in MiB; it excludes the CUDA context and other processes. The [training](#training-and-capacity) and [evaluation](#quality-and-sampling-cost) tables link every underlying observation. Seed uncertainty belongs to the paired analysis below, not to KID subset SD.

Primary paired analysis:

```text
{result_text}
```

### Training and capacity

{training_table(runs)}

### Quality and sampling cost

{quality_table(evaluations)}

KID subset SD in the table is metric subset variability, not a training-seed confidence interval. FID is reported only with its finite generated/real population sizes. A missing entry denotes an unmeasured quantity. Status `untrained` is an explicit control and `step_limit` is a development run; neither qualifies as a final 45-minute checkpoint.

"""  # Fixed methods precede measured results and never imply that planned runs have completed.
    for figure in figures:  # Include only figures generated from actual recorded observations.
        text += f"![Observed evidence: {Path(figure).stem}]({figure})\n\n"  # Portable figure reference within the report bundle.
    text += "### Numerical checks and controls\n\n" + table(["Check", "Initial MSE", "Final MSE", "Ratio", "Passed"],
             [[evidence_link("diagnostic", row), row.get("initial_loss"), row.get("final_loss"), row.get("loss_ratio"), row.get("passed")] for row in diagnostics])
    text += "\n\n" + table(["Control", "Recorded result"], [[evidence_link("control", row), json.dumps({key: value for key, value in row.items()
                             if key in {"metrics", "repeat_max_absolute_difference", "identity_exact_equal"}})] for row in controls])
    text += f"""

### Failures and compute accounting

Recorded failed attempts: {len(failures)}. Cumulative charged GPU seconds: {number(budget.get('charged_seconds'))}. An active reservation, if present, is charged in advance and is not all completed work. Ledger history distinguishes successful completion, exceptions, and interrupted reservations. Individual reports retain exact errors and provenance; a later repair does not erase earlier attempts.

## Discussion and implications

The experimental unit is a full training seed, and the comparison applies to these implementations under equal occupied training time. Parameter matching does not equalize operations, receptive fields, memory access, optimization trajectories, or initial conditioning. A measured advantage would motivate replication across larger models and datasets; a disadvantage would constrain this specific design rather than all oscillator networks. A feasible generator can exist even when its comparative quality is worse. Inference corruption can show dependence on a representation while remaining compatible with several explanations of how that representation was learned.

## Limitations

This study uses small native-resolution datasets, unconditional pixel-space generation, one GPU, a small transformer, and six seeds. It does not address text conditioning, large pretrained generators, latent diffusion, or cloud-scale performance. The 24% planning power at moderate standardized effect makes inconclusive results likely. Two-seed trained ablations have weak uncertainty resolution and coupling removal changes capacity. Fixed hyperparameters may suit architectures unequally. Equal wall time depends on this hardware/software and background load; it is not a universal compute-normalized ranking. EMA with a fixed per-update decay also spans different wall-time windows when throughputs differ. Exact-duplicate exclusion does not remove semantic near-duplicates. Inception-based metrics have representation and finite-sample limitations, and MNIST pixel SWD is especially limited as a semantic measure. Deterministic settings support repeatability on this runtime, not guaranteed bitwise equivalence across hardware or library versions. Training interruptions can lose work since the last checkpoint; the budget still charges that occupied time conservatively.

## Conclusion

{analysis['conclusion']} The released interface, full artifacts, and pre-specified paired analysis support independent reproduction and inspection. Stronger claims require stronger completed evidence, not additional unplanned significance searches.

## Appendix A: reproducibility and artifact map

The repository README contains installation and exact CLI examples. `uv.lock` pins dependencies; the local environment uses Python 3.12 and CUDA 12.8 PyTorch wheels for Blackwell compatibility. Each run records environment versions, Git revisions, configuration, dataset manifests, logs, optimizer/EMA/RNG state, metrics, and full generated tensors. Source data and credentials are excluded from Git and model exports. Published safetensors exports load through the same model constructor with strict key checking; optimizer checkpoints use tensor/primitive-only loading. Timestamped command transcripts preserve emitted stdout/stderr, and full tensors remain binary artifacts. There is no runtime LLM service. The experiment ledger provides a lay glossary and all individual experiment records; the source protocol distinguishes pre-specified analyses from later diagnostics.

## Appendix B: methodology safeguards

This design follows the distinction between evidence and speculation emphasized by [Lipton and Steinhardt](https://arxiv.org/abs/1807.03341), data/validation pitfalls summarized by [Lones](https://arxiv.org/abs/2108.02497), and the [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist). Fixed-batch overfitting follows the debugging practice described in [Karpathy's training recipe](https://karpathy.github.io/2019/04/25/recipe/). These references motivate procedures; they are not empirical evidence that the proposed architecture is better. All comparisons, failures, resource limits, and distributional assumptions must be read alongside the measured results.
"""  # Complete technical context is retained even when a report is generated before study completion.
    return text  # This Markdown is the single source for the standalone technical PDF.
