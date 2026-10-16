# Pre-specified exploratory protocol — 2026-10-16

## Question and hypothesis

Can an attention-free, AKOrN-inspired velocity network generate competitive images for its measured cost? The primary directional hypothesis is lower CIFAR-10 Kernel Inception Distance (KID) than a small transformer after equal 45-minute training budgets. The primary estimand is mean paired KID(ONN) minus KID(DiT), at 32 velocity evaluations and 5,000 generated images per training seed. The comparison is conditional on this small-model regime and dataset, not all transformers or all flow-matching systems.

A positive interval excluding zero contradicts the improvement hypothesis; an interval spanning zero is inconclusive, not evidence of equivalence. Report the two-sided exact paired sign-flip test at alpha 0.05 and approximate Student-t 95% intervals. Both require assumptions about the seed-difference distribution (symmetry for sign flipping, approximate normality for t inference); six seeds cannot establish these assumptions. Do not select seeds or extend runs based on p-values.

## Model and objective

Independent conditional flow matching uses Gaussian x0, image x1, uniform t, xt=(1-t)x0+t*x1, target x1-x0, and mean squared velocity error. TorchCFM supplies the conditional path. The backbone has four blocks, width 128, 4x4 patches, continuous sinusoidal time conditioning, and four-dimensional vector oscillators. Within each block perform four projected Euler updates of size 0.1, normalizing each oscillator to unit length. Learn 3x3 convolutional coupling, skew-symmetric natural frequencies, conditional stimuli, and a linear-mixing-then-norm readout. Initialize states from a learned deterministic embedding on every forward call. The outer flow ODE uses the midpoint solver, with 16 steps / 32 function evaluations. Internal oscillator time is separate from flow time.

The transformer uses adaptive time conditioning, attention, and feed-forward layers; tune only its feed-forward width to match ONN parameters within 5%. The recurrent convolutional control retains the scaffold/readout while replacing projected oscillator dynamics with normalized conventional nonlinear recurrence. The natural-frequency ablation sets the entire rotation term to zero; this does not remove all state dynamics. The coupling ablation removes the coupling term, not all spatial effects of the scaffold. Report parameter counts for every arm; ablations intentionally remove active parameters.

## Data and optimization

Use torchvision MNIST (28x28 grayscale) and CIFAR-10 (32x32 RGB), with x/127.5-1 normalization and no augmentation for the first study. Hold out a deterministic stratified 5,000-item validation subset from each official training split, using split seed 20261016. Preserve official test splits. Hash decoded image bytes; exclude training/validation overlaps with later held-out splits, and record all removed counts and manifests. Test data are not used for tuning or checkpoint selection. Inception is an external fixed evaluator, not a trainable part of the generator.

AdamW learning rate 3e-4, betas (0.9, 0.999), weight decay 0.01, effective batch size 128, microbatch 32 with accumulation, gradient norm cap 1.0, EMA 0.999. Keep oscillator geometry in float32 inside mixed-precision training. No adaptive stopping on validation quality. Save final budget checkpoints and intermediate checkpoints; only final budget checkpoints enter the primary comparison. Initialization is model-specific; ordering/noise/time streams share seed prefixes across arms. Evaluation streams are disjoint from training.

## Allocation and stopping

| Phase | Maximum GPU-hours |
|---|---:|
| Numerical checks, profiling, MNIST pilot | 1.5 |
| CIFAR-10: ONN, DiT, recurrent control; seeds 101–106; 45 minutes each | 13.5 |
| No coupling and no natural frequency; seeds 101–102; 45 minutes each | 3 |
| Sampling, evaluation, contingency | 6 |

The ledger measures elapsed occupied GPU time, including stalls, rather than GPU utilization or FLOPs. Account for training, sampling, and evaluation, and preserve usage across restarts. A crashed reservation is charged conservatively. Never exceed 24 total hours. Run orders are fixed before comparison using seed 20261016. Pilot changes must be documented and frozen before CIFAR-10. Budget exhaustion or a failed arm makes the relevant comparison incomplete; do not silently replace it with a favorable subset.

## Evaluation and controls

KID: torch-fidelity Inception features, 100 subsets of 1,000 examples, 5,000 generated images against all official test images. Use the same rendering, clipping, resize implementation, and metric seed across arms within each training seed. KID subset standard deviation is not uncertainty across trained models. Secondary FID is labeled with its finite sample count. Record held-out velocity MSE, parameters, peak memory, updates/images per second, synchronized sampling latency, and actual NFE. A fixed sampling grid uses 8, 16, 32, and 64 NFE for the first paired seed; all-seed primary results use 32.

Provide real-versus-real and untrained-generator controls. Use deterministic norm-preserving spatial state permutations and identity interventions as inference sensitivity checks, not independent evidence that synchronization explains quality. Trained ablations have only two paired seeds and are descriptive. Show all preselected sample grids, losses, failures, and raw measurements. No best-seed illustrations.

## Planning power analysis

A pre-training Monte Carlo calculation used 20,000 Gaussian difference vectors, six paired seeds, RNG seed 20261016, all 64 sign flips, and a two-sided alpha-0.05 test. Rejection probabilities for standardized effects 0, 0.5, 0.8, 1, 1.5, 2 were respectively 0.0299, 0.1133, 0.2406, 0.3547, 0.6567, 0.8685. Monte Carlo standard error at effect 0.8 was 0.0030. The discrete test is conservative at six pairs. This budget supports feasibility and detection of large effects, not a well-powered test of moderate advantages. This is a planning model, not observed training-seed variability.

## Primary sources

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [TorchCFM reference implementation](https://github.com/atong01/conditional-flow-matching)
- [Artificial Kuramoto Oscillatory Neurons, equations 4–7](https://arxiv.org/html/2410.13821v2)
- [DiT](https://arxiv.org/abs/2212.09748)
- [Kuramoto Orientation Diffusion Models](https://arxiv.org/abs/2509.15328): related work changes the stochastic process, unlike this backbone substitution.
- [torch-fidelity](https://torch-fidelity.readthedocs.io/en/latest/)
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
- [ML pitfalls](https://arxiv.org/abs/2108.02497), [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist), [Troubling Trends](https://arxiv.org/abs/1807.03341)
