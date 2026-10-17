# Scientific validity audit — 2026-10-18

This is a post-hoc audit of the completed study, specified after observing its primary results and before computing the additional replay and filtered-reference outcomes. It does not replace the original [protocol](protocol.md), change its hypothesis, or authorize further training. Libor Burian, independent researcher, is the report author.

## Questions and fixed decisions

1. Do the frozen schedule, data, configurations, checkpoints, generated populations, and recorded metrics agree?
2. Does independent feature extraction reproduce all 43 recorded evaluations and the real-versus-real control?
3. Does the oscillator-minus-transformer KID comparison change when externally cataloged questionable reference images are excluded?

The original primary estimand, six training seeds, final checkpoints, and 5,000 generated images per seed remain unchanged. The sensitivity analysis changes only the reference population. It is exploratory and conditional on the original split, generated noise, metric RNG, hardware, and training procedure. There is no significance-based extension, seed removal, checkpoint selection, or hyperparameter search.

## External annotation provenance and exclusion rule

Use the ciFAIR repository at commit `4c2764277f5fda8fec6784a78c1818eab13236c5`:

| File under `meta/` | SHA-256 |
| --- | --- |
| `duplicates_cifar10.csv` | `4cb7e99a7dfff346082c9d8fa4c2989a196e4a37d1e58f75936046696f1ba6a4` |
| `duplicates_cifar10_test.csv` | `ef7b33e2f32d056cdb342046b437ee819541a07cb3bb0520a3ada9b9c035f532` |

The authors' annotations distinguish near-duplicates (judgment 1) from very similar images (judgment 2). The conservative sensitivity population excludes the union of **all `TestID` values in both files**, including judgment 2, without using generator outcomes. These are zero-based original CIFAR indices. `TrainID` in the cross-split file indexes the original 50,000 training images; it must not be interpreted as a test index. Keep surviving test images in original index order. The fixed union contains 308 images, leaving 9,692 references. It is a catalog-filtered population, not a certified duplicate-free dataset. Do not silently substitute the ciFAIR replacement images.

Cross-reference the annotations against the actual 45,000 training and 5,000 validation indices. Report known near-duplicate and very-similar counts separately, including the 217 training-linked near-duplicate test images found during planning. Preserve both downloaded CSVs and exact included/excluded indices with the audit evidence.

Sources: [annotation definitions](https://github.com/cvjena/cifair/blob/4c2764277f5fda8fec6784a78c1818eab13236c5/meta/README.md), [cross-split catalog](https://github.com/cvjena/cifair/blob/4c2764277f5fda8fec6784a78c1818eab13236c5/meta/duplicates_cifar10.csv), [within-test catalog](https://github.com/cvjena/cifair/blob/4c2764277f5fda8fec6784a78c1818eab13236c5/meta/duplicates_cifar10_test.csv), [Barz and Denzler](https://arxiv.org/abs/1902.00423).

## Replay and analysis

- Hash the original scientific artifacts before and after the audit. The shared compute ledger and newly captured command logs are expected to grow; existing experiment files must remain byte-identical.
- Check all scheduled runs and all 43 evaluation identities, not just a favorable complete subset. Verify configuration, checkpoint/data hashes, final training status, EMA count, sample shape/finiteness, recorded clipping, and actual sampling NFE. Preserve the original primary analysis and independently reconstruct its arithmetic.
- Re-extract Inception 2048-dimensional features from saved raw samples and prepared real pixels on the original CUDA runtime with the original preprocessing, batching, metric seed, and 100 KID subsets. Never read old feature/statistic caches. New caches belong to one timestamped audit attempt; record extractor-weight and feature hashes.
- Replay all CIFAR KID means, KID subset SDs, and FID values, the four MNIST pixel-SWD evaluations, held-out velocity MSE, and the real-versus-real control. CUDA-derived features need not equal CPU-derived features bitwise. Record runtime precision settings; do not silently change TF32/AMP policy.
- Require absolute replay differences no larger than `1e-6` for KID mean/SD, `1e-3` for FID, and `1e-8` for pixel SWD and velocity MSE. Projection counts must match exactly. Report discrepancies without relaxing thresholds after observing them.
- Independently evaluate the cubic polynomial MMD U-statistic in float64 on the first 64 features from each CIFAR population (or the entire population if smaller). Compare against torch-fidelity's float64 estimator within `1e-12`; also disclose its float32 difference. This validates the estimator algebra separately from replaying the original wrapper.
- For all 18 main checkpoints, compute filtered-reference KID with the unchanged generated population, seed 20261016, 100 subsets of 1,000, and 32 NFE. Report all original/filtered seed scores and changes. Compute the six ONN-minus-DiT differences, their mean, approximate t interval, and exact sign-flip p-value with the original statistical routine. Label these summaries post-hoc, not a new confirmatory test.
- If the filtered comparison reverses direction or no longer supports the original directional interpretation, qualify the conclusion. Agreement supports robustness only to this specified catalog filter. It cannot establish absence of all contamination or explain a causal oscillator mechanism.

Sources: [torch-fidelity 0.4.0 metric implementation](https://github.com/toshas/torch-fidelity/tree/v0.4.0/torch_fidelity), [KID paper](https://arxiv.org/abs/1801.01401), [SciPy paired permutation inference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), [PyTorch 2.11 numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).

## Resource, failure, and reporting policy

The entire audit has a cumulative **3,600-second GPU reservation allowance**, including retries, within the existing 24-hour `runs/budget.json` ledger. All audit reservations use one stable label prefix so changing the report directory cannot renew this allowance. Crashes retain their full reservation charge. Do not create a separate compute ledger. Stop before the deadline and preserve incomplete results. CPU-only tests and rendering do not spend GPU allocation.

Store immutable timestamped attempts under `runs/scientific-audit/`, fresh features under the workspace cache, and derived Markdown/PDF/JSON under `reports/scientific-audit/`. Persist each completed replay before starting another. Report missing evidence, exceptions, failed tolerance checks, and source-preservation failures explicitly. A nonzero audit exit means verification failed or remains incomplete; the scientific effect need not be significant for a valid audit to complete.

Tests precede implementation. Commit and push this protocol and audit code before GPU replay. Review correctness, security, maintainability, reproducibility, and inferential validity once before release. Keep original six-seed results visible alongside the audit. Timing remains descriptive of the workstation; unequal update counts, fixed per-update EMA horizons, six-seed uncertainty, and two-seed ablation limitations remain in force.
