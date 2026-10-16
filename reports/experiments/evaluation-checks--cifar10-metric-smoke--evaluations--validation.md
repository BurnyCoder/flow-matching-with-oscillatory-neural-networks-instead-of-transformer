# Evaluation: checks/cifar10-metric-smoke/evaluations/validation

## Purpose and interpretation

This evaluation measures a fixed checkpoint and fixed noise population. KID subset SD describes metric subsampling variability; it is not a confidence interval across trained seeds. The first 64 images are shown without selection.

## Recorded evidence

```json
{
  "dataset": "cifar10",
  "family": "onn",
  "training_seed": 101,
  "split": "validation",
  "intervention": "none",
  "generated_count": 128,
  "real_count": 5000,
  "sampling_seed": 20261016,
  "metric_seed": 20261016,
  "nfe": 32,
  "kid_subsets": 100,
  "kid_subset_size": 128,
  "checkpoint_sha256": "8e8cafa8870ab0ff9c618a39ba1f368be3966ff67459009f4ca195f437cd1aec",
  "data_sha256": "51ca348f81e4c280d10942cf2fa0d4dd1ce688a16b0088283069b54bbaaf6f97",
  "source_revision": "7926ff79c12be5d85eadb49b81aad5d3acfaedcd",
  "primary_eligible": false,
  "device": "cuda",
  "blas_threads": 4,
  "status": "completed",
  "sampling": {
    "count": 128,
    "seed": 20261016,
    "nfe": 32,
    "batch_size": 32,
    "seconds": 2.7344285610015504,
    "seconds_per_image": 0.021362723132824613,
    "precision": "float32",
    "timing_excludes": "model load and warmup",
    "clipped_fraction": 0.3879445493221283,
    "peak_memory_bytes": 60721664
  },
  "metrics": {
    "frechet_inception_distance": 487.66587247902373,
    "kernel_inception_distance_mean": 0.6097407078742981,
    "kernel_inception_distance_std": 0.005960302714145461
  },
  "validation_velocity_mse": 1.622360736131668,
  "metric_seconds": 15.261793892001151,
  "id": "checks/cifar10-metric-smoke/evaluations/validation",
  "artifact": "runs/checks/cifar10-metric-smoke/evaluations/validation/evaluation.json"
}
```

## Fixed sample grid

![First 64 generated samples](../figures/evaluation-checks--cifar10-metric-smoke--evaluations--validation.png)

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](../../docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
