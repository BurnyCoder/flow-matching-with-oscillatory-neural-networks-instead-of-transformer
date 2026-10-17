# Evaluation: study/ablations/cifar10-no_frequency-s101/evaluations/test-nfe32

## Purpose and interpretation

This evaluation measures a fixed checkpoint and fixed noise population. KID subset SD describes metric subsampling variability; it is not a confidence interval across trained seeds. The first 64 images are shown without selection.

## Recorded evidence

```json
{
  "dataset": "cifar10",
  "family": "no_frequency",
  "training_seed": 101,
  "split": "test",
  "intervention": "none",
  "generated_count": 5000,
  "real_count": 10000,
  "sampling_seed": 20261016,
  "metric_seed": 20261016,
  "nfe": 32,
  "kid_subsets": 100,
  "kid_subset_size": 1000,
  "checkpoint_sha256": "5de69e1afb911e81cba8030f52489023a73a601e54690a5babf6d0663d9d1b26",
  "data_sha256": "51ca348f81e4c280d10942cf2fa0d4dd1ce688a16b0088283069b54bbaaf6f97",
  "source_revision": "d63f7d219bedd7ab46057b5c4dae7029bd3944c6",
  "primary_eligible": true,
  "device": "cuda",
  "blas_threads": 4,
  "status": "completed",
  "sampling": {
    "count": 5000,
    "seed": 20261016,
    "nfe": 32,
    "batch_size": 32,
    "seconds": 82.52278135099914,
    "seconds_per_image": 0.016504556270199828,
    "precision": "float32",
    "timing_excludes": "model load and warmup",
    "clipped_fraction": 0.012793228961527348,
    "peak_memory_bytes": 60711424
  },
  "metrics": {
    "frechet_inception_distance": 163.11273570183076,
    "kernel_inception_distance_mean": 0.16979155778884888,
    "kernel_inception_distance_std": 0.0025134961417275985
  },
  "validation_velocity_mse": 0.21162250358611345,
  "metric_seconds": 18.965525784005877,
  "id": "study/ablations/cifar10-no_frequency-s101/evaluations/test-nfe32",
  "artifact": "runs/study/ablations/cifar10-no_frequency-s101/evaluations/test-nfe32/evaluation.json"
}
```

## Fixed sample grid

![First 64 generated samples](../figures/evaluation-study--ablations--cifar10-no_frequency-s101--evaluations--test-nfe32.png)

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
