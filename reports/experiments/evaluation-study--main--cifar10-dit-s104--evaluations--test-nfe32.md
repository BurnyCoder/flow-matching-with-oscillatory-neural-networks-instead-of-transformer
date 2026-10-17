# Evaluation: study/main/cifar10-dit-s104/evaluations/test-nfe32

## Purpose and interpretation

This evaluation measures a fixed checkpoint and fixed noise population. KID subset SD describes metric subsampling variability; it is not a confidence interval across trained seeds. The first 64 images are shown without selection.

## Recorded evidence

```json
{
  "dataset": "cifar10",
  "family": "dit",
  "training_seed": 104,
  "split": "test",
  "intervention": "none",
  "generated_count": 5000,
  "real_count": 10000,
  "sampling_seed": 20261016,
  "metric_seed": 20261016,
  "nfe": 32,
  "kid_subsets": 100,
  "kid_subset_size": 1000,
  "checkpoint_sha256": "931c047b6ac7ef2a657e898a26f94ab432497b05a07e2c281eef36f50e20588a",
  "data_sha256": "51ca348f81e4c280d10942cf2fa0d4dd1ce688a16b0088283069b54bbaaf6f97",
  "source_revision": "5cb68c27c6dddcba6b49df92776493e18f828f06",
  "primary_eligible": true,
  "device": "cuda",
  "blas_threads": 4,
  "status": "completed",
  "sampling": {
    "count": 5000,
    "seed": 20261016,
    "nfe": 32,
    "batch_size": 32,
    "seconds": 67.29390516399872,
    "seconds_per_image": 0.013458781032799743,
    "precision": "float32",
    "timing_excludes": "model load and warmup",
    "clipped_fraction": 0.01615540310740471,
    "peak_memory_bytes": 58955776
  },
  "metrics": {
    "frechet_inception_distance": 105.1405045558987,
    "kernel_inception_distance_mean": 0.0926866340637207,
    "kernel_inception_distance_std": 0.0019192328223508978
  },
  "validation_velocity_mse": 0.1988416574895382,
  "metric_seconds": 19.559638132996042,
  "id": "study/main/cifar10-dit-s104/evaluations/test-nfe32",
  "artifact": "runs/study/main/cifar10-dit-s104/evaluations/test-nfe32/evaluation.json"
}
```

## Fixed sample grid

![First 64 generated samples](../figures/evaluation-study--main--cifar10-dit-s104--evaluations--test-nfe32.png)

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
