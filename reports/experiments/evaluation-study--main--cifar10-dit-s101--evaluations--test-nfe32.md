# Evaluation: study/main/cifar10-dit-s101/evaluations/test-nfe32

## Purpose and interpretation

This evaluation measures a fixed checkpoint and fixed noise population. KID subset SD describes metric subsampling variability; it is not a confidence interval across trained seeds. The first 64 images are shown without selection.

## Recorded evidence

```json
{
  "dataset": "cifar10",
  "family": "dit",
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
  "checkpoint_sha256": "030d935262170666e1c5f3da7020026a573aa3de03b9f86a9c67f036afbe0dce",
  "data_sha256": "51ca348f81e4c280d10942cf2fa0d4dd1ce688a16b0088283069b54bbaaf6f97",
  "source_revision": "c749856d1c9fe07d6ce2f39d01de78f664aea9a8",
  "primary_eligible": true,
  "device": "cuda",
  "blas_threads": 4,
  "status": "completed",
  "sampling": {
    "count": 5000,
    "seed": 20261016,
    "nfe": 32,
    "batch_size": 32,
    "seconds": 50.793108586993185,
    "seconds_per_image": 0.010158621717398637,
    "precision": "float32",
    "timing_excludes": "model load and warmup",
    "clipped_fraction": 0.015694791451096535,
    "peak_memory_bytes": 58955776
  },
  "metrics": {
    "frechet_inception_distance": 104.11149832999416,
    "kernel_inception_distance_mean": 0.09092151165008545,
    "kernel_inception_distance_std": 0.0017264464745771816
  },
  "validation_velocity_mse": 0.19273782055824995,
  "metric_seconds": 36.25003975699656,
  "id": "study/main/cifar10-dit-s101/evaluations/test-nfe32",
  "artifact": "runs/study/main/cifar10-dit-s101/evaluations/test-nfe32/evaluation.json"
}
```

## Fixed sample grid

![First 64 generated samples](../figures/evaluation-study--main--cifar10-dit-s101--evaluations--test-nfe32.png)

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
