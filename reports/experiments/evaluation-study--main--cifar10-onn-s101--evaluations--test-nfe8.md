# Evaluation: study/main/cifar10-onn-s101/evaluations/test-nfe8

## Purpose and interpretation

This evaluation measures a fixed checkpoint and fixed noise population. KID subset SD describes metric subsampling variability; it is not a confidence interval across trained seeds. The first 64 images are shown without selection.

## Recorded evidence

```json
{
  "dataset": "cifar10",
  "family": "onn",
  "training_seed": 101,
  "split": "test",
  "intervention": "none",
  "generated_count": 5000,
  "real_count": 10000,
  "sampling_seed": 20261016,
  "metric_seed": 20261016,
  "nfe": 8,
  "kid_subsets": 100,
  "kid_subset_size": 1000,
  "checkpoint_sha256": "e3eb0aabeaf1f372fbcb2f23bf94625b9fd2a1424504818a02b5d37a4522d6b1",
  "data_sha256": "51ca348f81e4c280d10942cf2fa0d4dd1ce688a16b0088283069b54bbaaf6f97",
  "source_revision": "d63f7d219bedd7ab46057b5c4dae7029bd3944c6",
  "primary_eligible": false,
  "device": "cuda",
  "blas_threads": 4,
  "status": "completed",
  "sampling": {
    "count": 5000,
    "seed": 20261016,
    "nfe": 8,
    "batch_size": 32,
    "seconds": 31.14771006500814,
    "seconds_per_image": 0.006229542013001628,
    "precision": "float32",
    "timing_excludes": "model load and warmup",
    "clipped_fraction": 0.008473698049783707,
    "peak_memory_bytes": 60721664
  },
  "metrics": {
    "frechet_inception_distance": 162.82255470910587,
    "kernel_inception_distance_mean": 0.17437845706939697,
    "kernel_inception_distance_std": 0.0024655637044763345
  },
  "validation_velocity_mse": 0.21588426921516657,
  "metric_seconds": 19.132179868000094,
  "id": "study/main/cifar10-onn-s101/evaluations/test-nfe8",
  "artifact": "runs/study/main/cifar10-onn-s101/evaluations/test-nfe8/evaluation.json"
}
```

## Fixed sample grid

![First 64 generated samples](../figures/evaluation-study--main--cifar10-onn-s101--evaluations--test-nfe8.png)

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
