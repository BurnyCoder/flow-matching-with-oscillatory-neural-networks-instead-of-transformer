# Evaluation: study/main/cifar10-onn-s102/evaluations/shuffle

## Purpose and interpretation

This evaluation measures a fixed checkpoint and fixed noise population. KID subset SD describes metric subsampling variability; it is not a confidence interval across trained seeds. The first 64 images are shown without selection.

## Recorded evidence

```json
{
  "dataset": "cifar10",
  "family": "onn",
  "training_seed": 102,
  "split": "test",
  "intervention": "shuffle",
  "generated_count": 5000,
  "real_count": 10000,
  "sampling_seed": 20261016,
  "metric_seed": 20261016,
  "nfe": 32,
  "kid_subsets": 100,
  "kid_subset_size": 1000,
  "checkpoint_sha256": "e5cf67558e832212ded9b04225e34091cd62715d9df0773c5d22685da64e8366",
  "data_sha256": "51ca348f81e4c280d10942cf2fa0d4dd1ce688a16b0088283069b54bbaaf6f97",
  "source_revision": "b49e52bb12834bd0f151a4f45a36462182130a8d",
  "primary_eligible": false,
  "device": "cuda",
  "blas_threads": 4,
  "status": "completed",
  "sampling": {
    "count": 5000,
    "seed": 20261016,
    "nfe": 32,
    "batch_size": 32,
    "seconds": 135.94804691401077,
    "seconds_per_image": 0.027189609382802155,
    "precision": "float32",
    "timing_excludes": "model load and warmup",
    "clipped_fraction": 0.004106835927814245,
    "peak_memory_bytes": 60721664
  },
  "metrics": {
    "frechet_inception_distance": 233.55040683569993,
    "kernel_inception_distance_mean": 0.2577728080749512,
    "kernel_inception_distance_std": 0.0036487067006191246
  },
  "validation_velocity_mse": 0.24558429699391127,
  "metric_seconds": 20.21528776199557,
  "id": "study/main/cifar10-onn-s102/evaluations/shuffle",
  "artifact": "runs/study/main/cifar10-onn-s102/evaluations/shuffle/evaluation.json"
}
```

## Fixed sample grid

![First 64 generated samples](../figures/evaluation-study--main--cifar10-onn-s102--evaluations--shuffle.png)

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
