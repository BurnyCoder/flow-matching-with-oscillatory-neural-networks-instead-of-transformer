# Evaluation: study/main/cifar10-dit-s102/evaluations/test-nfe32

## Purpose and interpretation

This evaluation measures a fixed checkpoint and fixed noise population. KID subset SD describes metric subsampling variability; it is not a confidence interval across trained seeds. The first 64 images are shown without selection.

## Recorded evidence

```json
{
  "dataset": "cifar10",
  "family": "dit",
  "training_seed": 102,
  "split": "test",
  "intervention": "none",
  "generated_count": 5000,
  "real_count": 10000,
  "sampling_seed": 20261016,
  "metric_seed": 20261016,
  "nfe": 32,
  "kid_subsets": 100,
  "kid_subset_size": 1000,
  "checkpoint_sha256": "a87b3e659c6398bf2b2791e05297293a8c154ada863778310148018b4f940c15",
  "data_sha256": "51ca348f81e4c280d10942cf2fa0d4dd1ce688a16b0088283069b54bbaaf6f97",
  "source_revision": "0afa17670bc7339fd28a2beea4396e47d30b1d9c",
  "primary_eligible": true,
  "device": "cuda",
  "blas_threads": 4,
  "status": "completed",
  "sampling": {
    "count": 5000,
    "seed": 20261016,
    "nfe": 32,
    "batch_size": 32,
    "seconds": 48.89814832000411,
    "seconds_per_image": 0.009779629664000823,
    "precision": "float32",
    "timing_excludes": "model load and warmup",
    "clipped_fraction": 0.015701953321695328,
    "peak_memory_bytes": 58955776
  },
  "metrics": {
    "frechet_inception_distance": 103.12542282658637,
    "kernel_inception_distance_mean": 0.08995173454284668,
    "kernel_inception_distance_std": 0.00169647956062002
  },
  "validation_velocity_mse": 0.1970291268080473,
  "metric_seconds": 18.674224670001422,
  "id": "study/main/cifar10-dit-s102/evaluations/test-nfe32",
  "artifact": "runs/study/main/cifar10-dit-s102/evaluations/test-nfe32/evaluation.json"
}
```

## Fixed sample grid

![First 64 generated samples](../figures/evaluation-study--main--cifar10-dit-s102--evaluations--test-nfe32.png)

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
