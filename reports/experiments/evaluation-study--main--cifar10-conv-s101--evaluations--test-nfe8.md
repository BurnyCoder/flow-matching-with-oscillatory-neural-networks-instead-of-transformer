# Evaluation: study/main/cifar10-conv-s101/evaluations/test-nfe8

## Purpose and interpretation

This evaluation measures a fixed checkpoint and fixed noise population. KID subset SD describes metric subsampling variability; it is not a confidence interval across trained seeds. The first 64 images are shown without selection.

## Recorded evidence

```json
{
  "dataset": "cifar10",
  "family": "conv",
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
  "checkpoint_sha256": "d283104b77bb38b474d78f48ee64bf89e90cd8ecb2fc32d2cced963261e28cbd",
  "data_sha256": "51ca348f81e4c280d10942cf2fa0d4dd1ce688a16b0088283069b54bbaaf6f97",
  "source_revision": "b49e52bb12834bd0f151a4f45a36462182130a8d",
  "primary_eligible": false,
  "device": "cuda",
  "blas_threads": 4,
  "status": "completed",
  "sampling": {
    "count": 5000,
    "seed": 20261016,
    "nfe": 8,
    "batch_size": 32,
    "seconds": 18.49844118001056,
    "seconds_per_image": 0.003699688236002112,
    "precision": "float32",
    "timing_excludes": "model load and warmup",
    "clipped_fraction": 0.007258593570441008,
    "peak_memory_bytes": 59929088
  },
  "metrics": {
    "frechet_inception_distance": 122.5654617919215,
    "kernel_inception_distance_mean": 0.1270967674255371,
    "kernel_inception_distance_std": 0.00227732058444641
  },
  "validation_velocity_mse": 0.20359265431761742,
  "metric_seconds": 19.94960921900929,
  "id": "study/main/cifar10-conv-s101/evaluations/test-nfe8",
  "artifact": "runs/study/main/cifar10-conv-s101/evaluations/test-nfe8/evaluation.json"
}
```

## Fixed sample grid

![First 64 generated samples](../figures/evaluation-study--main--cifar10-conv-s101--evaluations--test-nfe8.png)

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
