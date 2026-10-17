# Evaluation: study/pilot/mnist-onn-s101/evaluations/validation

## Purpose and interpretation

This evaluation measures a fixed checkpoint and fixed noise population. KID subset SD describes metric subsampling variability; it is not a confidence interval across trained seeds. The first 64 images are shown without selection.

## Recorded evidence

```json
{
  "dataset": "mnist",
  "family": "onn",
  "training_seed": 101,
  "split": "validation",
  "intervention": "none",
  "generated_count": 512,
  "real_count": 5000,
  "sampling_seed": 20261016,
  "metric_seed": 20261016,
  "nfe": 32,
  "kid_subsets": 100,
  "kid_subset_size": 512,
  "checkpoint_sha256": "f11cdc5df1a58a9be3dd5973d540755c240dcb1c67246627de8aaf8229e0b7e6",
  "data_sha256": "39da09b6ccbbfae8c6424de59b611882eba7f0fb590999ef1851e80f2d027b1c",
  "source_revision": "10f468893bb3293d617ecc2d1493bb1d7f5a3228",
  "primary_eligible": false,
  "device": "cuda",
  "status": "completed",
  "sampling": {
    "count": 512,
    "seed": 20261016,
    "nfe": 32,
    "batch_size": 32,
    "seconds": 14.26068795399624,
    "seconds_per_image": 0.027852906160148905,
    "precision": "float32",
    "timing_excludes": "model load and warmup",
    "clipped_fraction": 0.37238919734954834,
    "peak_memory_bytes": 55247872
  },
  "metrics": {
    "pixel_sliced_wasserstein": 0.07142370551176533,
    "projections": 128
  },
  "validation_velocity_mse": 0.2315111868083477,
  "metric_seconds": 0.2839574440004071,
  "id": "study/pilot/mnist-onn-s101/evaluations/validation",
  "artifact": "runs/study/pilot/mnist-onn-s101/evaluations/validation/evaluation.json"
}
```

## Fixed sample grid

![First 64 generated samples](../figures/evaluation-study--pilot--mnist-onn-s101--evaluations--validation.png)

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
