# Training: study/main/cifar10-dit-s102

## Purpose and interpretation

This run fits the pre-specified velocity MSE objective. Completion and held-out error establish execution; generation quality requires its separate evaluation. No checkpoint was selected by best validation or test score.

## Recorded evidence

```json
{
  "status": "completed",
  "dataset": "cifar10",
  "family": "dit",
  "seed": 102,
  "step": 17303,
  "train_seconds": 2700.0645746579976,
  "parameters": 1298864,
  "device": "cuda",
  "initial_validation_mse": 1.7011075615882874,
  "final_validation_mse": 0.1970291268080473,
  "images_per_second": 820.2707523321121,
  "source_revisions": [
    "c749856d1c9fe07d6ce2f39d01de78f664aea9a8"
  ],
  "peak_memory_bytes": 161050112,
  "checkpoint_sha256": "a87b3e659c6398bf2b2791e05297293a8c154ada863778310148018b4f940c15",
  "failure": null,
  "id": "study/main/cifar10-dit-s102",
  "artifact": "runs/study/main/cifar10-dit-s102/training.json"
}
```

## Exact configuration

```json
{
  "model": {
    "family": "dit",
    "channels": 3,
    "size": 32,
    "width": 128,
    "depth": 4,
    "patch": 4,
    "rotating_dim": 4,
    "inner_steps": 4,
    "step_size": 0.1,
    "ffn_hidden": 576
  },
  "dataset": "cifar10",
  "seed": 102,
  "device": "cuda",
  "batch_size": 128,
  "microbatch": 32,
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "clip_grad": 1.0,
  "ema_decay": 0.999,
  "minutes": 45,
  "max_steps": 0,
  "validation_size": 5000,
  "data_root": "data",
  "runs_root": "runs",
  "budget_hours": 24.0,
  "log_every": 50,
  "checkpoint_seconds": 300.0,
  "amp": true
}
```

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
