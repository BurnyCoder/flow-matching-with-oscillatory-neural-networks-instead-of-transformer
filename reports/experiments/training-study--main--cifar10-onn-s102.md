# Training: study/main/cifar10-onn-s102

## Purpose and interpretation

This run fits the pre-specified velocity MSE objective. Completion and held-out error establish execution; generation quality requires its separate evaluation. No checkpoint was selected by best validation or test score.

## Recorded evidence

```json
{
  "status": "completed",
  "dataset": "cifar10",
  "family": "onn",
  "seed": 102,
  "step": 8588,
  "train_seconds": 2700.0862612680066,
  "parameters": 1297712,
  "device": "cuda",
  "initial_validation_mse": 1.5781860128045082,
  "final_validation_mse": 0.22063074633479118,
  "images_per_second": 407.1218078357863,
  "source_revisions": [
    "c749856d1c9fe07d6ce2f39d01de78f664aea9a8"
  ],
  "peak_memory_bytes": 239299072,
  "checkpoint_sha256": "e5cf67558e832212ded9b04225e34091cd62715d9df0773c5d22685da64e8366",
  "failure": null,
  "id": "study/main/cifar10-onn-s102",
  "artifact": "runs/study/main/cifar10-onn-s102/training.json"
}
```

## Exact configuration

```json
{
  "model": {
    "family": "onn",
    "channels": 3,
    "size": 32,
    "width": 128,
    "depth": 4,
    "patch": 4,
    "rotating_dim": 4,
    "inner_steps": 4,
    "step_size": 0.1,
    "ffn_hidden": 0
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
