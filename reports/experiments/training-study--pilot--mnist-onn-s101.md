# Training: study/pilot/mnist-onn-s101

## Purpose and interpretation

This run fits the pre-specified velocity MSE objective. Completion and held-out error establish execution; generation quality requires its separate evaluation. No checkpoint was selected by best validation or test score.

## Recorded evidence

```json
{
  "status": "completed",
  "dataset": "mnist",
  "family": "onn",
  "seed": 101,
  "step": 3736,
  "train_seconds": 1200.245444398999,
  "parameters": 1289488,
  "device": "cuda",
  "initial_validation_mse": 2.2877872735261917,
  "final_validation_mse": 0.2315111868083477,
  "images_per_second": 398.4251739771892,
  "source_revisions": [
    "b0eb1f5a5fca9604e3751cebf10a70a0ffed39c4"
  ],
  "peak_memory_bytes": 203779584,
  "checkpoint_sha256": "f11cdc5df1a58a9be3dd5973d540755c240dcb1c67246627de8aaf8229e0b7e6",
  "failure": null,
  "id": "study/pilot/mnist-onn-s101",
  "artifact": "runs/study/pilot/mnist-onn-s101/training.json"
}
```

## Exact configuration

```json
{
  "model": {
    "family": "onn",
    "channels": 1,
    "size": 28,
    "width": 128,
    "depth": 4,
    "patch": 4,
    "rotating_dim": 4,
    "inner_steps": 4,
    "step_size": 0.1,
    "ffn_hidden": 0
  },
  "dataset": "mnist",
  "seed": 101,
  "device": "cuda",
  "batch_size": 128,
  "microbatch": 32,
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "clip_grad": 1.0,
  "ema_decay": 0.999,
  "minutes": 20.0,
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

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](../../docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
