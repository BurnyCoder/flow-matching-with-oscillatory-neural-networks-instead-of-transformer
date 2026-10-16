# Training: study/pilot/mnist-conv-s101

## Purpose and interpretation

This run fits the pre-specified velocity MSE objective. Completion and held-out error establish execution; generation quality requires its separate evaluation. No checkpoint was selected by best validation or test score.

## Recorded evidence

```json
{
  "status": "completed",
  "dataset": "mnist",
  "family": "conv",
  "seed": 101,
  "step": 5245,
  "train_seconds": 1200.0462538739957,
  "parameters": 1288464,
  "device": "cuda",
  "initial_validation_mse": 2.4294664412736893,
  "final_validation_mse": 0.21519694197922945,
  "images_per_second": 559.4451029139186,
  "source_revisions": [
    "7926ff79c12be5d85eadb49b81aad5d3acfaedcd"
  ],
  "peak_memory_bytes": 170023936,
  "checkpoint_sha256": "3fffc15466b2561ab3b0ebe9ad66272c4c4e3fe99e0f13602910d1f7da80b651",
  "failure": null,
  "id": "study/pilot/mnist-conv-s101",
  "artifact": "runs/study/pilot/mnist-conv-s101/training.json"
}
```

## Exact configuration

```json
{
  "model": {
    "family": "conv",
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
