# Diagnostic: study/checks/fixed-batch

## Purpose and interpretation

This fixed-batch probe checks optimization, finite geometry, and deterministic predictions. It intentionally memorizes 16 training examples with fixed noise and times. Passing does not establish generalization or generative quality.

## Recorded evidence

```json
{
  "status": "completed",
  "steps": 500,
  "initial_loss": 2.2557530403137207,
  "final_loss": 0.0006026483024470508,
  "loss_ratio": 0.00026716058525770044,
  "deterministic": true,
  "norm_max_error": 5.960464477539063e-08,
  "tangent_max_error": 3.0547380447387695e-07,
  "seconds": 42.347297642998456,
  "passed": true,
  "id": "study/checks/fixed-batch",
  "artifact": "runs/study/checks/fixed-batch/diagnostic.json"
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
  "batch_size": 16,
  "microbatch": 16,
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "clip_grad": 1.0,
  "ema_decay": 0.999,
  "minutes": 45.0,
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
