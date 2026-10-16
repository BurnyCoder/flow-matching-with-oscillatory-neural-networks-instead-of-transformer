# Control: checks/cifar10-metric-smoke/evaluations/validation/repeat

## Purpose and interpretation

This diagnostic checks metric scale/repetition or identity-intervention behavior. It is excluded from primary paired-seed inference and cannot establish an oscillator mechanism by itself.

## Recorded evidence

```json
{
  "kind": "validation-metric-reproducibility",
  "status": "completed",
  "generated_count": 128,
  "real_count": 5000,
  "first_metrics": {
    "frechet_inception_distance": 487.66587247902373,
    "kernel_inception_distance_mean": 0.6097407078742981,
    "kernel_inception_distance_std": 0.005960302714145461
  },
  "repeat_metrics": {
    "frechet_inception_distance": 487.66587247902373,
    "kernel_inception_distance_mean": 0.6097407078742981,
    "kernel_inception_distance_std": 0.005960302714145461
  },
  "repeat_max_absolute_difference": 0.0,
  "data_sha256": "51ca348f81e4c280d10942cf2fa0d4dd1ce688a16b0088283069b54bbaaf6f97",
  "source_evaluation": "runs/checks/cifar10-metric-smoke/evaluations/validation",
  "id": "checks/cifar10-metric-smoke/evaluations/validation/repeat",
  "artifact": "runs/checks/cifar10-metric-smoke/evaluations/validation/repeat/control.json"
}
```

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](../../docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
