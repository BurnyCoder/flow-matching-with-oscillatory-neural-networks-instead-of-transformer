# Control: study/controls/real-versus-real

## Purpose and interpretation

This diagnostic checks the contract recorded below, such as checkpoint reconstruction, metric repetition, or identity-intervention behavior. It is excluded from primary paired-seed inference and cannot establish an oscillator mechanism by itself.

## Recorded evidence

```json
{
  "status": "completed",
  "kind": "real-versus-real",
  "seed": 20261016,
  "left_count": 5000,
  "right_count": 5000,
  "data_sha256": "51ca348f81e4c280d10942cf2fa0d4dd1ce688a16b0088283069b54bbaaf6f97",
  "metrics": {
    "frechet_inception_distance": 10.272234758028844,
    "kernel_inception_distance_mean": -2.55584716796875e-05,
    "kernel_inception_distance_std": 0.00010886578191701542
  },
  "repeat_metrics": {
    "frechet_inception_distance": 10.272234758028844,
    "kernel_inception_distance_mean": -2.55584716796875e-05,
    "kernel_inception_distance_std": 0.00010886578191701542
  },
  "repeat_max_absolute_difference": 0.0,
  "id": "study/controls/real-versus-real",
  "artifact": "runs/study/controls/real-versus-real/control.json"
}
```

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
