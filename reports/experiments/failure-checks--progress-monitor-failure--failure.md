# Failure: checks/progress-monitor-failure/failure

## Purpose and interpretation

This attempt did not complete normally. Its existence is retained; missing metrics are not replaced by zero or silently removed from the planned study. Diagnose the recorded error before any authorized retry.

## Recorded evidence

```json
{
  "kind": "read-only-progress-monitor",
  "status": "failed",
  "timestamp": "2026-10-17T19:09:58.889005+00:00",
  "error": "IndexError: list index out of range",
  "traceback": "File <stdin>, line 20, in <module>",
  "context": "The separate read-only monitor exited during the final real-versus-real control, whose newly created run.log had no progress entries. The study process continued independently.",
  "impact": "No training, sampling, metric, checkpoint, or budget state was modified by the monitor.",
  "resolution": "Use direct artifact/process checks that tolerate an empty log; no scientific run is restarted.",
  "id": "checks/progress-monitor-failure/failure",
  "artifact": "runs/checks/progress-monitor-failure/failure.json"
}
```

## Method choices and learning

The shared optimizer, data policy, architecture, and stopping rule were chosen before final comparison and are described in the technical paper. The 3e-4 learning rate is a conservative AdamW starting point, not a tuned optimum. Microbatches bound laptop memory; four internal updates bound recurrent cost; EMA smooths noisy parameter updates. These choices trade study breadth for a fixed local compute budget. This record supports only the measurements above; causes of quality differences require the trained controls and uncertainty analysis. Full trajectories, populations, RNG states, and logs remain in the referenced artifact bundle. Sources: [protocol](https://github.com/BurnyCoder/flow-matching-with-oscillatory-neural-networks-instead-of-transformer/blob/main/docs/protocol.md), [TorchCFM](https://github.com/atong01/conditional-flow-matching), [AKOrN](https://arxiv.org/html/2410.13821v2).
