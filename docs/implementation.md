# Implementation and artifact contracts

The public entry point is `ofm` (`src/ofm/cli.py`). It loads `.env` only from the current directory, captures complete emitted Python stdout/stderr, and calls the named phases in `pipeline.py`. The wrapper imports configuration, data, model, training, sampling, evaluation, reporting, and publication modules. Numerical code has no dependency on the CLI. CLI help is the definitive argument reference.

## Model contract

`ofm.models.factory.build_model(ModelConfig(...))` returns a `torch.nn.Module`. Inputs are NCHW images and either a scalar tensor time or one time per batch element. Outputs are unbounded NCHW velocities. Spatial dimensions/channels are checked against the saved configuration. `ofm.checkpoints.load_generator(path, device)` loads an EMA generator from a run, direct checkpoint, or exported safetensors directory.

`models/backbone.py` names encoding, feature processing, and decoding. `oscillator.py` owns tangent projection, normalization, internal dynamics, and readout; `transformer.py` uses PyTorch attention for the explicit baseline. `embeddings.py` shares continuous-time features, positional grids, and unpatchifying. The model never stores a live hidden state between forward calls. The fixed spatial permutation buffer is used only when `model.intervention = "shuffle"`; `none` and `identity` have identical computation.

The matching factory creates reference parameter counts under a restored RNG context, computes the transformer MLP count increment, and picks its closest positive multiple-of-eight width. This avoids consuming the training initialization RNG while searching. Ablations construct a seeded reference ONN on CPU and copy every retained tensor, controlling their initialization. The publication/reload path uses strict state-dictionary key checks.

## Training and recovery

`training_state.py` constructs seeded models, AdamW, and PyTorch's EMA wrapper. `training.py` owns the bounded loop. A dedicated CPU permutation generator controls minibatches; a dedicated device generator controls Gaussian noise and uniform flow time. Initialization uses a separate model-specific derived seed. Model-independent validation uses a further stream. The same microbatch size is required for exact stream pairing because noise/time draw ordering depends on chunking.

Snapshots occur only after complete optimizer/EMA updates. They include trainable weights, EMA weights and averaging count, AdamW state, data permutation/cursor/epoch, flow-generator state, global CPU/CUDA RNG states, cumulative training seconds, steps, full configuration, dataset hash, and all source revisions used by resumed segments. Checkpoint loading requests `weights_only=True`. Atomic rename protects the previous complete checkpoint against interrupted writes. A failure in the middle of accumulation retains the previous safe snapshot; lost work is still counted by the budget ledger.

Resume also reads the latest failed-run elapsed time, so recovering older weights does not refund known discarded computation. If a process was hard-killed with an unclosed GPU reservation, the original run horizon is conservatively spent and its checkpoint remains incomplete. The original time-limited arm cannot silently restart from its older saved clock. An explicitly extended standalone configuration is a different, non-primary experiment. Failed compatibility checks do not overwrite the previous configuration or dataset manifest.

Ordinary resume compares the saved scientific configuration and dataset hash. Only stopping limits and logging/checkpoint cadence may differ at the internal API boundary; the CLI exposes stopping-limit overrides. Cross-device training resumption is not an exact-resume guarantee. Exported EMA sampling on CPU is supported independently of the training device.

## Budget and failure semantics

`budget.py` uses a stable lock inode plus an atomically replaced JSON ledger. Overlapping GPU commands cannot reserve simultaneously. Requested occupied time is charged up front; clean exit refunds the unused part. An interrupted `active` reservation is retained in history and never automatically refunded. `deadline.py` waits for cancellation on a pipe in an independent process, then kills only its parent worker at the deadline if cancellation never arrives. This also handles native-code metric stalls; OS scheduling and CUDA teardown are not precision energy accounting.

Training and sampling make cooperative checks before bounded operations so they normally save state and exit before the hard deadline. The hard-stop fallback may leave only the last periodic checkpoint, and its full reservation remains charged. A change to the configured cap is rejected against an existing ledger. Keep the same `OFM_RUNS_ROOT` for the entire authorized study; different roots intentionally represent different studies and must not be used to evade its allocation.

## Measurements and reproducibility boundaries

Evaluation retains raw float samples before metric clipping/quantization. `ImagePopulation` accepts uint8 RGB tensors and delegates Inception preprocessing to torch-fidelity. Real feature-cache keys include the prepared dataset hash and split. Generated features are recomputed, avoiding a stale cache keyed only by a run name. The metric seed, subset count/size, image counts, checkpoint hash, split hash, device, precision, clipping fraction, actual NFE, sampling time, and memory allocation accompany results.

Sampling timing includes generation and host transfer after warmup, excluding model loading and warmup. CUDA synchronization brackets timing. Training throughput is examples processed divided by accumulated in-loop wall time, including periodic checkpoint/log overhead. Peak memory means PyTorch allocated memory, not total device occupancy from other applications. Fixed wall time naturally permits more updates for faster architectures.

The training stopping check occurs at complete optimizer boundaries, so measured durations can exceed the nominal per-run horizon by the final minibatch. This small granularity and other setup/cleanup costs consume contingency within the independently enforced total allocation; reported durations are the actual values, not rounded back to the requested limit. BLAS pools for metrics and power calculations are limited with threadpoolctl to avoid native-library oversubscription.

Numerical determinism is tested on the same runtime. CPU and CUDA random generators need not produce the same noise for the same integer seed, so cross-device sample equality is not claimed. Exact same-device identity-intervention outputs and repeated metric controls are checked separately. KID subset variability, seed uncertainty, and statistical assumptions are specified in the protocol; the reports do not substitute one for another.

## Reporting and release boundaries

`reporting.py` discovers only the requested artifact root, checks that there is at most one eligible primary evaluation per model/seed, and requires all six pairs before primary inference. `report_text.py` composes role-aware evidence records; `report_figures.py` plots observed points and renders Markdown through Pandoc → Typst → PDF. Planning power is reproduced with common Gaussian random numbers across the fixed effect-size scenarios. Reports may be generated while work is incomplete and must say so.

`publishing.py` selects only reported checkpoints, creates fresh export staging, copies a positive allowlist, refuses artifact symlinks, writes safe EMA tensors and model cards, and hashes exported bytes. Original optimizer checkpoints remain available for resumable runs. A dry run produces concrete exports without Hub authentication. Public uploads verify the account name, create/reuse dataset repositories, and add them to one collection. Tokens and environment-file contents are never part of the scientific record. Full downloaded source datasets and disposable caches are not exported.

## Sources

- [PyTorch serialization](https://docs.pytorch.org/docs/2.11/notes/serialization.html), [AMP examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html), [EMA utilities](https://docs.pytorch.org/docs/2.11/optim.html#weight-averaging-swa-and-ema)
- [TorchCFM](https://github.com/atong01/conditional-flow-matching), [torchdiffeq](https://github.com/rtqichen/torchdiffeq), [torch-fidelity API](https://torch-fidelity.readthedocs.io/en/latest/usage_api.html)
- [POT sliced transport](https://pythonot.github.io/gen_modules/ot.sliced.html), [SciPy exact permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
- [uv accelerator installation](https://docs.astral.sh/uv/guides/integration/pytorch/), [safetensors](https://huggingface.co/docs/safetensors/index), [Hub upload](https://huggingface.co/docs/huggingface_hub/en/guides/upload)
- [Pandoc](https://pandoc.org/MANUAL.html), [Typst Python binding](https://github.com/messense/typst-py), [Python subprocess](https://docs.python.org/3/library/subprocess.html)
