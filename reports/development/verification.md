# Development verification record — 2026-10-17

These checks establish implementation behavior, not comparative sample quality. The primary CIFAR-10 study was incomplete when this record was written.

## Sequence and source provenance

The public repository and pre-specified protocol were pushed in commit `55dc2e4`. Core numerical/training implementation (`2c9a45d`) and CLI (`a3dd622`) were pushed before the first MNIST optimizer run. The first GPU run used `a3dd622`. Subsequent evaluation/deadline support was pushed in `5b69c90`; the fixed-batch/pilot scheduler was pushed in `b0eb1f5`. Full reporting/export support was pushed in `af53767`. Git history and run `source_revisions` retain the exact full revisions.

The originating user instructions and implementation plan specified local training, a 24 GPU-hour ceiling, public code/checkpoints, and exploratory paired inference. No runtime LLM was introduced. The implementation used TorchCFM, torchdiffeq, torch-fidelity, PyTorch, POT, and SciPy instead of recreating their maintained algorithms. AKOrN source had no repository license during inspection, so its published equations were implemented independently.

## Experiments and observations

| Check | Observed result | Interpretation |
| --- | --- | --- |
| MNIST GPU smoke, 50 updates | Completed; about 17.68 training seconds; peak allocated memory 203,779,584 bytes | CUDA forward/backward/optimizer/checkpoint path executes |
| Resume to 100 updates | Completed; cumulative training time 33.96 seconds; EMA validation velocity MSE 2.00869 | Saved run state can continue on GPU; early EMA remains close to initialization |
| Exact CPU resume test | Resumed and uninterrupted model/EMA/noise-generator states match exactly | Optimizer-boundary persistence is complete in the tested CPU setting |
| Fixed batch, 500 updates | MSE 2.255753 → 0.000602648; ratio 0.000267161 | Full oscillator model can learn fixed production-path velocity targets |
| Same-input determinism | Exact repeated predictions in the fixed-batch GPU check | No hidden state is carried across these forward calls |
| Geometry | Maximum norm error 5.96e-8; maximum tangent inner-product magnitude 3.05e-7 | FP32 spherical operations meet the predeclared 1e-5 numerical tolerance |
| CPU sampling | 16 samples at 32 NFE completed; full float outputs and grid saved | GPU is not required for inference |
| GPU sampling/evaluation smoke | 128 samples at 32 NFE completed; MNIST pixel SWD 1.101009 | Full evaluation path works; this noise-like early model is not a quality result |
| Portable checkpoint reload | Original and safetensors-export CPU samples exactly equal; maximum difference zero | Safe export preserves the actual inference function |
| Budget enforcement | Isolated deliberately blocked worker was terminated; reservation retained | The ceiling is enforced independently of cooperative model code |
| Report/PDF workflow | Markdown and PDF records generated; first page and equations inspected | Pandoc/Typst artifacts exist and render the mathematical method |
| Planning-power reproduction | 0.2406 at standardized effect 0.8 using 20,000 Gaussian simulations | Reproduces the pre-training low-power warning; not observed empirical power |

The CPU preview after 100 optimizer updates visually remained noise-like. This was expected from a very short development run and a slowly updated EMA; it was retained rather than selected away. A longer fixed-time pilot, not an extension based on significance, was already part of the protocol.

## Issues, decisions, and limits

The initial resume invocation omitted the required path after `--resume`; argparse rejected it before computation. The corrected documented form is `ofm train --resume <run-directory>`. Import-order lint findings were repaired while implementation progressed; CI run 38977501341 failed only an import-format check, repaired in `549327f`. A third-party POT SyntaxWarning did not affect numerical tests. No nonfinite-loss or gradient failure was observed in the checks above. A CLI help traceback was removed, and the scheduler was hardened to load actual frozen pilot settings instead of allowing later environment-default drift.

Tests for missing flow/training/evaluation/reporting/publication interfaces were introduced before those implementations, and their initial collection failures were observed. Some initial model tests could only execute after CUDA dependencies finished installing; this record does not claim every line followed a strict observed red-green sequence. The initial complete local suite contained 27 passing tests; additional help/freeze regression tests were added for the two later repairs. CI must pass on the merged implementation.

The 20-minute-per-architecture MNIST pilot uses the pre-specified optimizer settings and validates generalization/visual behavior before freezing CIFAR settings. No CIFAR result was used to choose these parameters. A fixed per-update EMA decay and differing architecture throughput are potential comparative confounders; they are reported as limitations rather than silently adjusted after outcomes. GPU utilization and system background load are not hardware-independent compute measures.

## Completed pilot and metric preflight

All three MNIST pilots completed their original 20-minute budgets. Their fixed first-64 sample grids were inspected: they contain recognizable but rough digits, including ambiguous shapes. Full measurements and images are retained in the generated experiment records rather than repeated here. The freeze was written at `20261016T233806.675239Z`, source `7926ff79c12be5d85eadb49b81aad5d3acfaedcd`, with an empty changes list. CIFAR scheduling inherits these actual saved settings. The CIFAR decoded-pixel audit retained 45,000 training, 5,000 validation, and 10,000 test examples, with no exact cross-split overlap found; this is not a near-duplicate audit.

Before any CIFAR optimization, an untrained oscillator generated 128 validation-check samples. The real torch-fidelity feature/KID/FID path completed, and repeating all metrics from the saved tensors yielded a maximum absolute difference of zero. This preflight used 128-image KID subsets and validation data, so it is not a primary 5,000-image test comparison. Its artifact group is `runs/checks/cifar10-metric-smoke`.

The publicly downloaded trained oscillator pilot also generated 64 images on CPU at 32 velocity evaluations. Two fixed CPU input/time pairs gave exactly equal predictions from the public safetensors weights and the original EMA checkpoint. The saved public-weight SHA-256 is `06201135c2255dcc8b2ee988f9ce128ce6fe2e6eb35f3fd142b3a4db415fec96`; the checkpoint identity and all preview tensors are retained under the pilot's `cpu-public-check` directory. This checks a published trained artifact, in addition to the earlier development export test.

The implementation review and expanded regression suite concluded with 33 passing tests and a passing Ruff check. The frozen dependency sync completed before the CIFAR schedule. These checks justify starting the declared comparison; they do not establish an oscillator quality advantage.

Sources: [protocol](../../docs/protocol.md), [implementation contracts](../../docs/implementation.md), [Karpathy's training recipe](https://karpathy.github.io/2019/04/25/recipe/), [PyTorch serialization](https://docs.pytorch.org/docs/2.11/notes/serialization.html), [SciPy permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
