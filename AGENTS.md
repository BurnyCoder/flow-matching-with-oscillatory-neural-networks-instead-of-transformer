# Contributor instructions

Work in this checkout. Use Python 3.12, uv, and the local virtual environment.
Read the README for commands and docs/protocol.md for the pre-specified experiment.
The user's instructions in the originating chat take precedence over this file.

- Keep the pipeline wrapper readable; put implementation in reusable phase modules.
- Write behavior tests before implementing or repairing numerical/stateful behavior.
- Explain local mechanics and global purpose in module/function docstrings and code comments; cite primary sources.
- Push code and the protocol before training. Push fixes before rerunning affected experiments.
- Keep meaningful commits, feature PRs, and merges. Never commit credentials or downloaded datasets.
- During active workers, merge PRs without automatic checkout/deletion; avoid transient source removal while another process imports modules.
- Preserve all runs, failures, seed results, and protocol deviations. Do not stop or extend experiments based on significance.
- Treat the persisted 24-hour compute ledger as cumulative across restarts.
- Keep the same run-root ledger for all development, training, sampling, and evaluation; do not reset it to complete an arm.
- Use `ofm study --dry-run` to inspect the immutable schedule. Preserve freeze records, completed evaluations, and failed attempts.
- Distinguish internal oscillator updates from the outer sampling ODE; forward calls must be stateless.
- Use independent RNG streams for initialization, data ordering, flow noise, and evaluation.
- Keep README claims tied to implemented behavior and measured results. Detailed results belong in reports, methods in docs.
- Regenerate reports from artifacts. Primary inference requires all six declared pairs; KID subset SD is not seed uncertainty.
- Credit Libor Burian, independent researcher, as the paper's author.
- Source empirical claims to the recorded experiment evidence and methodological claims to primary references; keep speculation explicit.
- Publish through the positive artifact allowlist and verify exported prediction equivalence; never upload arbitrary run-directory contents.
- The runtime uses no LLM. If one is introduced, capture complete prompts/responses to timestamped logs without credentials.
- Review correctness, security, maintainability, reproducibility, and inference validity once before release.

Do not copy source from the AKOrN repository: no repository license was present during inspection.
Implement the cited published equations independently using PyTorch primitives.
