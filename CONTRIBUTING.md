# Contributing to TrajWeave

Start with a small, reproducible change and an existing recipe close to your use case. Explain the concrete behavior that changes and include the configuration needed to exercise it.

## Project boundaries

Keep MAS orchestration, task environments, trajectory schemas, reward rules, and paper-specific credit in `trajweave/`. Keep `verl/` usable as a training backend; change it only when a shared backend contract requires it. Preserve upstream copyright and license headers and add provenance for adapted code.

Use a dedicated conda environment and install the project in editable mode. The [getting-started guide](docs/getting-started.md) includes CPU and GPU dependency constraints.

## Local verification

From the repository root, run the lightweight release guard and backend protocol checks:

```bash
python scripts/check_repo_hygiene.py
python -m compileall -q verl trajweave
python -m pytest -q tests/test_protocol_on_cpu.py \
  tests/trainer/test_multi_trajectories_advantage_on_cpu.py \
  tests/trajweave/test_public_config_portability_on_cpu.py
```

Run the tests related to the component you changed. Broader algorithm tests live in `tests/trajweave/`; TransferQueue integration tests require a working local Ray runtime. GPU tests need explicit hardware and local model assets. Do not describe skipped tests or a generated command plan as completed training.

For training changes, inspect online trajectories, reward/advantage values, finite losses, expected nonzero gradients, policy-group routing, weight synchronization, and checkpoints. Report zero-gradient cases and resume limitations explicitly.

## Portable configurations

- Use relative output paths and public model identifiers as defaults.
- Use `TRAJWEAVE_QWEN05B_INSTRUCT_PATH`, `TRAJWEAVE_QWEN05B_BASE_PATH`, and `TRAJWEAVE_QWEN15B_INSTRUCT_PATH` for local Qwen model overrides. Structured role model references are resolved before recipe validation; Hydra overrides retain deferred resolution.
- Let the caller select `CUDA_VISIBLE_DEVICES`. Do not embed contributor-specific GPU indices or cluster directories in presets.
- Keep private data, logs, models, worktrees, and local notes out of Git. Do not remove test fixtures or license notices just to satisfy a scanner.
- Prefer a small regression test that checks the affected contract over a test that merely repeats the implementation.

## Pull requests

Describe the original problem, resulting behavior, relevant tests, and known limits. Include any new dependency or compatibility requirement. Keep historical experiments in development documentation and keep the project README focused on users.
