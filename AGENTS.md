# Agent Instructions for TrajWeave

These instructions apply to this repository only.

## Project Direction

TrajWeave is being rebuilt from a VERL-derived codebase into a Multi-Agent LLM RL framework. Keep the VERL core architecture usable while moving new MASRL logic into a separate project layer.

## Repository Boundaries

- Keep `verl/` as the backend training stack unless a change directly supports TrajWeave.
- Do not add upstream VERL recipe, Docker, CI, or documentation bulk back into the repo without a direct MASRL reason.
- Prefer new TrajWeave logic under a future `trajweave/` package instead of placing product-level MASRL orchestration inside `verl/trainer/ppo`.
- Preserve upstream copyright and license headers in imported VERL source files.

## Change Rules

- Read relevant code and configs before editing.
- Keep cleanup and behavior changes in separate commits when possible.
- Do not delete tests that protect `DataProto`, `single_controller`, `trainer`, or worker behavior unless replacement tests are added.
- For large repository cleanup, prefer deleting obvious upstream project shell first, then pruning examples/docs/scripts after the backend still imports and compiles.

## Verification

For structural cleanup, at minimum run:

```bash
python -m compileall -q verl
python -m pytest tests/test_protocol_on_cpu.py tests/trainer/test_multi_trajectories_advantage_on_cpu.py
```

If dependencies are missing, report that explicitly instead of silently skipping verification.
