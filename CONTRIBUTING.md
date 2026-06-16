# Contributing to TrajWeave

TrajWeave is currently in an early repository-cleanup and architecture-reset phase.

## Priorities

1. Keep the VERL backend importable and testable.
2. Add TrajWeave MASRL abstractions without rewriting VERL trainer internals first.
3. Build one runnable Solver-Verifier Math MVP before adding more paper recipes.
4. Keep examples and docs focused on the TrajWeave path.

## Cleanup Policy

- Remove upstream project shell that does not serve TrajWeave.
- Keep tests that protect the retained backend.
- Do not mix large cleanup with algorithmic changes in the same commit.
- Keep Apache-2.0 attribution and copied source headers.

## Local Checks

Use the smallest relevant checks first:

```bash
python -m compileall -q verl
python -m pytest tests/test_protocol_on_cpu.py tests/trainer/test_multi_trajectories_advantage_on_cpu.py
```

Broaden verification when changing shared backend code.
