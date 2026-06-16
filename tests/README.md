# Tests

This directory keeps a reduced test set for the retained VERL backend.

Currently kept categories:

- `tests/test_protocol*_on_cpu.py`: DataProto and protocol checks.
- `tests/test_base_config_on_cpu.py`: base configuration checks.
- `tests/single_controller/`: controller and worker-group behavior.
- `tests/trainer/`: trainer algorithm utilities.
- `tests/workers/`: worker-level helpers.
- `tests/tools/`: tool abstractions useful for MASRL.
- `tests/utils/`: backend utility tests retained from VERL.

Removed categories:

- upstream CI sanity checks.
- special GPU/NPU/e2e test scripts.
- upstream model/checkpoint/plugin test suites that are not part of the first TrajWeave cleanup base.

Recommended minimal cleanup verification:

```bash
python -m compileall -q verl
python -m pytest tests/test_protocol_on_cpu.py tests/trainer/test_multi_trajectories_advantage_on_cpu.py
```
