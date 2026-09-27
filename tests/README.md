# Tests

This directory keeps a reduced test set for the retained VERL backend.

Currently kept categories:

- `tests/test_protocol*_on_cpu.py`: DataProto and protocol checks.
- `tests/test_base_config_on_cpu.py`: base configuration checks.
- `tests/single_controller/`: controller and worker-group behavior.
- `tests/trainer/`: trainer algorithm utilities.
- `tests/trajweave/`: paper recipe, AgentLoop bridge, TransferQueue, padding, and credit regressions.
- `tests/workers/`: worker-level helpers.
- `tests/tools/`: tool abstractions useful for MASRL.
- `tests/utils/`: backend utility tests retained from VERL.

The CPU commands below cover only part of this test suite. Retained GPU, NPU, distributed, and end-to-end tests require their own hardware and dependencies; they are not included in the CPU smoke check. Upstream CI and test coverage are not imported in full.

Recommended minimal cleanup verification:

```bash
python -m compileall -q verl
python -m pytest tests/test_protocol_on_cpu.py tests/trainer/test_multi_trajectories_advantage_on_cpu.py
pytest -q \
  tests/trajweave/test_atgrpo_recipe_on_cpu.py \
  tests/trajweave/test_atgrpo_agent_loop_tq_on_cpu.py \
  tests/trajweave/test_matpo_recipe_on_cpu.py \
  tests/trajweave/test_matpo_verl_integration_on_cpu.py \
  tests/trajweave/test_verl_workflow_runtime_on_cpu.py
```

`test_atgrpo_agent_loop_tq_on_cpu.py` writes and reads an in-process real TransferQueue and executes the AgentLoop worker path, but it is a CPU bridge/contract test rather than a full `main_ppo` Trainer run.
