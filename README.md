# TrajWeave

TrajWeave is a Multi-Agent LLM reinforcement learning framework under active development. The project starts from the VERL training stack and keeps its core distributed RL architecture, while cleaning the repository into a smaller base for MASRL-oriented secondary development.

## Current Scope

This repository currently keeps the VERL core as the training backend:

- `verl/protocol.py`: DataProto and batch exchange protocol.
- `verl/trainer/main_ppo.py`: PPO-like training entrypoint.
- `verl/trainer/ppo/ray_trainer.py`: Ray-based RL training loop.
- `verl/trainer/ppo/core_algos.py`: advantage estimators and policy losses.
- `verl/workers/`: actor, rollout, reference, critic, reward and engine workers.
- `verl/single_controller/`: controller and worker-group infrastructure.
- `verl/tools/` and `verl/experimental/agent_loop/`: tool and agent-loop primitives useful for MASRL.

The removed upstream recipe, CI, Docker, and documentation material can be restored from upstream VERL when needed, but they are no longer part of this project base.

## Direction

TrajWeave will add a MASRL layer above the VERL backend:

```text
Task / Benchmark
  -> TeamSpec / AgentSpec / PolicyGroup
  -> Orchestra
  -> Multi-Agent Rollout
  -> MultiAgentTrajectory
  -> Reward + CreditAssigner
  -> VERL Adapter
  -> DataProto
  -> VERL Trainer / Workers
```

Planned first recipe:

- `Solver -> Verifier -> Solver refine -> final answer`
- math exact-match reward
- `global_broadcast`, `last_actor_only`, and agent-wise normalization credit assignment
- shared-model first, heterogeneous policy groups later

## Development Notes

This repository is intentionally not a full VERL mirror anymore. Avoid adding upstream-only recipes, hardware-specific CI, and large Docker matrices back into the main tree unless they directly support the TrajWeave MASRL path.

Recommended install pattern for development:

```bash
pip install --no-deps -e .
```

Minimal verification after structural edits:

```bash
python -m compileall -q verl
python -m pytest tests/test_protocol_on_cpu.py tests/trainer/test_multi_trajectories_advantage_on_cpu.py
```

## Attribution

TrajWeave is built on top of code imported from VERL / HybridFlow. The original source is Apache-2.0 licensed. Keep upstream copyright headers in copied source files.
