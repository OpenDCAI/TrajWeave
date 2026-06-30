# TrajWeave MAS Layer

TrajWeave keeps VERL as the training backend and adds a separate MAS layer for multi-agent rollout, trajectory storage, reward propagation, and credit assignment.

## Module Boundaries

| Module | Responsibility | Must not own |
| --- | --- | --- |
| `trajweave.core` | Stable specs and trajectory dataclasses. | Policy execution, reward logic, VERL tensors. |
| `trajweave.orchestration` | Agent turn order, shared team context, stop conditions. | Reward computation or advantage normalization. |
| `trajweave.envs` | Task observation and task success evaluation. | Agent scheduling or trainer calls. |
| `trajweave.backends` | Policy backend interface and local smoke backends. | Recipe-specific reward or credit rules. |
| `trajweave.rollout` | Connect team, orchestra, env, backend, and credit assigner. | Algorithm-specific advantage math. |
| `trajweave.credit` | Convert trajectories into training samples. | LLM generation, env stepping, or trainer control flow. |
| `trajweave.backends.verl` | Convert `TrainingSample` objects to VERL `DataProto`, prepare trainer launch commands, and expose the custom AgentLoopManager bridge. | Agent orchestration or paper recipe logic. |
| `trajweave.recipes` | Compose modules into runnable paper recipes. | Shared abstractions that belong in core modules. |
| `trajweave.cli` / `trajweave.runner` | Load YAML and run one configured recipe. | Recipe internals or VERL trainer implementation. |

## DrMAS First Slice

```mermaid
flowchart LR
    A[MathTask] --> B[SolverVerifierOrchestra]
    B --> C[PolicyBackend]
    C --> D[AgentTurn]
    D --> E[MultiAgentTrajectory]
    E --> F[MathEnvironment.evaluate]
    F --> G[DoctorMASCreditAssigner]
    G --> H[TrainingSample]
    H --> I[VerlDataProtoAdapter]
    I --> J[VERL DataProto]
```

The DrMAS behavior in this first slice is intentionally narrow:

1. The orchestra runs a fixed Solver -> Verifier -> Solver-refine loop.
2. The environment computes a final outcome reward from the solver final answer.
3. `GlobalBroadcastCreditAssigner` writes the same episode reward to each trainable agent turn.
4. `DoctorMASCreditAssigner` computes GRPO-style advantages by `rollout_group + agent_name`, so Solver and Verifier do not share one reward baseline.
5. `VerlDataProtoAdapter` is the only boundary that imports `verl.protocol.DataProto`.

## Smoke Command

```bash
python3 examples/trajweave/doctor_mas_math/smoke.py --backend tiny-torch --device cpu
```

Use `--backend rule` for deterministic CPU-only tests. Use `--backend tiny-torch --device cuda:0` only when CUDA is available.

## Tiny Training Command

```bash
PYTHONPATH=. python3 examples/trajweave/doctor_mas_math/train_tiny.py \
  --steps 160 \
  --batch-tasks 16 \
  --rollouts-per-task 8 \
  --eval-interval 20 \
  --lr 0.05 \
  --entropy-coef 0.0 \
  --seed 7 \
  --task-mode random \
  --num-solver-candidates 3 \
  --max-turns 2 \
  --output-dir outputs/doctor_mas_tiny_train_random_solver_2turn_delta
```

This command runs a real torch policy update loop on top of the MAS layer:

1. `SolverVerifierOrchestra` collects Solver -> Verifier -> Solver trajectories.
2. `SolverVerifierMathEnvironment` scores the final answer.
3. `DoctorMASCreditAssigner` computes agent-wise GRPO-style advantages.
4. `TrainableTinyMathPolicyBackend.policy_loss` rebuilds logits from trajectory metadata and applies policy-gradient loss.
5. `torch.optim.Adam` updates the tiny solver policy.

The command writes `config.json`, `metrics.jsonl`, and `tiny_policy.pt` under the output directory. This is a small diagnostic policy for validating the MAS training path; it is not intended to replace the later VERL LLM trainer integration.

## DrMAS Search Slice

```mermaid
flowchart LR
    A[SearchTask] --> B[SearchAnswerOrchestra]
    B --> C[Verifier]
    C -->|SEARCH| D[Searcher]
    D --> E[SearchAnswerEnvironment.search]
    E --> C
    C -->|APPROVED| F[Answer]
    F --> G[MultiAgentTrajectory]
    G --> H[SearchAnswerEnvironment.evaluate]
    H --> I[DoctorMASCreditAssigner]
    I --> J[TrainingSample]
```

The Search slice is a fixed three-agent protocol:

1. Verifier checks whether enough evidence exists.
2. Searcher writes a query.
3. The environment executes the search tool and appends evidence to shared context.
4. Verifier approves once evidence exists.
5. Answer writes the final answer.
6. DrMAS credit assignment groups advantages by `rollout_group + agent_name`.

Run it through YAML:

```bash
PYTHONPATH=. python3 -m trajweave.cli.run \
  --config examples/trajweave/configs/doctor_mas_search_smoke.yaml
```

## YAML Launch Path

Every new recipe should have a YAML config under `examples/trajweave/configs/`. The intended user path is:

```text
YAML config
-> recipe builder
-> RolloutEngine
-> MultiAgentTrajectory
-> CreditAssigner
-> optional DataProto export
-> optional VERL trainer launch
-> optional VERL AgentLoopManager bridge
```

Current examples:

| Config | Purpose |
| --- | --- |
| `doctor_mas_math_smoke.yaml` | Deterministic Math rollout and DrMAS credit check. |
| `doctor_mas_search_smoke.yaml` | Deterministic Search rollout and DrMAS credit check. |
| `doctor_mas_math_tiny_train.yaml` | Tiny torch policy update loop. |
| `doctor_mas_math_verl_export.yaml` | Optional DataProto export and VERL trainer dry-run command generation. |
| `doctor_mas_math_verl_agent_loop_dryrun.yaml` | Optional DataProto export plus VERL V1 custom AgentLoopManager dry-run. |
| `doctor_mas_math_hf_gpu_smoke.yaml` | Local random Transformers model on CUDA for backend plumbing validation. |

The VERL path currently exposes three integration boundaries:

1. `VerlDataProtoAdapter` builds a VERL `DataProto` from TrajWeave `TrainingSample` objects.
2. `VerlTrainerLauncher` writes or runs a `verl.trainer.main_ppo` command from YAML overrides.
3. `TrajWeaveAgentLoopManager` can be loaded through `actor_rollout_ref.rollout.agent.agent_loop_manager_class`.

`TrajWeaveAgentLoopManager` is an importable bridge over VERL V1 `AgentLoopManagerTQ`. It validates TrajWeave runtime metadata from Hydra overrides, then delegates generation to VERL TransferQueue workers. Full GPU validation still needs the next adapter layer: a TrajWeave-native TransferQueue worker that turns `MultiAgentTrajectory` turns into the fields expected by VERL trainer sampling.
