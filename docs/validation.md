# Validation record

This record describes the **2026-09-27 integration acceptance**, using the QF/LZ merge and subsequent fixes through commit `b0e1566`. The public summary was first recorded at `56b562d`. It is not a benchmark leaderboard or a fresh GPU run for the documentation refresh.

## Protocol

- Real local Qwen2.5-0.5B-Instruct weights; AgentFlow additionally used Qwen3-0.6B.
- NVIDIA H20 GPUs, one to four GPUs per generated case.
- Four training examples, two validation examples, and two recorded training steps per case.
- HF CPU generation plus GPU optimization for most cases; native vLLM GPU generation was also checked for MARTI.
- Process exit, optimizer steps, online trajectories, finite losses/gradients, validation output, and checkpoints checked together.

The acceptance result was **25/25 completed configurations; 24/25 with nonzero gradients**. These are pipeline checks using tiny fixtures, not evidence of convergence, task-level superiority, or original-paper reproduction.

## Per-case results

| Case                 | Steps | Train turns | Validation turns | Gradient signal |
| -------------------- | ----- | ----------- | ---------------- | --------------- |
| `drmas_math`         | 2     | 19          | 6                | nonzero         |
| `drmas_search`       | 2     | 37          | 10               | nonzero         |
| `maporl`             | 2     | 32          | 8                | nonzero         |
| `agentflow`          | 2     | 32          | 8                | zero            |
| `gigpo`              | 2     | 15          | 4                | nonzero         |
| `comas`              | 2     | 48          | 12               | nonzero         |
| `atgrpo`             | 2     | 32          | 4                | nonzero         |
| `matpo`              | 2     | 11          | 2                | nonzero         |
| `mrlx`               | 2     | 29          | 8                | nonzero         |
| `wideseek_r1`        | 2     | 40          | 10               | nonzero         |
| `marshal`            | 2     | 18          | 6                | nonzero         |
| `marft`              | 2     | 16          | 4                | nonzero         |
| `c3`                 | 2     | 24          | 4                | nonzero         |
| `marti_hf`           | 2     | 16          | 8                | nonzero         |
| `marti_vllm`         | 2     | 16          | 8                | nonzero         |
| `comlrl_magrpo`      | 2     | 32          | 4                | nonzero         |
| `comlrl_mareinforce` | 2     | 32          | 4                | nonzero         |
| `comlrl_marloo`      | 2     | 32          | 4                | nonzero         |
| `comlrl_maremax`     | 2     | 32          | 4                | nonzero         |
| `comlrl_iac`         | 2     | 16          | 6                | nonzero         |
| `comlrl_maac`        | 2     | 16          | 8                | nonzero         |
| `comlrl_madpo`       | 2     | 68          | 2                | nonzero         |
| `comlrl_marlhf`      | 2     | 64          | 4                | nonzero         |
| `comlrl_madpo_iter`  | 2     | 16          | 2                | nonzero         |
| `comlrl_marlhf_iter` | 2     | 48          | 4                | nonzero         |

Turn counts describe collected agent turns, not independent dataset examples or training steps.

## Additional checks

- MARTI native vLLM: **10/10 strict acceptance checks**, including PRIME scoring, actor routing, rollout weight synchronization, finite nonzero gradients, and rollout correction.
- Checkpoint audit: sampled q_proj or LoRA_B tensors from layers 0, 12, and 23 of saved actors/critics were finite and changed. This was not an exhaustive numerical audit of every tensor.
- MARLHF: reward-model parameter changes and preference pairs checked separately.
- Iterative MADPO and MARLHF: controller workflows completed and produced optimization metrics, validation, and final checkpoints.
- Joint CPU regression at `b0e1566`: **957 passed, 70 warnings**, 253.78 seconds; additional real PRIME subprocess checks: **16 passed**.

## Limits that matter

**AgentFlow:** Qwen3-0.6B produced valid planner formats, but all rewards in the tiny batch were 1. Advantages and gradients were zero. The run establishes workflow completion, not effective learning. Small parameter changes alone do not establish a learning signal.

**Resume:** support varies by trainer. MrlX does not persist its adapter replay state and rejects checkpoint resume. Multi-actor, independent-critic, and preference trainers have different constraints; test the exact trainer before a long run.

**Asynchrony:** general cross-step asynchronous buffering is not enabled in the live multi-actor trainer. Do not infer that every asynchronous preset has full training acceptance.

**External tools and scale:** local search/browse fixtures do not validate a production web search service. Two steps do not cover long-horizon stability, larger models, all hyperparameters, throughput, or benchmark quality.

## Evidence and reproduction

The maintainer retains raw logs, generated configurations, and checkpoints outside Git. This repository does not bundle those private storage paths or large artifacts. Evidence records include `results.json`, `final-acceptance.json`, `checkpoint-audit.json`, `reward-model-audit.json`, the CPU JUnit report, and the generated `suite.json`.

The [suite generator](../trajweave/cli/prepare_smoke_suite.py), [recipe catalog](recipes.md), and [setup guide](getting-started.md) provide a reproducible way to run new integration checks. New runs should produce their own evidence; the historical numbers above should not be reused as new experimental results.
