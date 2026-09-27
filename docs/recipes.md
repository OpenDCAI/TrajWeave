# Recipe catalog

The maintained minimal training suite contains 25 configurations. Generate them with
`python -m trajweave.cli.prepare_smoke_suite --model /path/to/model --output-dir outputs/smoke-suite`.
Pass `--only` followed by one or more case names to select a subset.

GPU counts below describe **generated two-step configurations**, not every possible setting of the source template.
The generator changes mode, model paths, data, batch sizes, and training budgets; some templates are planning or synthetic presets on their own.
Check the generated `suite.json` before launching.

| Case                 | GPUs | Source template                                                                                          |
| -------------------- | ---- | -------------------------------------------------------------------------------------------------------- |
| `drmas_math`         | 1    | [math_qwen05b_2gpu.yaml](../configs/drmas/math_qwen05b_2gpu.yaml)                                        |
| `drmas_search`       | 1    | [search_qwen05b_2gpu.yaml](../configs/drmas/search_qwen05b_2gpu.yaml)                                    |
| `maporl`             | 2    | [debate_math_multi_actor_qwen05b_2gpu.yaml](../configs/maporl/debate_math_multi_actor_qwen05b_2gpu.yaml) |
| `agentflow`          | 1    | [flow_grpo_qwen05b_2gpu.yaml](../configs/agentflow/flow_grpo_qwen05b_2gpu.yaml)                          |
| `gigpo`              | 1    | [solver_verifier_math_qwen05b_2gpu.yaml](../configs/gigpo/solver_verifier_math_qwen05b_2gpu.yaml)        |
| `comas`              | 2    | [peer_review_math_qwen05b_2gpu.yaml](../configs/comas/peer_review_math_qwen05b_2gpu.yaml)                |
| `atgrpo`             | 1    | [solver_verifier_math_qwen05b_2gpu.yaml](../configs/atgrpo/solver_verifier_math_qwen05b_2gpu.yaml)       |
| `matpo`              | 1    | [browse_qwen05b_1gpu.yaml](../configs/matpo/browse_qwen05b_1gpu.yaml)                                    |
| `mrlx`               | 2    | [mgrpo_research_qa_2gpu.yaml](../configs/mrlx/mgrpo_research_qa_2gpu.yaml)                               |
| `wideseek_r1`        | 1    | [broad_search_qwen05b_1gpu.yaml](../configs/wideseek_r1/broad_search_qwen05b_1gpu.yaml)                  |
| `marshal`            | 1    | [tictactoe_selfplay_qwen05b_1gpu.yaml](../configs/marshal/tictactoe_selfplay_qwen05b_1gpu.yaml)          |
| `marft`              | 1    | [deepscaler_2agent_verl_tiny.yaml](../configs/marft/deepscaler_2agent_verl_tiny.yaml)                    |
| `c3`                 | 3    | [reasoner_actor_math_verl_tiny.yaml](../configs/c3/reasoner_actor_math_verl_tiny.yaml)                   |
| `marti_hf`           | 2    | [stage1e_multi_agent_hf_smoke.yaml](../configs/marti_mars2/stage1e_multi_agent_hf_smoke.yaml)            |
| `marti_vllm`         | 2    | [stage1e_multi_agent_vllm_smoke.yaml](../configs/marti_mars2/stage1e_multi_agent_vllm_smoke.yaml)        |
| `comlrl_magrpo`      | 2    | [magrpo_verl_tiny.yaml](../configs/comlrl/magrpo_verl_tiny.yaml)                                         |
| `comlrl_mareinforce` | 2    | [mareinforce_verl_tiny.yaml](../configs/comlrl/mareinforce_verl_tiny.yaml)                               |
| `comlrl_marloo`      | 2    | [marloo_verl_tiny.yaml](../configs/comlrl/marloo_verl_tiny.yaml)                                         |
| `comlrl_maremax`     | 2    | [maremax_verl_tiny.yaml](../configs/comlrl/maremax_verl_tiny.yaml)                                       |
| `comlrl_iac`         | 4    | [iac_verl_tiny.yaml](../configs/comlrl/iac_verl_tiny.yaml)                                               |
| `comlrl_maac`        | 3    | [maac_verl_tiny.yaml](../configs/comlrl/maac_verl_tiny.yaml)                                             |
| `comlrl_madpo`       | 2    | [madpo_verl_tiny.yaml](../configs/comlrl/madpo_verl_tiny.yaml)                                           |
| `comlrl_marlhf`      | 2    | [marlhf_verl_tiny.yaml](../configs/comlrl/marlhf_verl_tiny.yaml)                                         |
| `comlrl_madpo_iter`  | 2    | [madpo_iter_verl_tiny.yaml](../configs/comlrl/madpo_iter_verl_tiny.yaml)                                 |
| `comlrl_marlhf_iter` | 2    | [marlhf_iter_verl_tiny.yaml](../configs/comlrl/marlhf_iter_verl_tiny.yaml)                               |

All cases use four training examples and two validation examples. The math, search, browse, code, and game fixtures are integration inputs, not public benchmark evaluations. Search and browse use local document environments.

CoMLRL's ten entries cover policy-gradient, independent/centralized critic, preference, reward-model, and iterative training paths. MrlX's two steps exercise its delayed adapter update and pending-batch flush. MARFT exercises cooperative LoRA training. MARTI has separate HF and native vLLM cases.

Most cases use HF CPU generation with GPU optimization. Independent actors and critics explain the two-to-four-GPU layouts. Hardware memory requirements outside the tested H20 setup have not been characterized.

See [setup](getting-started.md), [measured validation and limits](validation.md), and [architecture](trajweave-mas-layer.md). Additional experimental presets under `configs/` are not automatically covered by these 25 results.

## MARTI entry points

MARTI-MARS² combines `CodeExecutionEnvironment`, `TreeSearchProtocol`, and the `trajweave_multi_actor_sync` trainer in its multi-actor path.

- [Protocol smoke](../configs/marti_mars2/single_mcts_smoke.yaml): lightweight code/tree-search fixture.
- [Tiny VERL plan](../configs/marti_mars2/single_mcts_verl_tiny.yaml): inspect the training bridge and generated assets.
- [Native vLLM multi-actor preset](../configs/marti_mars2/stage1e_multi_agent_vllm_smoke.yaml): source for the generated `marti_vllm` training case.

General cross-step asynchronous buffering is not enabled in the live multi-actor trainer. Review the [validation limits](validation.md) before changing concurrency settings.
