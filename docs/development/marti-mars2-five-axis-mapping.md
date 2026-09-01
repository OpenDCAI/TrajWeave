# MARTI-MARS2 五轴映射

本文档是 MARTI-MARS2 接入 TrajWeave 的 Phase 2 产物。目标是先把算法语义映射清楚，再决定哪些能力进入公共层、哪些留在 recipe、哪些属于后端扩展。

## 0. 固定源码

- 官方仓库：`https://github.com/TsinghuaC3I/MARTI`
- 固定提交：`a2fe2c7b9ec46cf24769c90575c51d847f41d04e`
- TrajWeave 接入仓库：`https://github.com/OpenDCAI/TrajWeave`
- 验证说明：贡献者本地验证副本和 run 目录未包含在本仓库，不作为可携带的验收产物。

## 1. 五轴表

| 轴 | MARTI-MARS2 语义 | 官方源码锚点 | TrajWeave 映射 |
| --- | --- | --- | --- |
| 控制权 | MCTS 算法决定 selection/expansion；workflow 控制节点上限、eval/train 分支和 checkpoint；LLM generation 作为 MCTS expansion 的动作生成器 | `MARTI-original/marti/agent_workflows/ab_mcts_workflow.py:90` 选择同步/异步算法；`:149` 计算 `max_num_nodes`；`:165` 执行 MCTS step | `TreeSearchProtocol`，负责 selection、expand、refine、stop；第一版 recipe 先覆盖 single-agent MCTS |
| 通信图 | single-agent 时是一棵 prompt-local search tree；multi-agent 时多个 agent 的 generation function 写入同一 tree，节点带 `agent_id/agent_role` | `ab_mcts_workflow.py:116` 为每个 agent 构造 `generate_fns`；`:218` 输出 node record 的 `agent_id/agent_role` | `TreeTrajectory` + `SearchNode`，显式记录 `tree_id/prompt_id/node_id/parent_idx/path/agent_id`；multi-agent 后续扩展为 shared-tree communication graph |
| 训练对象 | 每个 agent 配置可绑定不同模型、save path、ckpt path 和 tuning 标记；single-agent 是一个 trainable policy group；multi-agent 可扩展为多个 worker group | `examples/mars2/run_train_single_mcts.sh` 的单 agent 配置；`run_train_multi_mcts.sh` 的多 agent 配置；`multi_agent_train_ppo_ray.py` 解析 agents/worker resources | `AgentSpec`、`PolicyGroupSpec`、worker group metadata；第一版注册 `marti_mars2.single_mcts.smoke`，后续再接 multi-agent worker group |
| Credit | verifier 先给 node reward；processor 预留 parent/sibling shaping；trainer 使用 `group_norm`，按 `max_num_nodes` 固定 reshape 后计算组内 mean/std advantage | `ab_mcts_processor.py:10` parent/sibling shaping 读取 `metadata.parent_idx`；`:211` 收集 node reward；`experience_maker.py:705` 取 `max_num_nodes`；`:707` 固定 reshape；`:725` group_norm | `TreeGroupBuilder` 按显式 tree 分组；`group_normalized_advantages` 复刻两节点 group_norm 数值；输入乱序时 fail fast；IS/TIS 作为可选 loss/backend hook |
| 聚合规则 | 训练阶段用每个 node 作为样本；eval 阶段根据 valid nodes 计算 coverage/pass@k；最终答案来自 tree/path 中的候选选择 | `ab_mcts_workflow.py:198` 收集 valid nodes；`:201` eval 计算 coverage/pass@k；`:238` 返回 trajectory/final_reward | recipe metadata 标记 `best_path_or_mcts_eval`；后续新增 `TreePathCreditAllocator` 和 eval adapter 对齐 Pass@1、Pass@1(MCTS)、Pass@N |

## 2. 当前已落地的 TrajWeave 公共能力

| 能力 | 文件 | 说明 |
| --- | --- | --- |
| `SearchNode` | `trajweave/core/tree.py` | 单个 tree node 的身份、parent、agent、tokens、reward、logprob、path |
| `TreeTrajectory` | `trajweave/core/tree.py` | 同一 prompt/tree 下的节点集合，校验 node identity 和 parent link |
| `TreeGroupBuilder` | `trajweave/credit/tree_grouping.py` | 按显式 `tree_id` 分组，并能检查固定连续分组是否等价 |
| `group_normalized_advantages` | `trajweave/credit/tree_grouping.py` | CPU 纯函数；两节点 `[1, 0]` 得到约 `[+0.7071, -0.7071]`，对齐 MARTI `torch.std` 默认样本标准差口径 |
| `importance_correction_weights` | `trajweave/credit/tree_grouping.py` | CPU 纯函数；覆盖 token/sequence/geometric 与 truncate/mask，为 IS/TIS hook 保留公共数学逻辑 |
| `marti_mars2.single_mcts.smoke` | `trajweave/recipes/marti_mars2/` | 独立 recipe 的 CPU smoke 入口，输出 tree-group samples |

## 3. 官方代码性质与接入决策

当前不能把官方代码简单判定为“advantage 算错”。更准确的性质是：

- 官方最小运行路径中，同一 tree 的节点按顺序连续输出，因此固定 `max_num_nodes` reshape 在 smoke 场景里等价于显式 tree 分组。
- 官方实现没有用稳定 `tree_id/prompt_id` 作为训练分组边界；如果异步调度、动态过滤或数据重排破坏连续性，就可能跨 tree 归一化。
- 官方 workflow 输出只有 `expand_idx`，而 reward shaping 读取 `metadata.parent_idx`；因此 parent/sibling shaping 在原生 schema 下会退化为 no-op。
- 官方 shell 启用 IS correction，但当前固定依赖 `vllm==0.8.5.post1` 与代码中 `vLLM > 0.10.0` 的断言冲突；论文级 TIS 路径仍需隔离验真。

TrajWeave 第一版接入策略：

- 必须显式维护 `tree_id/prompt_id/node_id/parent_idx/path`。
- 必须用 `TreeGroupBuilder` 或等价逻辑按 tree 分组，不能只依赖样本排列。
- parent/sibling shaping 作为可测 credit 能力接入，但不假设官方最小路径已经启用。
- IS/TIS 先保留字段和 hook；隔离验真成功后再接入完整 correction。

## 4. Phase 2 完成标准

- 五轴均已绑定 MARTI 官方源码锚点和 TrajWeave 目标接口。
- 已识别公共层：tree schema、tree grouping、group advantage、IS/TIS 权重纯函数。
- 已识别 recipe 层：single-agent MCTS smoke、后续 multi-agent MCTS 配置。
- 已识别后端层：vLLM rollout logprob、policy logprob、IS/TIS loss hook。
- 已将当前 group audit 转为 CPU 单测：连续等价、乱序拒绝、advantage 数值、IS/TIS 纯函数。
