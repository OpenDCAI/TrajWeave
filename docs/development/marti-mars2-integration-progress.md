# MARTI-MARS2 集成开发归档

> 文档职责：记录 MARTI-MARS2 的原仓库验真、设计判断、真实实验与后续接入决策。算法集成路线图只维护状态和下一步，不承载完整实验日志。

> 归档说明（2026-09-28）：下文保留早期开发阶段的判断，不代表最新验收状态。2026-09-27 的 MARTI 原生 vLLM 两步训练已通过十项严格检查；完整口径和限制见 [Validation record](../validation.md)。

## 0. 早期开发状态快照

| 项目 | 状态 | 证据或下一步 |
| --- | --- | --- |
| 官方仓库 | 已固定 | `TsinghuaC3I/MARTI@a2fe2c7` |
| 论文归档 | 已定位 | arXiv `2602.07848`，MARTI-MARS2 technical report |
| 独立验证记录 | 历史记录 | 贡献者本地副本和 run 目录未随仓库提交，不作为当前可复验 artifact。 |
| 原仓库最小训练 | 已通过 | 当前 vLLM 0.8.5 下关闭 IS/TIS，完成 MCTS rollout -> reward -> group advantage -> policy update -> vLLM weight sync |
| grouping 语义验真 | 已通过最小场景 | 单 prompt 与双 prompt batch 均显示固定连续分组与显式 tree_id 分组一致 |
| vLLM IS/TIS | token-level 已通过 | native vLLM correction 已进入 TrajWeave one-step，finite correction 与 actor→vLLM sync 均有真实证据 |
| TrajWeave 实现 | 双 Actor native vLLM 同步闭环已接入 | 路由、权重版本、checkpoint/resume 和 endpoint 命名空间有 CPU 回归；待 GPU 环境恢复后重跑。 |
| 跨 step 异步 buffer | 暂不可用 | VERL 会在 step 末清理当前 TQ batch，原实现只缓存 metadata，可能丢失跨 step 样本；真实训练现已 fail fast。 |

当前阶段结论：MARTI-MARS2 的 tree schema、code verifier、tree-group credit、多 Actor 路由和 native vLLM endpoint 扩展已进入 TrajWeave。同步路径可继续验收；跨 step 异步 buffer 在完成 TQ 样本所有权和清理协议前不得宣称已支持。后文保留的 GPU 结果是贡献者历史开发记录，由于原始 run artifacts 未入库，不替代当前分支的可复验验收。

## 1. 原仓库验真

目标：确认官方 MARTI-MARS2 代码不是只打印日志，而是能跑通最小真实训练闭环，并定位与论文完整复现相关但不阻塞 TrajWeave 接入的差异。

### 1.1 环境与入口

验证环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime | 12.4 |
| vLLM | 0.8.5.post1 |
| Ray | 2.48.0 |

验证仓库：

```text
CONTRIBUTOR_LOCAL_ARTIFACT/MARTI-verify-patched
```

最小运行脚本：

```text
CONTRIBUTOR_LOCAL_ARTIFACT/marti-verification/run_stage1c1d_minimal.sh
```

关键运行约束：

- 使用 `Qwen2.5-Coder-0.5B-Instruct` 做工程 smoke。
- `max_num_nodes=2`，保证每个 tree 至少两个候选节点。
- 关闭 `enable_vllm_is_correction`，绕开当前 vLLM 0.8.5 与 `logprobs_mode` 的版本冲突。
- 打开 `MARTI_VERIFY_AUDIT_GROUPS=1`，只审计 grouping，不改变默认训练语义。

### 1.2 训练闭环证据

单 prompt smoke：

```text
CONTRIBUTOR_LOCAL_ARTIFACT/marti-verification/runs/group-audit-20260802-1604
```

双 prompt batch smoke：

```text
CONTRIBUTOR_LOCAL_ARTIFACT/marti-verification/runs/group-audit-2prompts-20260802-1605
```

已观察到：

- `Calculated max_steps: 2`，不再停在空训练步。
- MCTS workflow 生成两个节点。
- code verifier 返回 reward。
- reward allocation 执行，记录 `raw_reward` 与 `shaped_reward`。
- experience maker 计算 group advantage/return。
- policy actor 执行训练，出现 `Global step`。
- vLLM engine 接收权重同步，出现 `MARTI_VERIFY_POLICY_SYNC`。
- checkpoint 与 HF 权重产物落盘。

因此，MARTI-MARS2 最小训练闭环验真通过。

### 1.3 Grouping 语义

当前官方训练代码的默认行为仍是按固定组大小 reshape：

```text
group_size = workflow_args.max_num_nodes
rewards.reshape(-1, group_size)
```

这依赖 rollout 样本顺序。为了验证该顺序是否等价于 tree 分组，验证补丁额外传播 `prompt_id/tree_id/node_id/turn_id`，并在不改变训练语义的前提下对比固定连续分组与显式 `tree_id` 分组。

单 prompt 结果：

```text
MARTI_VERIFY_GROUP_AUDIT group_size=2 tree_sizes=[2] partition_matches=True
```

双 prompt batch 结果：

```text
MARTI_VERIFY_NODE prompt_id=0 node_id=0
MARTI_VERIFY_NODE prompt_id=0 node_id=1
MARTI_VERIFY_NODE prompt_id=1 node_id=0
MARTI_VERIFY_NODE prompt_id=1 node_id=1
MARTI_VERIFY_GROUP_AUDIT group_size=2 tree_sizes=[2, 2] partition_matches=True
```

判断：在当前官方最小运行路径中，固定连续分组与按 tree_id 分组等价。因此不应把现有实现直接判定为 advantage 代码错误。更准确的性质是：官方实现依赖“同一 tree 的节点连续排列”这个不变量；TrajWeave 接入时应把这个不变量显式化，并在 schema 和测试里保护。

## 2. vLLM IS/TIS 决策

论文的 MARS2+ 配方包含 GSPO、Overlong Penalty 和 TIS。TIS/IS 的作用是校正 vLLM rollout 分布与训练侧 policy 分布之间的 mismatch，属于长上下文代码 RL 的训练稳定化组件。

当前官方代码在启用 `logprobs_mode` 时要求：

```text
vLLM > 0.10.0
```

但本次可跑通环境为：

```text
vLLM 0.8.5.post1
```

决策：

- 不在当前阶段升级 vLLM。
- 不把 TIS/IS 作为 TrajWeave 接入 MARTI-MARS2 的前置条件。
- 在 TrajWeave 的 rollout/backend schema 中保留 rollout logprob 与 correction hook，使后续可以接入 TIS/IS。
- 若目标是复现论文最终成绩或验证 MARS2+ 完整训练配方，再单独建立 `vLLM > 0.10.0` 环境验证 IS/TIS。

影响判断：

- 对论文完整复现：有影响，不能声称完整 MARS2+ 配方已复现。
- 对 TrajWeave 最小接入：不阻塞。
- 对五轴映射、trajectory schema、tree search protocol、reward/credit adapter：不构成前置依赖。

## 3. 对 TrajWeave 的接入结论

MARTI-MARS2 可以进入第二阶段“五轴映射”。下一步不需要继续围绕 vLLM/TIS 或原仓库大改打转，而应把 MARTI-MARS2 映射成 TrajWeave 的公共能力需求。

初步接入重点：

- 显式 `TreeTrajectory` / `SearchNode` schema。
- `tree_id/prompt_id/node_id/parent_idx/agent_id` 等身份字段。
- MCTS expansion/refinement 的控制权模型。
- 多 agent tree search 的角色、策略绑定和调度关系。
- code sandbox verifier 与 reward shaping 的边界。
- tree-level/group-level credit 与 path-level credit 的表示。
- 按 tree 分组的 advantage builder，避免仅依赖样本排列。

## 4. 下一阶段 Gate

第二阶段“五轴映射”需要输出：

- 控制权：谁决定 expand、refine、stop、select best path。
- 通信图：多 agent 在 tree search 中如何共享节点、答案和反馈。
- 训练对象：单 agent、同构多 agent、异构多 agent分别训练哪些 policy。
- Credit：node reward、tree reward、path reward、agent reward 如何进入 advantage。
- 聚合规则：最终答案、best path、MCTS eval、Pass@N 如何从 tree 中产生。

完成五轴映射后，再决定哪些能力放入 TrajWeave core、orchestration、reward、credit、backend extension 和 recipe。

## 5. 待办

- [ ] 完成 MARTI-MARS2 五轴映射。
- [ ] 设计 TrajWeave `TreeTrajectory` / `SearchNode` 最小 schema。
- [ ] 定义 MARTI-MARS2 recipe 的最小配置入口。
- [ ] 把 grouping audit 结论转成 CPU 单测：连续 tree 分组等价、交错 tree 分组报警或拒绝。
- [ ] 延后建立 `vLLM > 0.10.0` 环境验证 IS/TIS。

## 6. 2026-08-04 当前判断与执行计划

本节覆盖前述旧路线图之后的实际状态。当前目标不是要求 TrajWeave 在本机复现论文最终分数，而是先完成机制上可审计的 MARTI-MARS² fidelity 单 Agent 垂直闭环，再把稳定训练和增强能力逐项接入。

### 6.1 两条 recipe 必须分开

`tree_id`、`prompt_id`、`node_id`、`parent_idx` 和 `path` 是身份与结构字段，可以保留在 fidelity 模式中；它们不改变 MCTS、reward 或 GRPO 数学。

parent/sibling/path shaping 不是当前已确认的 MARTI 官方核心路径。当前 TrajWeave hook 若无条件执行 shaping，会把 fidelity 与实验增强混在一起，因此需要拆成：

- `marti_mars2_fidelity`：真实 verifier node reward + 显式 tree-group GRPO；默认关闭 parent/sibling/path shaping。
- `marti_mars2_tree_credit_experimental`：显式开启 parent/sibling/path credit，单独记录 shaped reward/path return，不标记为官方忠实实现。

### 6.2 当前未完成项的边界

| 能力 | 当前证据 | 还不能声称 |
| --- | --- | --- |
| Tree schema / tree grouping | CPU 测试、DataProto/TransferQueue 传输 | 已经在所有动态重排路径中完成显式 runtime grouping |
| TreeSearchProtocol | CPU 上完成 selection、expansion、refinement、backpropagation、stop | 已驱动真实 VERL/TQ rollout |
| Tree GRPO | TrajWeave HF tiny one-step 有非零梯度 | 已完成官方 fidelity recipe |
| VerifierAdapter | 接口和 CPU adapter 已有 | 真实 code verifier 已进入 TrajWeave emitter 的 reward/refinement loop |
| TIS/IS | MARTI 验证副本的 vLLM 0.10.2 token truncated correction 已跑通 | TrajWeave 内真实 vLLM correction optimizer step |
| GSPO | 尚无独立实现和验收证据 | GSPO 已接入 |
| Overlong | VERL 通用能力可用 | MARTI recipe 已验证 |
| 失败记录 / 深层引导 | 基础状态字段和 CPU feedback 路径 | 真实 verifier failure 驱动深层 refinement |
| Reward Model 选答案 | 当前是规则式 best reward/best node | 正式 Reward Model selector |
| Multi-Agent | 尚未完成 | Homo/Heter-MARS² |

### 6.3 推荐执行顺序

1. 先拆分 fidelity 与 experimental recipe；fidelity 默认不做 sibling/path shaping，并补对应 CPU 回归测试。
2. 将真实 `VerifierAdapter` 接入 TrajWeave VERL/TQ emitter，记录 verifier success、feedback、failure type 和测试结果。
3. 让真实 rollout worker 使用持久化 `TreeSearchProtocol` 控制 selection、refinement 和 stop，替换当前 deterministic two-root + refinement-child emitter。
4. 运行一次 `0.5B + vLLM + n>=2 + code verifier + correction` 的单 Agent 联合训练，验收有限 correction、非零 advantage/gradient、权重同步和 checkpoint tensor/hash 变化。
5. 在相同模型、数据和预算下完成 Vanilla GRPO、MARTI fidelity 和 tree-credit experimental 三方最小对照。
6. 再逐项验证 token/sequence TIS、GSPO、Overlong Penalty；不要把 vLLM correction smoke 直接称为 GSPO。
7. 再加入失败记录、深层引导和 Reward Model selector；最终才进入双 Agent Stage 1E。

### 6.4 第一阶段完成标准

在声称“TrajWeave 已接入 MARTI-MARS² 核心机制”前，必须同时满足：

- fidelity recipe 默认关闭 parent/sibling/path shaping；实验增强可显式 opt-in。
- 真实 verifier 产生 node reward，并将 feedback 传入后续 refinement。
- selection、refinement、stop 由 `TreeSearchProtocol` 或等价持久 controller 驱动。
- tree identity 在训练 grouping 中有显式契约；节点重排或过滤不会跨 tree 归一化。
- 至少一次真实单 Agent VERL optimizer step 具备有限 reward/advantage/loss、非零梯度、权重同步和 checkpoint 变化证据。
- Vanilla GRPO 对照已建立；未把论文最终 Pass@1 或多 Agent 结果作为当前阶段要求。

当前直接执行项：先修改 recipe/hook 的 credit mode，随后补 fidelity 与 experimental 的 CPU 测试；完成后再启动真实 verifier/dynamic TreeSearch 联合 run。

### 6.5 2026-08-04 首项实施结果

- 已新增 `TreeGroupCreditAllocator`，fidelity 使用 raw verifier node reward + tree-group GRPO。
- `MARTIMARS2TreeGRPOHooks` 新增 `credit_mode`：默认 `fidelity`，只有 `experimental` 才执行 parent/sibling/path shaping。
- `credit.mode` 已进入 MARTI recipe launch overrides；现有 tiny/vLLM dry-run 配置明确标记为 `fidelity`。
- 新增 `configs/marti_mars2/single_mcts_tree_credit_experimental.yaml` 作为显式增强 smoke 配置。
- 新增 recipe aliases：`marti_mars2_fidelity` 与 `marti_mars2_tree_credit_experimental`。
- 已完成 `compileall`、`git diff --check`，并用 MARTI 验证环境直接运行 fidelity/experimental smoke：两种模式均生成 6 个样本，fidelity credit 为 `marti_mars2_fidelity_group_grpo`，experimental credit 为 `marti_mars2_tree_path_grpo`。
- 在独立 MARTI 验证环境补齐 `pytest`、`omegaconf`、`tensordict`、`codetiming` 后，联合定向测试通过：hooks、registry、tree grouping、tree protocol、recipe 共 `28 passed`；仅有既有 Ray/PyNVML/NPU 兼容性 warning。

## 7. 2026-08-04 fidelity vLLM 联合训练验收

### 7.1 配置与环境

- 配置：`configs/marti_mars2/single_mcts_verl_vllm_fidelity_smoke.yaml`
- 运行环境：`CONTRIBUTOR_LOCAL_ENV/pettingllms-atgrpo/bin/python`
- 训练模型：`CONTRIBUTOR_LOCAL_MODEL/marti-mars2/Qwen2.5-Coder-0.5B-Instruct`
- vLLM：0.10.0；Torch：2.7.1+cu128；单卡 GPU 0；`gpu_memory_utilization=0.20`
- 配方：single-agent、`max_num_nodes=2`、`credit_mode=fidelity`、token-level truncated correction、1 个 optimizer step
- 为兼容当前 TrajWeave bridge，在隔离测试环境使用 `tensordict>=0.10`；旧版本 0.8.3 会在传输 non-tensor tree fields 时被仓库断言拒绝。

### 7.2 结果

运行目录：

```text
outputs/trajweave/runs/20260804-090651-marti-mars2-single-mcts-fidelity-14ee3e59
```

状态为 `completed`，VERL return code 为 0。关键指标：

```text
training/global_step=1
actor/loss=0.0353894457
actor/grad_norm=10.9313669
critic/advantages/max=0.7071057558
critic/advantages/min=-0.7071057558
trajweave/marti_mars2/raw_score/mean=0.5
trajweave/marti_mars2/parent_sibling_reward/mean=0.5
trajweave/marti_mars2/path_return/mean=0.5
rollout_corr/rollout_is_mean=0.0745918006
rollout_corr/rollout_is_min=2.223804e-07
rollout_corr/rollout_is_max=0.3734849989
rollout_corr/rollout_is_eff_sample_size=0.2304624
```

在线 trajectory 记录了同一 tree 的两个节点：

```text
node 0: parent=-1, path=[0], reward=1.0
node 1: parent=-1, path=[1], reward=0.0
```

checkpoint：

- `outputs/marti_mars2_vllm_fidelity/checkpoints/global_step_1/actor/model_world_size_1_rank_0.pt`
- 初始模型 SHA256：`f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44`
- checkpoint SHA256：`d709f752c94751e81f8df00ab63b64d1d5575450708b8c708e00dd2b4d28d141`
- 共同的 290 个 model tensor 中 290 个发生变化，最大绝对差 `1.9073486328125e-06`。

### 7.3 结论边界

本次已经证明：

- fidelity credit mode 在真实 VERL/VLLM 训练路径生效；
- sibling/path shaping 没有进入训练 reward（raw、parent_sibling、path_return 三者均为 0.5 均值）；
- tree metadata 穿过 AgentLoop、TransferQueue、DataProto 和 advantage 阶段；
- token-level correction、非零梯度、optimizer step、checkpoint 变化均真实发生。

本次仍不能证明：

- reward 来自真实 code sandbox verifier；
- verifier failure feedback 驱动了下一节点；
- selection/refinement/stop 由真实持久化 `TreeSearchProtocol` 控制；
- 已完成生产级 MCTS 或论文最终 recipe。

（本节记录 controller/verifier 接入前的 vLLM fidelity smoke 边界；后续 §8 已完成 worker-side controller 和 verifier adapter。）

## 8. 2026-08-04 verifier adapter 与动态 refinement 验收

### 8.1 已完成的实现

- 新增 `CodeVerifierAdapter`，优先调用 VERL PRIME `compute_score(..., continuous=True)`；当运行环境缺少 VERL/PRIME 的可选 `pyext` 扩展时，对小型 call-based test cases 使用明确标记的 local subprocess fallback。
- verifier 结果会写入 trajectory metadata：`verifier_success`、`verifier_terminal`、`verifier_feedback`、`failure_type`、`verification_mode` 和 `verifier_metadata`。
- emitter 会根据已有节点 reward 做 worker-side parent selection；refinement prompt 携带上一候选和 verifier feedback；AgentLoop 在 terminal success 时停止当前 session。
- tiny code assets 已包含 PRIME 风格 `fn_name`、`inputs`、`outputs`，因此不再只依赖文本 ground-truth fallback。

### 8.2 CPU 与真实 HF 证据

- 定向 tree protocol/verifier 测试：`9 passed`；组合 MARTI/VERL 定向回归：`29 passed`。
- 正确/错误 `add_one` 候选的 local fallback smoke：分别得到 score `1.0` 和 `0.0`，错误候选标记为 `wrong_answer`。
- 运行目录：

  `outputs/trajweave/runs/20260804-092534-marti-mars2-single-mcts-fidelity-7fb31986`

  配置为 0.5B Qwen2.5-Coder、HF local generation、`max_num_nodes=3`、2 个初始候选、fidelity credit，VERL return code 为 `0`。在线轨迹包含：

  - node 0/1：`parent_idx=-1`，`path=[0]`/`[1]`，阶段 `expand`；
  - node 2：`parent_idx=0`，`path=[0,2]`，阶段 `refine`；
  - 每个节点都有 verifier feedback，模式为 `local_subprocess_fallback`，fallback 原因明确记录为缺少 `pyext`。

本次模型生成的是不完整/不可执行代码，因此所有 node reward 为 `0.0`，`actor/loss=0`、`actor/grad_norm=0`。这不是训练链路失败，而是当前未训练模型没有产生可通过候选；它也说明不能把 verifier error 或模型失败伪装成非零学习信号。

### 8.3 证据边界与下一步

当前已证明真实 HF generation → verifier → feedback → dynamic refinement topology → 持久化 worker-side controller → tree metadata → VERL step 可以完整执行；尚未证明真实模型能产生通过测试的候选，也尚未完成 vLLM 实际模型 rollout 与 verifier 的联合 correction run。

下一步优先级：

1. 保留 local fallback 作为依赖受限 smoke，但在具备 `pyext` 的环境重复 PRIME sandbox 验收。
2. 为 verifier integration 增加一个已知正确候选的独立 rollout fixture，验证 `success → terminal/stop`，不把该 fixture 当作模型训练效果。
3. 将 `TreeSearchController` 接入 vLLM 实际模型生成路径，并在同一运行中启用 correction hook。
4. 在得到非零 verifier reward 后，重复 fidelity 的 correction、advantage、非零梯度和 checkpoint 变化验收；然后才做 Vanilla GRPO 对照及 GSPO/TIS/Overlong 分项接入。

### 8.4 worker-side controller 回归

- 新增 `TreeSearchController`，按 tree 持久化 `records`、`visits`、`value_sums`，提供 `select_parent`、`path_for`、backpropagation 和 `should_stop`。
- emitter 已改为使用该 controller；AgentLoop session 结束时同时清理 records 和 controller，避免跨样本串树。
- 回归运行目录：

  `outputs/trajweave/runs/20260804-093111-marti-mars2-single-mcts-fidelity-d3e02596`

  HF/VERL return code 为 `0`。轨迹仍保持 node 0/1 为两个 root、node 2 为 `parent_idx=0` 的 refinement，且 verifier failure 分类为 `compile_error` / `wrong_answer`，不再误报 `verifier_error`。由于模型候选仍全部失败，reward、advantage 和梯度仍为零；该运行只作为 controller/metadata/feedback 回归，不作为训练效果证明。

### 8.5 已知正确候选 stop fixture

使用同一 emitter 的独立 fixture，输入正确候选：

```python
def add_one(x):
    return x + 1
```

结果为 `score=1.0`、`success=True`、`terminal=True`。这验证了 `success → terminal/stop` 语义；该 fixture 是 verifier/controller 集成测试，不代表模型已经学会生成正确答案。由于当前环境没有 PRIME 的 `pyext` 扩展，fixture 使用的是已显式标记的 local subprocess fallback。

## 9. 2026-08-04 原生 vLLM AgentLoop 接入

### 9.1 接入内容

- 新增 `vllm_marti_tq` backend。vLLM 生成由 VERL 原生 `AgentLoop`/`LLMServerClient` 完成，TrajWeave worker 只负责串行 session、parent/refinement prompt、verifier、controller 和 TQ 写入。
- native TQ writer 现在同时保留 nested `extra_fields` 并把 tree/MAS fields 提升到顶层，保证 grouping/credit hook 可见。
- 对 native `_InternalAgentLoopOutput` 做了 Tensor、batch 维和 rollout logprob 维度归一化。
- vLLM smoke 环境显式设置 `VLLM_USE_V1=1`；GPU cache 配额提高到 `0.35`，避免 Qwen2.5-Coder-0.5B 在 FSDP 共卡时没有 KV cache。

### 9.2 真实运行证据

运行目录：

`outputs/trajweave/runs/20260804-113358-marti-mars2-single-mcts-fidelity-85d82593`

配置为原生 vLLM、`vllm_marti_tq`、`max_num_nodes=3`、2 个初始 root、token-level correction。状态为 `completed`，VERL launch return code 为 `0`。在线 trajectory 证明：

- node 0：`parent_idx=-1`、`path=[0]`、`search_stage=expand`；
- node 1：`parent_idx=-1`、`path=[1]`、`search_stage=expand`；
- node 2：`parent_idx=0`、`path=[0,2]`、`search_stage=refine`、`search_stop=true`。

训练日志同时出现原生 vLLM correction 指标，例如 `rollout_is_mean≈0.9726`、`rollout_is_max=2.0`、`training/global_step=1`，并完成 optimizer/checkpoint 路径。当前模型生成候选仍全部未通过代码测试，因此 raw reward、advantage、loss 和 gradient 为 `0`；这次运行证明的是原生 vLLM → controller/verifier → tree metadata → correction/TQ → VERL step 的链路，而不是模型训练效果。

### 9.3 当前边界

- 原生 vLLM rollout 已不再是 synthetic emitter；真实模型请求和 logprob 已进入 MARTI worker。
- PRIME `pyext` 缺失时仍使用 local fallback，因此尚未完成具备 `pyext` 的官方 PRIME sandbox 验收。
- 该 2026-08-04 run 尚未获得非零 verifier reward；后续第 10 节已使用受控任务完成非零 advantage、梯度和 checkpoint tensor 变化验收。

## 10. 2026-08-06 native vLLM fidelity 最终验收

最终成功 run：

`outputs/trajweave/runs/20260806-084939-marti-mars2-single-mcts-fidelity-03efa275`

状态和核心指标：

- 顶层 `status=completed`，VERL return code `0`，自动 fidelity acceptance `passed`。
- `zigzag_code` 真实 verifier reward 为 `[4/9, 0, 4/9]`，同一 tree 内存在真实方差。
- advantage max/min 为 `0.5773479939 / -1.1546959877`。
- actor loss 为 `0.0861684382`，actor grad norm 为 `4.9206008911`。
- token correction mean/min/max 为 `0.9963454008 / 0.5721966624 / 1.3964718580`。
- checkpoint 在本 run 独立目录保存，actor→vLLM 权重同步完成。

checkpoint tensor diff：

- 原始 SHA256：`f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44`。
- checkpoint SHA256：`4ec9999bb3d54bb05b59f0bf6aa99790b4f360a6f9563580fb8b340148b7e838`。
- 290 个共享 tensor 中 290 个发生变化；变化元素 `455,201,190 / 494,032,768`，最大绝对差
  `1.9073486328125e-06`；无 shape mismatch。
- 结构化证据：本 run 的 `checkpoint_tensor_diff.json`。

为恢复和固定该验收，本轮还补齐了训练环境中的 `TransferQueue==0.1.8`，并使当前 VERL
`RolloutConfig` 显式接受 vLLM `seed`。定向回归 `55 passed`，底层协议回归 `39 passed`，Ruff、compileall
和 `git diff --check` 全部通过。

当前准确边界：单 Agent native vLLM fidelity 工程闭环已经完成；PRIME `pyext` sandbox、Vanilla GRPO
对照、sequence correction/GSPO/TIS/Overlong、多 Agent独立策略与异步 buffer、DeepCoder/LiveCodeBench 和
论文规模指标仍未完成。

## 11. 2026-08-06 Vanilla GRPO 对照完成

- Run：`outputs/trajweave/runs/20260806-115024-marti-mars2-vanilla-grpo-baseline-994ba6c7`
- 状态：`completed`，VERL return code `0`，Vanilla acceptance `passed`。
- verifier rewards：`[0.4444444444, 0.0, 0.0]`；advantage max/min：`1.1546959877 / -0.5773479939`。
- actor loss：`0.0861713290`；grad norm：`7.6755609512`；response length：`96`；Global step：`1`。
- correction 关闭（Vanilla acceptance 不要求 correction）。
- checkpoint SHA256：`cd0036725edbff0bd8ea9d900ef6d56374d4e68cf9e9312cc31ef294b6539fa9`；
  共享 tensor `290/290` 变化，变化元素比例 `90.4331%`，最大绝对差 `1.9073486328125e-06`。
- 轨迹拓扑：三个 independent roots，均为 `expand`、`parent_idx=-1`；MARTI 对照则是两个 roots 加一个
  `refine` child。完整表格见该 run 的 `marti_vs_vanilla_comparison.md`。

该 run 证明 Vanilla GRPO 与 MARTI fidelity 共用真实 verifier、group advantage、optimizer、checkpoint 和
权重同步入口；它是单步工程对照，不是论文规模收敛或 benchmark 结果。下一步进入 Stage 1E 双 Agent contract
与最小真实 run 设计。

## 12. 2026-08-06 Stage 1E 双 Agent contract 开始

新增 `configs/marti_mars2/stage1e_multi_agent_contract.yaml` 作为不执行训练的 launch contract。MARTI recipe
现在校验并传播 `agent_ids/model_ids/model_sharing`；tree metadata 使用确定性 round-robin 绑定，并记录
`agent_id`、`policy_group`、`worker_group`、`agent_turn_index`。`multi_actor_training=true` 时复用
`trajweave_maporl_multi_actor_sync`，按 worker group 路由并分别保存 actor checkpoint；相关 CPU contract 测试已通过。

这不是双 Agent 真实训练完成：per-agent replay/buffer、异步更新和 multi-actor vLLM 权重同步仍待实现；当前
multi-actor 路径限制为 `hf_local_tq` 或 `synthetic_tq`，需要先完成最小 HF-local run 验收。

HF-local routing run 已执行：
`outputs/trajweave/runs/20260806-122709-marti-mars2-single-mcts-fidelity-ace663f2`。两个 actor group 都进入
update/save，routing acceptance 通过；但四个候选 reward 全为 0，两个 actor 的 `0/290` tensor 变化也说明这
不是学习信号验收。下一步是构造保持真实 verifier 语义的非零 reward fixture，再验证 per-agent buffer 与异步更新。

## 13. 2026-08-06 Stage 1E 受控 verifier-positive fixture 完成

fixture 配置：`configs/marti_mars2/stage1e_multi_agent_fixture.yaml`。首次运行暴露 Hydra/JSON CLI
override 会把候选代码中的换行保留为字面量 `\\n`，导致 verifier 报 `SyntaxError`；emitter 现已在 fixture
边界解码该传输表示，并增加 CPU 回归测试。该修复不改变 native vLLM fidelity 路径。

正式 run：`outputs/trajweave/runs/20260806-131413-marti-mars2-single-mcts-fidelity-1f886062`，状态
`completed`，VERL return code `0`，自动 acceptance `passed`。真实 `CodeVerifierAdapter` 产生四节点
reward `[1.0, 0.0, 1.0, 0.0]`；trajectory 保留 `generator -> policy_a`、`critic -> policy_b` routing
和 MCTS path metadata。两个 actor 都完成真实 optimizer update/save：policy_a loss `-0.8660238`、
grad norm `9.27737`；policy_b loss `0.8660240`、grad norm `56.76477`；advantage extrema 为
`[-0.866024, 0.866024]`，actor groups updated `2/2`。两个 actor 各自 290/290 shared tensors
changed，逐 actor SHA256 与 diff 见该 run 的 `multi_actor_tensor_diff.json`，详细验收见
`stage1e_fixture_report.md`。

相关 MARTI/acceptance/routing/bridge 回归 `68 passed, 2 warnings`；`compileall -q trajweave verl`、
`git diff --check` 和 tensor-diff JSON 解析均通过。

这是 Stage 1E routing + verifier + multi-actor optimizer 的受控单步闭环证据，不是论文规模收敛或论文
多 Agent 复现。multi-actor vLLM 权重同步仍未实现；下一阶段应设计多 actor 权重版本/同步 contract，再扩展
真实 vLLM 路径。

## 14. 2026-08-06 Multi-actor weight-sync contract 固化

新增 `trajweave/backends/verl/weight_sync.py`，将 actor 更新版本与 rollout engine 已同步版本分离记录，
并接入 `trajweave_maporl_multi_actor_sync` trainer。每个 `global_step` checkpoint 现在生成
`multi_actor_weight_sync.json`，包含 group、checkpoint SHA256、actor version、rollout-synced version 和
pending groups。

验证 run：`outputs/trajweave/runs/20260806-132719-marti-mars2-single-mcts-fidelity-ff5325bd`。
真实 verifier/optimizer acceptance 通过；manifest 显示 `actor_versions=2`、`rollout_synced=0`、
`pending=[policy_a, policy_b]`、`synchronized=false`。这明确记录了当前 multi-actor vLLM transport 尚未
实现，而不是把 actor checkpoint 保存误报成 vLLM 已同步。CPU contract/routing 测试通过，详细说明见该 run
的 `stage1e_weight_sync_contract_report.md`。

下一步是实现一个真正的 per-policy-group rollout weight transport，并在同步确认后调用
`mark_rollout_sync`；在此之前仍保持 multi-actor `vllm_marti_tq` launch guard。

本轮已先完成 transport adapter contract：`MultiActorWeightSyncContract.sync_pending(transport)` 会逐组
调用 `load_policy(version)`，只有 transport 返回 engine acknowledgement 时才推进
`rollout_synced_step`；异常和未确认结果继续保留为 pending。CPU transport contract 测试已通过，尚未接入真实
vLLM server。

同时新增 `GroupedPolicyWeightTransport`，要求每个 `policy_group` 显式绑定 endpoint，未知 group 直接失败，
禁止静默回退到其他 server。

multi-actor checkpoint resume 也已补齐 `auto/resume_path` contract：逐组加载
`global_step_*/actors/<policy_group>`，恢复 dataloader 与 weight-sync actor version；缺少任一 trainable group
时 fail fast。CPU resume 测试已通过，相关 MARTI/VERL 回归现为 `71 passed, 2 warnings`。真实 GPU resume
smoke 尚待独立配置执行。

## 15. 2026-08-06 multi-actor checkpoint resume GPU smoke 完成

真实 resume run：`outputs/trajweave/runs/20260806-140608-marti-mars2-single-mcts-fidelity-dd096d49`。
运行从上一轮 `global_step_1` 按 policy group 分别恢复 `policy_a`/`policy_b` 的 model、optimizer、scheduler/RNG、dataloader 和 weight-sync state，继续训练并保存到 `global_step_2/actors/`。四个 verifier-positive fixture 样本的 reward 为 `[1.0, 0.0, 1.0, 0.0]`；两个 actor 均再次完成非零 loss/gradient 更新，acceptance 为 `passed`。

该 run 的 `stage1e_resume_report.md` 和 `checkpoints/global_step_2/multi_actor_weight_sync.json` 是结构化证据。后者明确显示两个 actor version 已到 `2`，但 `rollout_synced_step` 均为 `null`、pending groups 为两个 policy。因此 Stage 1E 的真实 HF/synthetic multi-actor resume 已通过，但 multi-actor vLLM endpoint lifecycle、实际权重加载 acknowledgement 和 per-agent async buffer 仍未完成，不能把该结果描述为 multi-actor vLLM 已支持。

本轮新增 native vLLM 多 actor lifecycle 接入：按 policy group 创建独立 `LLMServerManager`、`CheckpointEngineManager`，并通过显式 group client/weight transport 路由生成与权重同步。新增 `configs/marti_mars2/stage1e_multi_agent_vllm_dryrun.yaml` 作为只生成 Hydra/VERL 命令的配置；它不启动 server，不作为真实 vLLM 训练证据。

## 16. 2026-08-06 checkpoint-engine endpoint adapter contract

新增 `CheckpointEnginePolicyEndpoint`，将一个 VERL `CheckpointEngineManager` 绑定为单一 policy group 的 `PolicyWeightTransport` endpoint。调用会携带该 group 的 actor `global_step`；VERL 当前 API 在成功时返回 `None`，因此 adapter 将“调用完成且未抛异常”记录为 acknowledgement；异常仍由 `MultiActorWeightSyncContract` 保留为 pending。`GroupedPolicyWeightTransport` 继续要求所有 group 显式绑定，不允许未知 group fallback。

CPU adapter/transport 回归和完整 MARTI/VERL 相关回归均通过（`105 passed, 2 warnings`）。代码已具备按 policy group 创建 vLLM server/replica、生成 client 和 checkpoint-engine endpoint 的 lifecycle，但尚未在真实 GPU run 验证多 server 启动与权重 acknowledgement；下一步是启动受控 vLLM smoke。

本轮只读显存检查显示 GPU 0-5 均已有约 21-24 GiB 占用，当前安全卡剩余不足以启动双 actor/双 vLLM replica，因此真实 GPU smoke 暂缓，未触碰其他任务进程。

## 17. 2026-08-09 Stage 1E 双 actor native vLLM GPU 验收完成

成功配置：`configs/marti_mars2/stage1e_multi_agent_vllm_smoke.yaml`。

成功 run：

`/tmp/trajweave-marti-stage1e/runs/20260809-080547-marti-mars2-stage1e-multi-agent-vllm-smoke-faacfe43`

本次使用 GPU 0、3，分别为 `policy_a`、`policy_b` 创建独立 server namespace、generation client、
checkpoint-engine endpoint 和 ZMQ weight-transfer handle。两组 vLLM server 使用 seed 42/43；训练前先从 live
actor 执行 step 0 初始同步，避免 `load_format=dummy` 权重参与首次生成。训练后两个 endpoint 均返回真实
acknowledgement：

```text
Initial multi-actor vLLM weight sync completed: [('policy_a', 0), ('policy_b', 0)]
Multi-actor vLLM weight sync step=1 results=[('policy_a', True, None), ('policy_b', True, None)]
```

自动 acceptance 为 `passed`，并同时通过真实 verifier、tree 内 mixed reward、非零 advantage、双 actor
非零 loss/gradient、token correction、checkpoint、multi-agent routing 和 multi-actor weight sync。关键结果：

```text
verifier rewards = [0.4444444444, 0.0, 0.0, 0.1111111111]
advantage min/max = -0.6603351 / 1.4527371
policy_a loss / grad_norm = -0.2739110 / 14.4270535
policy_b loss / grad_norm = 0.3963773 / 4.5749059
rollout_is min/mean/max = 0.5988577 / 1.0009208 / 2.0
```

`global_step_1/multi_actor_weight_sync.json` 最终显示：

```text
synchronized = true
pending_groups = []
policy_a rollout_synced_step = 1
policy_b rollout_synced_step = 1
step metric rollout_synced / pending / synchronized = 2 / 0 / 1
```

轨迹中 `generator -> policy_a` 与 `critic -> policy_b` 各出现两个 turn。checkpoint tensor diff 进一步确认两个
actor 都相对同一初始模型发生实际更新，且彼此分叉：

| 对比 | 变化元素 | 变化比例 | L2 | 最大绝对差 |
| --- | ---: | ---: | ---: | ---: |
| policy_a vs source | 565,741,772 / 630,167,424 | 89.7764% | 0.0189966 | 1.90735e-6 |
| policy_b vs source | 600,762,758 / 630,167,424 | 95.3338% | 0.0190705 | 1.90735e-6 |
| policy_a vs policy_b | 551,244,935 / 630,167,424 | 87.4759% | 0.0269332 | 3.81470e-6 |

针对 multi-actor sync、grouped routing、VERL routing、rollout config 和 ZMQ namespace 的定向回归为
`39 passed, 4 warnings`；`compileall -q trajweave verl` 与 `git diff --check` 通过。run 结束后 Ray/vLLM 进程已退出，
GPU 0、3 显存恢复空闲。

当前准确边界：Stage 1E 的同步双 actor、双 native vLLM endpoint 单步工程闭环已完成。下一阶段优先实现
per-agent buffer/采样所有权与异步更新语义，再做多步更新/恢复/版本滞后压力测试；该结果仍不是论文规模收敛、
异构模型训练或完整 MARS² benchmark 复现。

## 18. 2026-08-09 三部分工程接入补齐

- 新增 `PolicyBufferCoordinator`：按 policy group 独立维护 queue、actor/rollout step、readiness 和
  `policy_lag`；完整 tree reward 平均值采用严格区间 dynamic filtering，stale 样本显式丢弃并统计，resume
  只恢复版本元数据、未消费内存队列明确丢弃。
- 新增独立 stable 数学入口 `trajweave.recipes.marti_mars2.stable`，对齐 VERL 已注册的 `gspo` loss（含
  `seq-mean-token-mean` 聚合），提供 sequence-level GSPO、token/
  sequence TIS、ESS、官方 overlong penalty 和机制验收；默认 fidelity 配方不变，附 token/sequence smoke 配置。
- `TreeSearchProtocol.run_async` 支持 initial candidate 并发和 refinement wave，并按预留 node id 确定性落树；
  增加 `mode: marti_eval` 合约、verifier-best/RM selector、固定 LiveCodeBench v5 12 题 manifest 及标准输入
  verifier。该评测路径不进入 optimizer，缺少独立 endpoint/model 时 fail fast。

本轮完成 CPU 可审计公共能力和 launch contract；GPU stable 三配方与 LiveCodeBench 12 题结果仍需在资源可用时
按 acceptance 清单执行，不以搜索分数提升作为代码正确性的硬门槛。

CPU 验证：`tests/trajweave` `112 passed, 2 warnings`；`compileall -q trajweave verl` 和 `git diff --check` 通过。

## 19. 2026-08-10 稳定配方与异步 buffer CPU 验收补齐

- `PolicyBufferCoordinator` 的三步异步 fixture 已修正为真实覆盖版本边界：两个 policy group 分别完成 `3/2` 次更新，最后一轮故意注入一个落后两版的 `policy_b` 样本，该样本被计为 stale 并拒绝训练。
- fixture 现在显式推进 actor step 和 rollout-synced step，最终两个 group 的 policy lag 均为 `0`、pending queue 为 `0`，并输出 `async_buffer_acceptance.status=passed`。
- `stage1e_async_buffer_fixture.yaml` 已接入普通 runner smoke，运行时会执行上述验收，而不是只生成配置。
- 新增 stable CPU fixture，覆盖 token/sequence TIS、有限 ESS、GSPO loss 和 overlong penalty；两个 stable smoke 配置均可输出 `stable_cpu_fixture.acceptance.status=passed`。
- 新增回归测试后，MARTI 定向测试为 `28 passed, 2 warnings`；完整 TrajWeave MARTI/VERL 回归需在目标环境继续执行。

本节仍属于机制级 CPU 验收，不等价于 stable 配方的真实 GPU optimizer run，也不改变论文规模训练、PRIME `pyext` 和 LiveCodeBench 尚未完成的边界。

## 20. 2026-08-10 async buffer trainer 接入

- `marti_mars2.async_updates=true` 现在会向 VERL multi-actor launch 注入 `trajweave.async_buffer.enabled=true`；默认值为 `false`，因此既有 fidelity/Stage 1E 配置行为保持不变。
- multi-actor trainer 在启用该开关后，从 TransferQueue 读取 `worker_group/tree_id/raw_score/rollout_policy_step/rollout_global_step`，按 tree 原子写入 `PolicyBufferCoordinator`，只消费 ready 的 trainable policy group。
- stale 样本在进入 actor worker 前被丢弃并计数；每次 acknowledged weight sync 会推进对应 buffer 的 rollout-synced version。
- 新增 TQ metadata 归一化和 async 配置回归，当前 `tests/trajweave` 为 `120 passed, 2 warnings`。

本轮仍未在 GPU 上执行 `async_updates=true` 的多步训练；下一步必须用受控双 actor 配置验证真实 batch 门控、连续多步更新、policy lag 和 resume。

## 21. 2026-08-10 async 多步 GPU 验证（历史阻塞记录）

- HF-local multi-actor 3-step smoke 已完成：
  - `training/global_step=3`；
  - `trees_seen=3`、`samples_accepted=12`、`buffer/pending=0`；
  - `policy_a` 和 `policy_b` 各消费 6 个样本并各完成 3 次 actor update；
  - run return code 为 `0`，real verifier、checkpoint 和 multi-agent routing acceptance 通过。
- 该 run 初次暴露 no-transport HF 路径没有推进 buffer rollout-synced version，已补上同步 bookkeeping 和 manifest 回写，并加入 CPU 回归。
- 同配置 native vLLM 3-step 尝试在第一个 step 后续请求触发 `CUDA illegal memory access`，随后为 `EngineDeadError`；日志保留在临时 run 的 Ray worker error 中。该失败暂不能归因于 async buffer，当前把它作为 vLLM 多步生命周期/显存路径阻塞，不继续盲目重跑。

本节最后一条是当时的阶段性判断。第 22 节解决了 native vLLM 多步生命周期问题，但该次运行每步都有足量样本，没有覆盖样本真正跨 step 留在队列的场景。当前边界以第 0 节为准。

## 22. 2026-08-10 native vLLM 多步 lifecycle 修复与历史验收

> 本节是贡献者当时的本地实验记录，原始 run artifact 未入库。它证明了 vLLM sleep/wake 和权重同步问题的修复，但不证明跨 step 异步样本所有权完整。

`CUDA illegal memory access` 不是显存不足，也不是 async buffer 或 data parallel 本身造成。真实触发链路是：

1. 每轮 rollout 后 multi-actor trainer 调用 `sleep_replicas()`；
2. `MultiActorWeightSyncContract.sync_pending()` 原先只有在当前 actor version 已有磁盘 checkpoint 时才执行 transport；
3. `save_freq > 1` 时，前几个 step 没有 checkpoint，live actor weight sync 被跳过并返回 `checkpoint is not available`；
4. vLLM endpoint 因此没有被 weight-sync transport 唤醒；
5. 下一轮 generation 请求进入 sleeping EngineCore，并在 `execute_dummy_batch()` 中表现为 CUDA illegal memory access，随后成为 `EngineDeadError`。

修复后，无 checkpoint 的当前 actor version 也允许通过 live transport 同步；`CheckpointEnginePolicyEndpoint`
直接从 live actor 加载权重并 wake 对应 vLLM replica。新 actor version 不再继承旧 checkpoint 路径，避免旧文件被误认为当前版本。同时修复普通单 endpoint 请求携带 grouped routing 字段的问题；多步 acceptance 改为审计整个 run 的学习信号，并在 weight sync 后重新写入 buffer version metrics，确保 step summary 中 `rollout_synced_step` 为当前 step、`policy_lag=0`。

隔离验证依次通过：

- 单 actor、单 endpoint、V1 sleep mode 2-step：`CONTRIBUTOR_LOCAL_ARTIFACT/cxy-marti-v1-single2/runs/20260810-073122-native-vllm-single-actor-2step-f31a7b84`；
- 双 actor、双 endpoint、async 关闭、DP=1 2-step：`CONTRIBUTOR_LOCAL_ARTIFACT/cxy-marti-v1-dual2/runs/20260810-075807-native-vllm-dual-actor-2step-no-async-dp1-0cfb5b68`；
- 双 actor、双 endpoint、async 关闭、DP=2 2-step：`CONTRIBUTOR_LOCAL_ARTIFACT/cxy-marti-v1-dual2/runs/20260810-080131-native-vllm-dual-actor-2step-no-async-dp2-fixed-b8298f0e`。

DP=1 与 DP=2 均成功，确认 vLLM data-parallel multiprocess 不是根因。正式配置
`configs/marti_mars2/stage1e_multi_agent_vllm_async_smoke.yaml` 的 async 3-step run：

`CONTRIBUTOR_LOCAL_ARTIFACT/cxy-marti-stage1e-vllm-async/runs/20260810-081019-marti-mars2-stage1e-multi-agent-vllm-async-smoke-a6ec40d7`

最终状态为 `completed`，VERL return code `0`，自动 acceptance `passed`，且未再出现 CUDA 或 EngineDead
错误。关键指标：

```text
training/global_step = 3
trees_seen = 3
samples_accepted / samples_stale = 12 / 0
buffer/pending = 0
policy_a actor_step / rollout_synced_step / policy_lag = 3 / 3 / 0
policy_b actor_step / rollout_synced_step / policy_lag = 3 / 3 / 0
weight sync synchronized / pending = 1 / 0
policy_a loss / grad_norm = 0.8599502444 / 14.42705345
policy_b loss / grad_norm = -0.8702063560 / 4.81680727
```

`global_step_3/multi_actor_weight_sync.json` 显示 `synchronized=true`、`pending_groups=[]`。该历史结果说明 native vLLM
多步生命周期阻塞已经解除，且当前 batch 每步均可立即训练的双 Actor 同步路径可行。由于它没有触发样本跨 step 等待，不能作为跨 step 异步 buffer 的验收证据；该能力在生产路径中会 fail fast。

## 23. 2026-08-10 README 规范对齐、历史复验与推送状态

按照仓库 README 的 paper recipe 接入规范，对 MARTI-MARS² 做了完整工程结构清理：

- 新增通用 `CodeExecutionEnvironment` / `CodeTask`，将代码任务 observation、verifier 和 reward boundary 从
  MARTI recipe/emitter 中分离；
- 将 `TreeGroupCreditAllocator`、`TreePathCreditAllocator` 隔离到 `credit/marti_mars2/`，共享
  `tree_path.py` 只保留通用树计算；
- runner 改为只识别通用 `acceptance` 契约，MARTI 旧字段仅作为兼容别名；
- 增加独立 MARTI VERL extension、AgentLoop backend registry 和通用 online-turn writer，移除
  `runtime_config.py` 中的 MARTI recipe 特判；
- multi-actor trainer 的 canonical mode 改为 `trajweave_multi_actor_sync`，旧
  `trajweave_maporl_multi_actor_sync` 名称继续兼容 MAPoRL 配置；
- VERL 正式运行结束后会将 `global_step_*` checkpoint 目录、`multi_actor_weight_sync.json` 和
  `latest_checkpointed_iteration.txt` 登记到统一 artifact index，不复制大模型分片；
- README 已更新为五条可运行 MAS 路径，并补齐 MARTI 的目录结构、运行命令和 Paper Recipe Catalog 条目。

CPU/静态回归最终为 `156 passed, 4 warnings`；`compileall -q trajweave verl` 与 `git diff --check` 均通过。

规范清理后的双卡 native vLLM async 3-step 复验 run：

`CONTRIBUTOR_LOCAL_ARTIFACT/cxy-marti-stage1e-vllm-async/runs/20260810-122415-marti-mars2-stage1e-multi-agent-vllm-async-smoke-a9ca470b`

该 run 使用 GPU 0、3，最终 `completed`、return code `0`、通用 `acceptance.status=passed`，并满足：

```text
training/global_step = 3
policy_a actor_step / rollout_synced_step / policy_lag = 3 / 3 / 0
policy_b actor_step / rollout_synced_step / policy_lag = 3 / 3 / 0
weight_sync pending = 0
checkpoint/global_step_3 已登记到 artifact_index.jsonl
multi_actor_weight_sync.json 与 latest_checkpointed_iteration.txt 已登记
```

日志未出现 CUDA illegal memory access、EngineDeadError 或训练 Traceback。该次复验同时确认新的通用 trainer
mode、通用 acceptance 和 checkpoint artifact 登记在真实双卡训练路径上生效。

代码已提交并推送到 `origin/cxy-dev`：

```text
1d1b73c refactor: align MARTI integration with TrajWeave architecture
b4857f3 docs: register MARTI recipe and acceptance status
```

当时结论：MARTI 已完成 smoke/tiny/0.5B 双卡 native vLLM 多步历史验收，但该验收没有触发真正的跨 step 样本留存。当前可携带、可复验的能力边界和待办项以第 0 节为准。
