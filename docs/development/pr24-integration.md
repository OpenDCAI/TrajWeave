**QF -> LZ 算法整合与最小训练验收**

合并基线：LZ `9a9cc395145331057505bbf988a95fbbe7d9e06c`，QF `90caf53c7181759ab5e32b0333713c922ee7bab8`。整合保留 LZ 的 MARTI、后端注册、权重版本管理和运行记录，同时接入 QF 的新算法。验收结果须以真实训练日志为准；配置可解析、CPU 测试通过均不等于 GPU 训练完成。

**生成并运行最小训练配置**

在项目专用环境中，从仓库根目录执行：

```bash
export PYTHONNOUSERSITE=1
# MARTI 的 PRIME 代码评分器需要 pyext；环境没有运行任务时安装。
python -m trajweave.cli.prepare_prime_dependency

python -m trajweave.cli.prepare_smoke_suite \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --agentflow-model /path/to/Qwen3-0.6B \
  --output-dir /path/to/trajweave-smoke \
  --python /path/to/project-conda/bin/python

python -m trajweave.cli.run \
  --config /path/to/trajweave-smoke/configs/atgrpo.yaml
```

`--only atgrpo matpo` 可只生成指定项。生成器从仓库正式 YAML 模板派生配置，使用用户指定的真实本地模型和独立输出目录，不下载模型。`suite.json` 列出每个配置的 recipe、GPU 数量、来源模板和执行命令。运行前应为命令设置与该项 GPU 数量匹配的 `CUDA_VISIBLE_DEVICES`，各任务使用互不重叠的卡。

`--agentflow-model` 可省略；本次 0.5B 模型的 Planner 格式经常无效，0.6B 补验格式有效。生成器遇到 Qwen3 时关闭 thinking，以免极短响应预算被思考内容占满。小样本中同组奖励全相同仍会产生零优势，不代表训练器异常。

- DrMAS：数学、搜索各一项。
- MAPoRL、AgentFlow、GiGPO、CoMAS、AT-GRPO、MATPO、MrlX、WideSeek-R1、MARSHAL、MARFT、C3：分别验证。
- MARTI-MARS²：HF 与原生 vLLM 分别验证。
- CoMLRL：MAGRPO、MAREINFORCE、MARLOO、MAREMAX、IAC、MAAC、MADPO、MARLHF、迭代 MADPO、迭代 MARLHF 共十项。

共 25 个配置。每项使用 4 条训练样本、2 条验证样本，目标为 2 个训练步；独立 Actor/Critic 方法需要 2 至 4 张卡。MrlX 的两步覆盖延迟 Adapter 更新和退出时的待处理数据训练。MARLHF 还包含偏好收集和奖励模型训练。HF 本地生成目前使用 CPU，参数训练使用 CUDA；原生 vLLM 项使用 GPU 生成。

`prepare_prime_dependency` 从 PyPI 下载固定 SHA256 的 pyext 0.6，只提取模块并修复 Python 3.11 已删除的 `inspect.getargspec`，保存来源及安装文件哈希。可用 `--target /path/to/dependencies` 写入独立目录，再将该目录加入 MARTI 配置的 `verl.env.PYTHONPATH`。没有依赖时代码判分会标记为本地 fallback，严格 PRIME 验收不会通过。MARTI 的代码响应预算为 256 token，其他最小配置为 96 token。

**整合时修复的问题**

- AT-GRPO 验证使用最终 Solver 答案和环境评分，训练混合奖励另存为指标；Verifier 的 `APPROVED` 不再被当作最终答案。
- AT-GRPO/CoMLRL 保留数学真值原文。仅支持整数的旧接口严格检查输入，避免把 `1/2`、`0.5` 静默变成 `2`、`5`。
- CoMLRL 接入统一后端注册；保留旧验证函数导入路径作为兼容转发。
- MrlX/MADPO 自定义更新路径同步记录 LZ 的策略版本和缓冲区更新步数。
- MrlX 尚未持久化 Adapter replay，因此拒绝从旧 checkpoint 恢复；新训练和空目录的 `auto` 模式可用。独立 Actor/Critic、偏好训练器保留各自的恢复限制。
- 普通多 Actor 训练保留 LZ 的完整恢复路径；存在 checkpoint 标记而实际目录丢失时明确报错。
- recipe 配置路径正确转义中文、空格及 Hydra 语法字符。
- MARTI 的 `verl_plan` 只生成计划，训练产物验收在 `verl_train` 执行。
- 补齐 HF worker 调用共享 TQ 写入器所需的统计接口，原生 vLLM 按当前批次记录版本。
- MATPO 的优势计算选列包含 `traj_uid`，确保父子回合可以按完整轨迹归组。
- MATPO 忽略没有有效 token 的补齐行，避免复制的身份字段造成父子关系误报；GRPO 基线也排除这些补齐行。
- 数学符号判分在 Ray 异步线程下改用限时独立进程，避免 `math-verify` 的主线程 signal 限制导致正确答案被静默判错。
- CoMLRL 迭代训练记录实际更新步数和优化指标，按频率保存 checkpoint，并在控制器完成后执行验证和最终保存。其迭代预算仍由 `num_iterations`、`num_train_epochs` 和偏好批次控制；smoke 配置使用一轮、一个 epoch。
- MARTI HF 使用专用 MCTS emitter，并兼容共享 Hook 未传默认选列的调用。
- HF 角色提示传递聊天模板参数，支持 Qwen3 关闭 thinking。
- 多 Actor 的启动校验按各角色实际 GPU 数验证；独立 critic 单独校验，避免误把三张独立模型卡当作三卡数据并行。
- Ray 临时目录使用短哈希，避免算法名和中文输出目录导致 Unix socket 超过 107 字节。
- PRIME 的函数调用任务逐项判分保留 `fn_name`，避免部分通过时错误切换到 stdin 测试。

**验收口径**

需共同检查进程正常退出、实际 optimizer 步数、在线轨迹、有限的损失/梯度和 checkpoint；预期有学习信号的方法还需检查非零梯度及参数变化。多 Actor 路径应确认各组正确分发、更新和同步。规则生成的 tiny 数据仅用于检查训练链路，不能据此推断论文指标、收敛质量或算法优劣。

本次实测状态和证据将在训练完成后补充。
