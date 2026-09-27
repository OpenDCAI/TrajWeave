# TrajWeave Bugfix Rubric

这套门禁用于判断运行时修复是否真的完成。它不是“跑一次 pytest 看总数”，而是逐项验证训练契约；任一项失败，整体结果就是 `FAIL`。

| ID  | 门禁                         | 合格条件                                                               |
|-----|------------------------------|------------------------------------------------------------------------|
| R01 | 配置入口完整性               | `recipe`、`mode` 必须显式填写，缺失或空白立即失败                       |
| R02 | Recipe/Mode 匹配             | mode 必须属于 recipe 白名单，拼写错误不能回退到 smoke                   |
| R03 | 训练状态真实性               | `verl_train` 必须实际启动 VERL，disabled、dry-run、缺失 launch 都失败   |
| R04 | Plan 与 Train 隔离           | 只有 `verl_plan` 可以 dry-run，最终状态必须是 `planned`                 |
| R05 | 训练进度真实性               | 目标步数为正整数，实际步数达标，指标无 NaN/Inf                          |
| R06 | 失败 Trajectory 隔离         | 任一失败 prompt 立即中止，失败轨迹不能进入训练 batch                    |
| R07 | GRPO Group 完整性与超时      | session 组完整采样；rollout 丢失时必须超时报错，不能永久等待             |
| R08 | Worker Group 更新完整性      | 每个 trainable group 每步都有样本；缺组时任何 Actor 都不能更新          |
| R09 | Rollout 非阻塞性             | 调度只提交后台任务；任务有强引用、异常被消费、完成后引用释放             |
| R10 | 禁止静默 Fallback            | tokenizer、PyTorch、CUDA、compat patch 失败默认硬报错                    |
| R11 | Reward/配置输入完整性        | ground truth 必填；仓库内所有配置满足严格 run contract                  |
| R12 | 全量回归                     | `tests/trajweave` 无失败、skip 或 xfail                                 |
| R13 | 真实训练产物审计             | run 完成，日志、指标、trajectory、checkpoint、artifact 均真实存在且有效 |

执行方式：

```bash
python scripts/run_bugfix_rubric.py \
  --e2e-run-dir outputs/trajweave/runs/<真实训练-run-id> \
  --output outputs/rubrics/bugfix-rubric.json
```

执行器会为每个 Rubric 单独运行测试、保存 JUnit XML，并记录 Git SHA 和当前 diff hash。零测试、skip、xfail、测试失败、真实产物缺失都会判为 `FAIL`。
