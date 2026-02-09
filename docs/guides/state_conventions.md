# 状态传递规范（Workflow）

这份规范用于让 workflow 里的“上游输出 → 下游输入”有统一、稳定的读取方式，避免每个示例/节点各写各的，导致串联困难。

## 1. 两类状态容器：`agent_results` 与 `temp_data`

在 FlowAgent 的 `MainState`（以及继承它的 State）中，约定使用：

- `state.agent_results: Dict[str, Any]`
  - **用途**：存放“节点（Agent）级别”的可复用产物（适合下游读取/汇总/落盘）
  - **建议**：下游读取上游输出，优先从这里拿

- `state.temp_data: Dict[str, Any]`
  - **用途**：存放“运行期临时上下文”（不一定是业务结果），例如 workflow 级注入对象、调试信息、缓存、运行参数等
  - **建议**：使用带前缀的 key（如 `_workflow_xxx`）避免和业务字段冲突

## 2. `agent_results` 的标准结构（推荐）

对每个节点（通常与 `agent_node(..., name=...)` 的 name 对齐），推荐结构如下：

```python
state.agent_results["<node>"] = {
    "status": "ok" | "error",
    "results": <dict | str | Any>,
    "error": <str | None>,
    # 可选字段（由 BaseAgent / 策略写入也允许）：
    # "pre_tool_results": {...},
    # "post_tools": [...],
    # "meta": {...},
}
```

### 读取约定

- 上游输出（业务结果）统一从：
  - `state.agent_results["<node>"]["results"]` 读取

- 判断失败统一从：
  - `state.agent_results["<node>"]["status"] == "error"` 或
  - `state.agent_results["<node>"]["error"] is not None`

说明：为了兼容早期代码，错误场景也会在 `results` 里放一份 `{ "error": "..." }`，但 **推荐** 下游以 `status/error` 为主。

## 3. `temp_data` 的使用约定

- 用于 workflow 级“注入/共享对象”，例如：
  - `_workflow_tool_manager`
  - `_workflow_tool_bindings`

- 用于节点间“短期中间态”，例如：
  - `_cache_xxx`
  - `_debug_xxx`

建议：业务可复用结果仍应写进 `agent_results`，把 `temp_data` 当作运行时环境变量。

## 4. 示例：下游读取上游输出

```python
expr_text = (
    state.agent_results
    .get("expr", {})
    .get("results", {})
    .get("text")
)

if state.agent_results.get("expr", {}).get("status") == "error":
    # 这里可走 fallback
    ...
```

## 5. 实现位置

- 节点适配器会尽力把 `agent_results` 规范化：
  - [flowagent/workflow/base.py](../../flowagent/workflow/base.py)

- 典型示例：
  - [examples/workflow_minimal_example.py](../../examples/workflow_minimal_example.py)
  - [examples/workflow_simple_plus_react_example.py](../../examples/workflow_simple_plus_react_example.py)
  - [examples/workflow_bind_tools_smoke.py](../../examples/workflow_bind_tools_smoke.py)
