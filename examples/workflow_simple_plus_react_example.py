"""Workflow 示例：create_simple_agent + create_react_agent 串联（带工具）

目标：演示一个最小可讲清楚的 workflow：
- 节点1：Simple agent 生成一个需要精确计算的表达式
- 节点2：ReAct agent 必须调用 calculator 工具得到精确结果

运行：
  D:/Anaconda/envs/flowagent/python.exe examples/workflow_simple_plus_react_example.py

依赖：
- .env 或环境变量：DF_API_URL / DF_API_KEY

提示：
- 该示例会强制 ReAct 节点 tool_mode="required"，确保确实发生工具调用
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from langchain_core.tools import tool

from flowagent.core.factory import create_simple_agent, create_react_agent
from flowagent.state import MainRequest, MainState
from flowagent.workflow.base import WorkflowBuilder, agent_node


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except Exception:
        pass


def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v or v.lower() == "test":
        raise RuntimeError(f"缺少环境变量 {name}（或仍为默认 test）")
    return v


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression. Input should be numbers and +-*/() only."""
    allowed = set("0123456789+-*/(). %")
    if any(ch not in allowed for ch in expression):
        raise ValueError("expression contains invalid characters")
    # 禁用 builtins，避免注入
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)


async def main() -> None:
    _try_load_dotenv()

    api_url = _require_env("DF_API_URL")
    api_key = _require_env("DF_API_KEY")

    model = os.getenv("DF_MODEL", "").strip() or "gpt-4o"

    state = MainState(
        request=MainRequest(
            language="zh",
            chat_api_url=api_url.rstrip("/"),
            api_key=api_key,
            model=model,
            target="生成一个需要精确计算的算式（只输出算式本身，不要解释），例如 123*45+6。",
        )
    )

    # 节点1：生成算式
    expr_agent = create_simple_agent(
        role_name="expr",
        system_prompt="你是一个严谨的出题助手。只输出一个算式字符串。",
        parser_type="text",
        temperature=0.0,
        max_tokens=64,
    )

    # 节点2：必须用工具计算
    react_agent = create_react_agent(
        tools=[calculator],
        role_name="calc",
        system_prompt=(
            "你是一个计算助手。你必须调用 calculator 工具完成计算。"
            "最终只输出计算结果数字（不要单位、不要解释）。"
        ),
        parser_type="text",
        temperature=0.0,
        max_tokens=64,
        tool_mode="required",
    )

    builder = WorkflowBuilder(type(state), name="workflow_simple_plus_react")

    async def calc_node(s: MainState):
        expr_out = s.agent_results.get("expr", {}).get("results")
        expr_text = expr_out.get("text") if isinstance(expr_out, dict) else str(expr_out)
        expr_text = (expr_text or "").strip()

        # 将算式作为 ReAct 的任务输入
        s.request.target = (
            "请计算以下表达式的值。必须调用 calculator(expression)。"
            f"\n\nexpression: {expr_text}"
        )
        return await agent_node(react_agent, name="calc")(s)

    wf = (
        builder.add_node("expr", agent_node(expr_agent, name="expr"))
        .add_node("calc", calc_node)
        .add_edge("expr", "calc")
        .set_entry("expr")
        .compile()
    )

    final_state = await wf.run_async(state)

    if isinstance(final_state, dict):
        agent_results = final_state.get("agent_results", {})
    else:
        agent_results = getattr(final_state, "agent_results", {})

    print("=== workflow simple + react example ===")
    print("model:", model)
    print("expr:", agent_results.get("expr", {}).get("results"))
    print("calc:", agent_results.get("calc", {}).get("results"))


if __name__ == "__main__":
    asyncio.run(main())
