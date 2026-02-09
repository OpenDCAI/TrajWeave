"""Workflow 示例：在 workflow 层 bind_tools，然后节点可调用工具

目标：验证不在 create_react_agent(tools=...) 里传工具，也能在 workflow 中声明工具并注入。

运行：
  D:/Anaconda/envs/flowagent/python.exe examples/workflow_bind_tools_smoke.py

依赖：
- .env 或环境变量：DF_API_URL / DF_API_KEY
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from langchain_core.tools import tool

from flowagent.core.factory import create_react_agent
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
            target=(
                "你必须调用 calculator 工具计算 789*32-15。\n"
                "最终只输出结果数字，不要解释。"
            ),
        )
    )

    # 注意：这里 tools=[]，后续完全通过 workflow.bind_tools 注入
    calc_agent = create_react_agent(
        tools=[],
        role_name="calc",
        system_prompt=(
            "你是一个计算助手。你必须调用 calculator 工具完成计算。"
            "最终只输出计算结果数字（不要解释）。"
        ),
        parser_type="text",
        temperature=0.0,
        max_tokens=64,
        tool_mode="required",
    )

    builder = WorkflowBuilder(type(state), name="workflow_bind_tools_smoke")
    builder.bind_tools("calc", [calculator])

    wf = builder.add_node("calc", agent_node(calc_agent, name="calc")).set_entry("calc").compile()

    final_state = await wf.run_async(state)
    agent_results = final_state.get("agent_results", {}) if isinstance(final_state, dict) else getattr(final_state, "agent_results", {})

    print("=== workflow bind_tools smoke ===")
    print("model:", model)
    print("calc:", agent_results.get("calc", {}).get("results"))


if __name__ == "__main__":
    asyncio.run(main())
