"""最小 workflow 示例：两个 create_simple_agent 串联

目标（验收点 4）：
- 用 workflow 装配层把两个 Agent 串起来（节点=Agent，边=顺序）
- 第二个节点读取第一个节点输出，形成最终回答

运行：
  D:/Anaconda/envs/flowagent/python.exe examples/workflow_minimal_example.py

依赖：
- 需要在 .env 或环境变量中设置：DF_API_URL / DF_API_KEY
  例如（.env）：
    DF_API_URL=http://.../v1
    DF_API_KEY=sk-...

说明：
- 这里使用 parser_type="text"，避免 JSON 解析带来的噪音
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from flowagent.core.factory import create_simple_agent
from flowagent.state import MainRequest, MainState
from flowagent.workflow.base import WorkflowBuilder, agent_node


def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v or v.lower() == "test":
        raise RuntimeError(f"缺少环境变量 {name}（或仍为默认 test）")
    return v


async def main() -> None:
    # 可选：如果装了 python-dotenv，则从项目根目录 .env 读取
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except Exception:
        pass

    api_url = _require_env("DF_API_URL")
    api_key = _require_env("DF_API_KEY")

    model = os.getenv("DF_MODEL", "").strip() or os.getenv("DF_VLM_MODEL", "").strip() or "gpt-4o"

    user_target = "给我 3 条要点，说明为什么 workflow 编排有价值。"

    state = MainState(
        request=MainRequest(
            language="zh",
            chat_api_url=api_url.rstrip("/"),
            api_key=api_key,
            model=model,
            target=user_target,
        )
    )

    # Agent 1：先生成草稿/要点
    drafter = create_simple_agent(
        role_name="draft",
        system_prompt="你是一个助手。请输出简洁的中文要点。",
        parser_type="text",
        temperature=0.2,
        max_tokens=256,
    )

    # Agent 2：读取上游输出后，润色为最终版本
    finisher = create_simple_agent(
        role_name="final",
        system_prompt="你是一个写作润色助手。基于上下文输出最终答案。",
        parser_type="text",
        temperature=0.2,
        max_tokens=256,
    )

    builder = WorkflowBuilder(type(state), name="workflow_minimal_example")

    async def final_node(s: MainState):
        upstream = s.agent_results.get("draft", {}).get("results")
        s.request.target = (
            "请把下面的草稿要点整理成一段更自然的最终答复（仍然保持 3 条要点）。\n\n"
            f"原始问题：{user_target}\n\n"
            f"草稿输出：\n{upstream}"
        )
        return await agent_node(finisher, name="final")(s)

    wf = (
        builder.add_node("draft", agent_node(drafter, name="draft"))
        .add_node("final", final_node)
        .add_edge("draft", "final")
        .set_entry("draft")
        .compile()
    )

    final_state = await wf.run_async(state)

    # LangGraph 可能返回 dict 或 state；两种都兼容打印
    if isinstance(final_state, dict):
        agent_results = final_state.get("agent_results", {})
    else:
        agent_results = getattr(final_state, "agent_results", {})

    print("=== workflow minimal example ===")
    print("model:", model)
    print("draft:\n", agent_results.get("draft", {}).get("results"))
    print("final:\n", agent_results.get("final", {}).get("results"))


if __name__ == "__main__":
    asyncio.run(main())
