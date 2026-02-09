"""Agent 节点适配器 smoke test（不依赖外部 LLM/Key）

运行：
  D:/Anaconda/envs/flowagent/python.exe examples/workflow_agent_node_smoke.py

预期：
  - workflow 能把“伪 agent”作为节点执行
  - state.agent_results 中出现对应 role 的 results
  - 下游节点能读取上游 agent_results 并继续更新 state
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from flowagent.workflow.base import WorkflowBuilder, agent_node


class SmokeState(BaseModel):
    agent_results: dict = {}
    summary: str = ""


class FakeAgent:
    def __init__(self, role_name: str, value: int):
        self._role_name = role_name
        self._value = value

    @property
    def role_name(self) -> str:
        return self._role_name

    async def execute(self, state: SmokeState, **kwargs):
        # 模拟 BaseAgent.update_state_result 产出结构
        state.agent_results[self.role_name] = {
            "pre_tool_results": {},
            "post_tools": [],
            "results": {"value": self._value},
        }
        return state


async def main() -> None:
    a1 = FakeAgent("A1", 7)

    builder = WorkflowBuilder(SmokeState, name="workflow_agent_node_smoke")

    async def summarize(state: SmokeState):
        v = state.agent_results.get("A1", {}).get("results", {}).get("value")
        return {"summary": f"A1.value={v}"}

    wf = (
        builder.add_node("agent_a1", agent_node(a1, name="A1"))
        .add_node("summarize", summarize)
        .add_edge("agent_a1", "summarize")
        .set_entry("agent_a1")
        .compile()
    )

    init_state = SmokeState(agent_results={}, summary="")
    final_state = await wf.run_async(init_state)

    print("=== workflow agent_node smoke ===")
    if isinstance(final_state, dict):
        print(final_state)
    else:
        print(final_state.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
