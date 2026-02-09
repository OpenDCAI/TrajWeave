"""Workflow 示例：条件边/路由（不依赖 LLM）

目标：验证 WorkflowBuilder.add_conditional_edges 支持 path_map，并演示：
- step1 成功 -> 走 step2
- step1 失败 -> 走 fallback

运行：
  D:/Anaconda/envs/flowagent/python.exe examples/workflow_conditional_routing_smoke.py
"""

from __future__ import annotations

import asyncio

from flowagent.state import MainRequest, MainState
from flowagent.workflow.base import WorkflowBuilder


def _mk_state(*, fail: bool) -> MainState:
    return MainState(
        request=MainRequest(
            language="zh",
            target="fail" if fail else "ok",
        )
    )


async def step1(state: MainState):
    if (state.request.target or "").strip().lower() == "fail":
        err = "step1 forced failure"
        state.agent_results["step1"] = {
            "status": "error",
            "error": err,
            "results": {"error": err},
        }
        return {"agent_results": state.agent_results}

    state.agent_results["step1"] = {
        "status": "ok",
        "error": None,
        "results": {"value": 42},
    }
    return {"agent_results": state.agent_results}


def router(state: MainState) -> str:
    status = (state.agent_results.get("step1", {}) or {}).get("status")
    return "fallback" if status == "error" else "next"


async def step2(state: MainState):
    v = (state.agent_results.get("step1", {}) or {}).get("results", {}).get("value")
    state.agent_results["step2"] = {
        "status": "ok",
        "error": None,
        "results": {"summary": f"step1.value={v}"},
    }
    return {"agent_results": state.agent_results}


async def fallback(state: MainState):
    reason = (state.agent_results.get("step1", {}) or {}).get("error")
    state.agent_results["fallback"] = {
        "status": "ok",
        "error": None,
        "results": {"summary": f"fallback because: {reason}"},
    }
    return {"agent_results": state.agent_results}


async def run_case(*, fail: bool) -> None:
    state = _mk_state(fail=fail)
    builder = WorkflowBuilder(type(state), name="workflow_conditional_routing_smoke")

    wf = (
        builder.add_node("step1", step1)
        .add_node("step2", step2)
        .add_node("fallback", fallback)
        .add_conditional_edges(
            "step1",
            router,
            path_map={
                "next": "step2",
                "fallback": "fallback",
            },
        )
        .set_entry("step1")
        .compile()
    )

    final_state = await wf.run_async(state)
    agent_results = (
        final_state.get("agent_results", {})
        if isinstance(final_state, dict)
        else getattr(final_state, "agent_results", {})
    )

    print("\n=== conditional routing case ===")
    print("fail:", fail)
    print("step1:", agent_results.get("step1"))
    print("step2:", agent_results.get("step2"))
    print("fallback:", agent_results.get("fallback"))


async def main() -> None:
    await run_case(fail=False)
    await run_case(fail=True)


if __name__ == "__main__":
    asyncio.run(main())
