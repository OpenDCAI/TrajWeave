"""Workflow 骨架最小自测（不依赖外部 LLM/Key）

运行：
  D:/Anaconda/envs/flowagent/python.exe examples/workflow_skeleton_smoke.py

预期：
  - 能成功 build/compile/run
  - final state 中 x/y 被节点更新
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from flowagent.workflow.base import WorkflowBuilder


class SmokeState(BaseModel):
    x: int = 0
    y: int = 0


def _get(state, key: str):
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key)


async def main() -> None:
    builder = WorkflowBuilder(SmokeState, name="workflow_skeleton_smoke")

    def inc_x(state):
        return {"x": int(_get(state, "x")) + 1}

    def set_y(state):
        return {"y": int(_get(state, "x")) + 10}

    wf = (
        builder.add_node("inc_x", inc_x)
        .add_node("set_y", set_y)
        .add_edge("inc_x", "set_y")
        .set_entry("inc_x")
        .compile()
    )

    init_state = SmokeState(x=1, y=0)
    final_state = await wf.run_async(init_state)

    print("=== workflow skeleton smoke ===")
    print("init:", init_state.model_dump())

    if isinstance(final_state, dict):
        print("final(dict):", final_state)
    else:
        # 兼容 BaseModel
        dump = getattr(final_state, "model_dump", None)
        print("final:", dump() if dump else final_state)


if __name__ == "__main__":
    asyncio.run(main())
