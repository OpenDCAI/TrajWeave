"""Workflow registry smoke test（不依赖 LLM）

目标：验证 workflow 可以按名字注册/创建/运行。

运行：
  D:/Anaconda/envs/flowagent/python.exe examples/workflow_registry_smoke.py
"""

from __future__ import annotations

import asyncio

from flowagent.state import MainRequest, MainState
from flowagent.workflow.base import WorkflowBuilder
from flowagent.workflow.registry import WorkflowRegistry, register_workflow


@register_workflow("conditional_demo")
def build_conditional_demo():
	"""返回一个已编译 Workflow（或 WorkflowBuilder 也可以）。"""
	builder = WorkflowBuilder(MainState, name="conditional_demo")

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
	return wf


def _mk_state(*, fail: bool) -> MainState:
	return MainState(
		request=MainRequest(
			language="zh",
			target="fail" if fail else "ok",
		)
	)


async def run_case(*, fail: bool) -> None:
	wf = WorkflowRegistry.create("conditional_demo")
	state = _mk_state(fail=fail)
	final_state = await wf.run_async(state)

	agent_results = (
		final_state.get("agent_results", {})
		if isinstance(final_state, dict)
		else getattr(final_state, "agent_results", {})
	)

	print("\n=== workflow registry case ===")
	print("fail:", fail)
	print("step1:", agent_results.get("step1"))
	print("step2:", agent_results.get("step2"))
	print("fallback:", agent_results.get("fallback"))


async def main() -> None:
	await run_case(fail=False)
	await run_case(fail=True)


if __name__ == "__main__":
	asyncio.run(main())
