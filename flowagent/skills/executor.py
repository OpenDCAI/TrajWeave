"""
Skill执行器
"""
from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from flowagent.logger import get_logger
from flowagent.skills.base import Skill

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


# 执行模式 → 工厂函数名 的映射
_MODE_FACTORY_MAP = {
    "simple": "create_simple_agent",
    "react": "create_react_agent",
    "plan_execute": "create_plan_execute_agent",
    "validation_retry": "create_validation_agent",
    "vlm": "create_vlm_agent",
    "parallel": "create_parallel_agent",
}


def _create_agent_for_skill(
    name: str,
    system_prompt: str,
    model_name: Optional[str],
    tools: list,
    mode: str = "simple",
):
    """根据 execution_mode 调用对应工厂函数创建 Agent。"""
    import flowagent.core.factory as factory

    factory_name = _MODE_FACTORY_MAP.get(mode, "create_simple_agent")
    create_fn = getattr(factory, factory_name)

    kwargs: Dict[str, Any] = {
        "role_name": name,
        "system_prompt": system_prompt,
    }
    if model_name:
        kwargs["model_name"] = model_name
    if tools and mode in ("react", "plan_execute"):
        kwargs["tools"] = tools

    return create_fn(**kwargs)


class SkillExecutor:
    """Skill执行器 - 通过 Agent/Workflow 执行 Skill"""

    async def execute(
        self,
        skill: Skill,
        input_data: Dict[str, Any],
        state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """执行 Skill。

        Args:
            skill: 要执行的 Skill
            input_data: 输入数据（用于格式化 user_prompt_template）
            state: 可选的 MainState，未提供时自动构建

        Returns:
            执行结果 dict
        """
        log.info(f"执行Skill: {skill.name}")

        # 运行前置钩子
        context = await skill.run_pre_hooks(input_data.copy())

        # 构建 state
        if state is None:
            state = self._build_state(skill, context)

        # 执行
        if not skill.steps:
            result = await self._execute_single(skill, state)
        else:
            result = await self._execute_workflow(skill, state)

        # 运行后置钩子
        result = await skill.run_post_hooks(result, context)
        return result

    @staticmethod
    def _build_state(skill: Skill, context: Dict[str, Any]):
        from flowagent.state.base import MainState, MainRequest

        user_prompt = skill.format_user_prompt(**context)
        return MainState(request=MainRequest(target=user_prompt))

    @staticmethod
    async def _execute_single(skill: Skill, state) -> Dict[str, Any]:
        """单步 Skill：创建单个 Agent 执行。"""
        agent = _create_agent_for_skill(
            name=skill.name,
            system_prompt=skill.system_prompt,
            model_name=skill.model_name,
            tools=skill.tools,
            mode=skill.execution_mode,
        )
        result = agent.execute(state)
        import asyncio
        if asyncio.iscoroutine(result):
            result = await result

        agent_results = getattr(state, "agent_results", {}) or {}
        return agent_results.get(skill.name, {"results": result})

    @staticmethod
    async def _execute_workflow(skill: Skill, state) -> Dict[str, Any]:
        """多步 Skill：构建 Workflow 执行。"""
        from flowagent.workflow.base import WorkflowBuilder

        builder = WorkflowBuilder(type(state), name=skill.name)
        node_names = []

        for step in skill.steps:
            name = step["name"]
            node_names.append(name)
            agent = _create_agent_for_skill(
                name=name,
                system_prompt=step.get("system_prompt", skill.system_prompt),
                model_name=step.get("model_name", skill.model_name),
                tools=step.get("tools", []),
                mode=step.get("execution_mode", skill.execution_mode),
            )
            builder.add_agent_node(name, agent)

        builder.chain(*node_names)
        workflow = builder.compile()
        final_state = await workflow.run_async(state)

        if isinstance(final_state, dict):
            return final_state.get("agent_results", {})
        return getattr(final_state, "agent_results", {})
