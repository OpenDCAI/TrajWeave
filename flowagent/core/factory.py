"""
FlowAgent 便捷工厂函数

提供快速创建各类Agent的工厂函数，简化Agent的创建流程。

与Paper2Any的区别：
- Paper2Any: 必须继承BaseAgent，通过property定义role_name和模板名称
- FlowAgent工厂函数: 可以直接传递system_prompt字符串，无需创建子类

使用示例：
    # 方式1: 使用工厂函数（简单快捷）
    agent = create_react_agent(
        tools=[search_tool],
        role_name="researcher",
        system_prompt="你是一个研究员"
    )

    # 方式2: 继承BaseAgent（与Paper2Any一致）
    class MyAgent(BaseAgent):
        @property
        def role_name(self) -> str:
            return "MyAgent"
        @property
        def system_prompt_template_name(self) -> str:
            return "my_agent_system"
        ...
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable, Type
from langchain_core.tools import Tool

from flowagent.logger import get_logger

log = get_logger(__name__)


# =============================================================================
# 动态Agent类 - 支持直接传递system_prompt
# =============================================================================

def _create_dynamic_agent_class(
    role_name: str,
    system_prompt: str,
    task_prompt_template: Optional[str] = None,
):
    """
    动态创建Agent子类，支持直接传递system_prompt

    这是与Paper2Any的主要区别：
    - Paper2Any必须通过继承和property定义
    - 这里可以直接传递字符串
    """
    from flowagent.core.base_agent import BaseAgent

    class DynamicAgent(BaseAgent):
        """动态创建的Agent类"""

        _role_name: str
        _system_prompt: str
        _task_prompt_template: str

        @property
        def role_name(self) -> str:
            return self._role_name

        @property
        def system_prompt_template_name(self) -> str:
            # 返回一个特殊标记，表示使用直接传递的prompt
            return "__direct_prompt__"

        @property
        def task_prompt_template_name(self) -> str:
            return self._task_prompt_template

        def build_messages(self, state, pre_tool_results):
            """重写消息构建，支持直接使用system_prompt字符串"""
            from langchain_core.messages import SystemMessage, HumanMessage
            from flowagent.prompts.prompt_template import PromptsTemplateGenerator

            # 使用直接传递的system_prompt
            sys_prompt = self._system_prompt

            # 添加解析器格式说明
            format_instruction = self.parser.get_format_instruction()
            if format_instruction and not self.use_vlm:
                sys_prompt += f"\n\n{format_instruction}"

            # 渲染任务提示词（如果有模板）
            task_params = self.get_task_prompt_params(pre_tool_results)

            # 尝试使用模板，如果失败则使用默认格式
            try:
                ptg = PromptsTemplateGenerator(state.request.language)
                task_prompt = ptg.render(self._task_prompt_template, **task_params)
            except Exception:
                # 模板不存在，使用简单格式
                task_prompt = f"任务: {state.request.target}\n\n上下文: {task_params}"

            return [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=task_prompt),
            ]

    # 注意：类体作用域不会捕获外层函数局部变量（role_name/system_prompt 等），
    # 需要在类定义之后再写入这些动态属性。
    DynamicAgent._role_name = role_name
    DynamicAgent._system_prompt = system_prompt
    DynamicAgent._task_prompt_template = task_prompt_template or "default_task"

    # 设置类名
    DynamicAgent.__name__ = f"Dynamic{role_name}Agent"
    DynamicAgent.__qualname__ = f"Dynamic{role_name}Agent"

    return DynamicAgent


def create_react_agent(
    tools: List[Tool],
    role_name: str = "react_agent",
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    task_prompt_template: Optional[str] = None,
    **kwargs
) -> "BaseAgent":
    """
    创建ReAct Agent（带工具调用的推理+行动循环）

    与Paper2Any的区别：这里可以直接传递system_prompt字符串，无需创建子类。

    Args:
        tools: 工具列表
        role_name: Agent角色名称
        system_prompt: 系统提示词（直接传递字符串）
        model_name: 模型名称
        task_prompt_template: 任务提示词模板名称（可选）
        **kwargs: 其他BaseAgent参数

    Returns:
        配置好的BaseAgent实例

    Example:
        >>> from langchain_core.tools import tool
        >>> @tool
        ... def search(query: str) -> str:
        ...     return f"搜索结果: {query}"
        >>> agent = create_react_agent(
        ...     tools=[search],
        ...     role_name="researcher",
        ...     system_prompt="你是一个研究员，负责搜索信息"
        ... )
        >>> result = await agent.execute(state)
    """
    from flowagent.core.execution_config import ExecutionConfig, ExecutionMode
    from flowagent.tools.manager import ToolManager

    # 创建工具管理器并注册工具
    tool_manager = ToolManager()
    for t in tools:
        tool_manager.register_post_tool(t, role=role_name)

    # 创建执行配置
    execution_config = ExecutionConfig(
        mode=ExecutionMode.REACT,
        model_name=model_name,
        **{k: v for k, v in kwargs.items() if k in ExecutionConfig.__dataclass_fields__}
    )

    # 动态创建Agent类（支持直接传递system_prompt）
    DynamicAgentClass = _create_dynamic_agent_class(
        role_name=role_name,
        system_prompt=system_prompt or "你是一个智能助手，可以使用工具来完成任务。",
        task_prompt_template=task_prompt_template,
    )

    # 创建Agent实例
    agent = DynamicAgentClass(
        tool_manager=tool_manager,
        model_name=model_name,
        execution_config=execution_config,
        **{k: v for k, v in kwargs.items() if k not in ExecutionConfig.__dataclass_fields__}
    )

    log.info(f"创建ReAct Agent: {role_name}，工具数量: {len(tools)}")
    return agent


def create_plan_execute_agent(
    tools: List[Tool],
    role_name: str = "plan_execute_agent",
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    max_plan_steps: int = 10,
    max_replanning_rounds: int = 3,
    require_plan_approval: bool = False,
    task_prompt_template: Optional[str] = None,
    **kwargs
) -> "BaseAgent":
    """
    创建Plan-and-Execute Agent（动态计划+重规划）

    Args:
        tools: 工具列表
        role_name: Agent角色名称
        system_prompt: 系统提示词（直接传递字符串）
        model_name: 模型名称
        max_plan_steps: 最大计划步骤数
        max_replanning_rounds: 最大重规划轮数
        require_plan_approval: 是否需要用户审批计划
        task_prompt_template: 任务提示词模板名称（可选）
        **kwargs: 其他参数

    Returns:
        配置好的BaseAgent实例
    """
    from flowagent.core.execution_config import ExecutionConfig, ExecutionMode
    from flowagent.tools.manager import ToolManager

    tool_manager = ToolManager()
    for t in tools:
        tool_manager.register_post_tool(t, role=role_name)

    execution_config = ExecutionConfig(
        mode=ExecutionMode.PLAN_EXECUTE,
        model_name=model_name,
        max_plan_steps=max_plan_steps,
        max_replanning_rounds=max_replanning_rounds,
        require_plan_approval=require_plan_approval,
        **{k: v for k, v in kwargs.items() if k in ExecutionConfig.__dataclass_fields__}
    )

    # 动态创建Agent类
    DynamicAgentClass = _create_dynamic_agent_class(
        role_name=role_name,
        system_prompt=system_prompt or "你是一个任务规划和执行专家。",
        task_prompt_template=task_prompt_template,
    )

    agent = DynamicAgentClass(
        tool_manager=tool_manager,
        model_name=model_name,
        execution_config=execution_config,
        **{k: v for k, v in kwargs.items() if k not in ExecutionConfig.__dataclass_fields__}
    )

    log.info(f"创建Plan-Execute Agent: {role_name}")
    return agent


def create_validation_agent(
    validators: List[Callable],
    role_name: str = "validation_agent",
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 3,
    task_prompt_template: Optional[str] = None,
    **kwargs
) -> "BaseAgent":
    """
    创建验证重试Agent（带输出验证的循环调用）

    Args:
        validators: 验证器函数列表
        role_name: Agent角色名称
        system_prompt: 系统提示词（直接传递字符串）
        model_name: 模型名称
        max_retries: 最大重试次数
        task_prompt_template: 任务提示词模板名称（可选）
        **kwargs: 其他参数

    Returns:
        配置好的BaseAgent实例
    """
    from flowagent.core.execution_config import ExecutionConfig, ExecutionMode

    execution_config = ExecutionConfig(
        mode=ExecutionMode.VALIDATION_RETRY,
        model_name=model_name,
        max_retries=max_retries,
        validators=validators,
        **{k: v for k, v in kwargs.items() if k in ExecutionConfig.__dataclass_fields__}
    )

    # 动态创建Agent类
    DynamicAgentClass = _create_dynamic_agent_class(
        role_name=role_name,
        system_prompt=system_prompt or "你是一个智能助手。",
        task_prompt_template=task_prompt_template,
    )

    agent = DynamicAgentClass(
        model_name=model_name,
        execution_config=execution_config,
        **{k: v for k, v in kwargs.items() if k not in ExecutionConfig.__dataclass_fields__}
    )

    log.info(f"创建Validation Agent: {role_name}，验证器数量: {len(validators)}")
    return agent


def create_simple_agent(
    role_name: str = "simple_agent",
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    task_prompt_template: Optional[str] = None,
    **kwargs
) -> "BaseAgent":
    """
    创建简单Agent（单次LLM调用）

    Args:
        role_name: Agent角色名称
        system_prompt: 系统提示词（直接传递字符串）
        model_name: 模型名称
        task_prompt_template: 任务提示词模板名称（可选）
        **kwargs: 其他参数

    Returns:
        配置好的BaseAgent实例
    """
    from flowagent.core.execution_config import ExecutionConfig, ExecutionMode

    execution_config = ExecutionConfig(
        mode=ExecutionMode.SIMPLE,
        model_name=model_name,
        **{k: v for k, v in kwargs.items() if k in ExecutionConfig.__dataclass_fields__}
    )

    # 动态创建Agent类
    DynamicAgentClass = _create_dynamic_agent_class(
        role_name=role_name,
        system_prompt=system_prompt or "你是一个智能助手。",
        task_prompt_template=task_prompt_template,
    )

    agent = DynamicAgentClass(
        model_name=model_name,
        execution_config=execution_config,
        **{k: v for k, v in kwargs.items() if k not in ExecutionConfig.__dataclass_fields__}
    )

    log.info(f"创建Simple Agent: {role_name}")
    return agent


def create_vlm_agent(
    role_name: str = "vlm_agent",
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    vlm_mode: str = "understanding",
    image_detail: str = "auto",
    max_image_size: int = 2048,
    task_prompt_template: Optional[str] = None,
    **kwargs
) -> "BaseAgent":
    """
    创建VLM Agent（视觉语言模型）

    支持图像理解、生成和编辑等多种视觉任务。

    Args:
        role_name: Agent角色名称
        system_prompt: 系统提示词（直接传递字符串）
        model_name: 模型名称
        vlm_mode: VLM模式 ("understanding", "generation", "edit")
        image_detail: 图像细节级别 ("auto", "low", "high")
        max_image_size: 最大图像尺寸
        task_prompt_template: 任务提示词模板名称（可选）
        **kwargs: 其他参数，包括additional_params等

    Returns:
        配置好的BaseAgent实例

    Example:
        >>> agent = create_vlm_agent(
        ...     role_name="image_analyzer",
        ...     system_prompt="你是一个图像分析专家",
        ...     vlm_mode="understanding",
        ...     image_detail="high"
        ... )
        >>> result = await agent.execute(state)
    """
    from flowagent.core.execution_config import ExecutionConfig, ExecutionMode

    execution_config = ExecutionConfig(
        mode=ExecutionMode.VLM,
        model_name=model_name,
        vlm_mode=vlm_mode,
        image_detail=image_detail,
        max_image_size=max_image_size,
        **{k: v for k, v in kwargs.items() if k in ExecutionConfig.__dataclass_fields__}
    )

    # 动态创建Agent类
    DynamicAgentClass = _create_dynamic_agent_class(
        role_name=role_name,
        system_prompt=system_prompt or "你是一个视觉语言模型助手，可以理解和处理图像。",
        task_prompt_template=task_prompt_template,
    )

    agent = DynamicAgentClass(
        model_name=model_name,
        execution_config=execution_config,
        **{k: v for k, v in kwargs.items() if k not in ExecutionConfig.__dataclass_fields__}
    )

    log.info(f"创建VLM Agent: {role_name}，模式: {vlm_mode}")
    return agent


def create_parallel_agent(
    role_name: str = "parallel_agent",
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    concurrency_limit: int = 5,
    task_prompt_template: Optional[str] = None,
    **kwargs
) -> "BaseAgent":
    """
    创建并行Agent（批量并行处理）

    支持同时处理多个任务，适用于批量数据处理场景。

    Args:
        role_name: Agent角色名称
        system_prompt: 系统提示词（直接传递字符串）
        model_name: 模型名称
        concurrency_limit: 并发限制数量，默认5
        task_prompt_template: 任务提示词模板名称（可选）
        **kwargs: 其他参数

    Returns:
        配置好的BaseAgent实例

    Example:
        >>> agent = create_parallel_agent(
        ...     role_name="batch_processor",
        ...     system_prompt="你是一个批量处理专家",
        ...     concurrency_limit=10
        ... )
        >>> result = await agent.execute(state)
    """
    from flowagent.core.execution_config import ExecutionConfig, ExecutionMode

    execution_config = ExecutionConfig(
        mode=ExecutionMode.PARALLEL,
        model_name=model_name,
        concurrency_limit=concurrency_limit,
        **{k: v for k, v in kwargs.items() if k in ExecutionConfig.__dataclass_fields__}
    )

    # 动态创建Agent类
    DynamicAgentClass = _create_dynamic_agent_class(
        role_name=role_name,
        system_prompt=system_prompt or "你是一个智能助手，可以并行处理多个任务。",
        task_prompt_template=task_prompt_template,
    )

    agent = DynamicAgentClass(
        model_name=model_name,
        execution_config=execution_config,
        **{k: v for k, v in kwargs.items() if k not in ExecutionConfig.__dataclass_fields__}
    )

    log.info(f"创建Parallel Agent: {role_name}，并发限制: {concurrency_limit}")
    return agent
