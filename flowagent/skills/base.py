"""
Skill基类定义
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
from langchain_core.tools import Tool

from flowagent.logger import get_logger

log = get_logger(__name__)


@dataclass
class Skill:
    """
    Skill - 可复用的能力包

    包含工具集、提示词模板和执行钩子，类似Claude Code的Skills概念。

    Attributes:
        name: 唯一标识
        description: 描述
        tools: 包含的工具集
        system_prompt: 系统提示词
        user_prompt_template: 用户提示词模板
        output_schema: 输出格式（Pydantic模型）
        pre_hooks: 执行前钩子
        post_hooks: 执行后钩子
    """
    name: str
    description: str
    tools: List[Tool] = field(default_factory=list)
    system_prompt: str = ""
    user_prompt_template: str = "{input}"
    output_schema: Optional[Type] = None
    pre_hooks: List[Callable] = field(default_factory=list)
    post_hooks: List[Callable] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 执行配置（Stage 3 新增）
    execution_mode: str = "simple"  # 对应 StrategyFactory 的 mode key
    model_name: Optional[str] = None
    # 多步 Skill 的工作流定义，每项: {"name": str, "system_prompt": str, "tools": [...], ...}
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def get_tools(self) -> List[Tool]:
        """获取工具列表"""
        return self.tools

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return self.system_prompt

    def format_user_prompt(self, **kwargs) -> str:
        """格式化用户提示词"""
        return self.user_prompt_template.format(**kwargs)

    async def run_pre_hooks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行执行前钩子"""
        for hook in self.pre_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    context = await hook(context) or context
                else:
                    context = hook(context) or context
            except Exception as e:
                log.error(f"Skill {self.name} pre_hook 执行失败: {e}")
        return context

    async def run_post_hooks(self, result: Any, context: Dict[str, Any]) -> Any:
        """运行执行后钩子"""
        for hook in self.post_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    result = await hook(result, context) or result
                else:
                    result = hook(result, context) or result
            except Exception as e:
                log.error(f"Skill {self.name} post_hook 执行失败: {e}")
        return result


import asyncio
