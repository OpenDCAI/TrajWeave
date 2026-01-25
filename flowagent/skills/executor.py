"""
Skill执行器
"""
from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from flowagent.logger import get_logger
from flowagent.skills.base import Skill

if TYPE_CHECKING:
    from flowagent.core.base_agent import BaseAgent

log = get_logger(__name__)


class SkillExecutor:
    """
    Skill执行器 - 执行Skill并管理生命周期
    """

    async def execute(
        self,
        skill: Skill,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行Skill

        Args:
            skill: 要执行的Skill
            input_data: 输入数据

        Returns:
            执行结果
        """
        log.info(f"执行Skill: {skill.name}")

        # 运行前置钩子
        context = await skill.run_pre_hooks(input_data.copy())

        # 格式化提示词
        user_prompt = skill.format_user_prompt(**context)

        result = {
            "skill_name": skill.name,
            "user_prompt": user_prompt,
            "system_prompt": skill.get_system_prompt(),
            "tools": skill.get_tools(),
            "context": context
        }

        # 运行后置钩子
        result = await skill.run_post_hooks(result, context)

        return result
