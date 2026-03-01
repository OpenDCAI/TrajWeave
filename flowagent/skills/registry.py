"""
Skill注册表
"""
from __future__ import annotations

from typing import Dict, List, Optional
import yaml

from flowagent.logger import get_logger
from flowagent.skills.base import Skill

log = get_logger(__name__)


class SkillRegistry:
    """
    Skill注册表 - 管理所有已注册的Skills
    """

    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """注册Skill"""
        if skill.name in self._skills:
            log.warning(f"Skill '{skill.name}' 已存在，将被覆盖")
        self._skills[skill.name] = skill
        log.info(f"注册Skill: {skill.name}")

    def get(self, name: str) -> Optional[Skill]:
        """获取Skill"""
        return self._skills.get(name)

    def list_skills(self) -> List[str]:
        """列出所有Skill名称"""
        return list(self._skills.keys())

    def all(self) -> Dict[str, Skill]:
        """获取所有Skills"""
        return self._skills.copy()

    def unregister(self, name: str) -> bool:
        """注销Skill"""
        if name in self._skills:
            del self._skills[name]
            log.info(f"注销Skill: {name}")
            return True
        return False

    def load_from_yaml(self, path: str) -> Skill:
        """从YAML文件加载Skill"""
        import importlib
        from langchain_core.tools import tool

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 加载工具函数
        tools = []
        for tool_config in config.get("tools", []):
            func_path = tool_config.get("function", "")
            if func_path:
                module_path, func_name = func_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                tools.append(func)

        skill = Skill(
            name=config.get("name", ""),
            description=config.get("description", ""),
            tools=tools,
            system_prompt=config.get("system_prompt", ""),
            user_prompt_template=config.get("user_prompt_template", "{input}"),
            metadata=config.get("metadata", {}),
            execution_mode=config.get("execution_mode", "simple"),
            model_name=config.get("model_name"),
            steps=config.get("steps", []),
        )

        self.register(skill)
        log.info(f"从YAML加载Skill: {path}")
        return skill

    def load_from_markdown(self, path: str) -> Skill:
        """从 Markdown 文件加载 Skill（YAML frontmatter + body sections）。

        格式::

            ---
            name: my_skill
            description: ...
            execution_mode: react
            model_name: gpt-4o
            tools:
              - function: module.path.func_name
            ---
            # System Prompt
            你是...

            # User Prompt Template
            请处理：{input}
        """
        import importlib

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # 解析 YAML frontmatter
        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid skill.md format (missing --- delimiters): {path}")

        config = yaml.safe_load(parts[1]) or {}
        body = parts[2]

        # 从 body 中提取 # 标题分段
        sections: dict[str, str] = {}
        current_section = None
        current_lines: list[str] = []
        for line in body.strip().split("\n"):
            if line.startswith("# "):
                if current_section:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = line[2:].strip().lower().replace(" ", "_")
                current_lines = []
            else:
                current_lines.append(line)
        if current_section:
            sections[current_section] = "\n".join(current_lines).strip()

        # 加载工具
        tools = []
        for tool_config in config.get("tools", []):
            func_path = tool_config.get("function", "") if isinstance(tool_config, dict) else str(tool_config)
            if func_path:
                module_path, func_name = func_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                tools.append(getattr(module, func_name))

        skill = Skill(
            name=config.get("name", ""),
            description=config.get("description", ""),
            tools=tools,
            system_prompt=sections.get("system_prompt", config.get("system_prompt", "")),
            user_prompt_template=sections.get("user_prompt_template", config.get("user_prompt_template", "{input}")),
            metadata=config.get("metadata", {}),
            execution_mode=config.get("execution_mode", "simple"),
            model_name=config.get("model_name"),
            steps=config.get("steps", []),
        )

        self.register(skill)
        log.info(f"从Markdown加载Skill: {path}")
        return skill


# 全局单例
_skill_registry_instance = None


def get_skill_registry() -> SkillRegistry:
    """获取全局Skill注册表"""
    global _skill_registry_instance
    if _skill_registry_instance is None:
        _skill_registry_instance = SkillRegistry()
    return _skill_registry_instance
