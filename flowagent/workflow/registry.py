"""Workflow Registry

提供按名字注册/获取 workflow 工厂的能力，让 workflow 也能像 Agent 一样被复用与按名运行。

约定：registry 中存储的是 *工厂函数*，工厂返回：
- `Workflow`（已编译可运行），或
- `WorkflowBuilder`（将自动 compile 为 Workflow）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from flowagent.workflow.base import Workflow, WorkflowBuilder


WorkflowFactory = Callable[..., Any]


class WorkflowRegistry:
    """Workflow 注册表（最小实现）。"""

    _workflows: Dict[str, WorkflowFactory] = {}

    @classmethod
    def register(cls, name: str, factory: WorkflowFactory) -> None:
        if name in cls._workflows:
            if cls._workflows[name] is factory:
                return
            raise ValueError(
                f"Workflow '{name}' already registered by {cls._workflows[name]} (now trying {factory})"
            )
        cls._workflows[name] = factory

    @classmethod
    def get(cls, name: str) -> WorkflowFactory:
        try:
            return cls._workflows[name]
        except KeyError:
            raise KeyError(f"Workflow '{name}' 未注册，可选: {list(cls._workflows)}")

    @classmethod
    def all(cls) -> Dict[str, WorkflowFactory]:
        return dict(cls._workflows)

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> Workflow:
        """按名字创建（并确保返回 Workflow）。"""
        factory = cls.get(name)
        obj = factory(*args, **kwargs)

        if isinstance(obj, Workflow):
            return obj
        if isinstance(obj, WorkflowBuilder):
            return obj.compile()
        raise TypeError(
            "Workflow factory must return Workflow or WorkflowBuilder; "
            f"got {type(obj).__name__} from {factory}"
        )


def register_workflow(name: str) -> Callable[[WorkflowFactory], WorkflowFactory]:
    """装饰器：@register_workflow('demo')"""

    def _decorator(factory: WorkflowFactory) -> WorkflowFactory:
        WorkflowRegistry.register(name, factory)
        return factory

    return _decorator


# Backward-compatible aliases
RuntimeRegistry = WorkflowRegistry
register = register_workflow