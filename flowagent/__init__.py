"""
FlowAgent - A flexible LLM agent framework

FlowAgent provides a clean, extensible framework for building LLM-powered agents
with support for multiple execution modes, tool integration, and graph-based workflows.
"""

# Load optional project-level environment variables from `.env`.
# This is dependency-free and safe: it only reads from the current project root.
try:
    from flowagent.env import load_project_env

    load_project_env(override=False)
except Exception:
    # Never fail import due to env loading issues.
    pass

__version__ = "1.0.0"

# Core exports - most commonly used
from flowagent.core.base_agent import BaseAgent
from flowagent.core.registry import AgentRegistry, register
from flowagent.core.configs import (
    BaseAgentConfig,
    SimpleConfig,
    ReactConfig,
    GraphConfig,
    VLMConfig,
    ParallelConfig,
    ExecutionMode,
)

from flowagent.state.base import MainState, MainRequest
from flowagent.llm.base import BaseLLMCaller
from flowagent.parsers.parsers import BaseParser
from flowagent.tools.manager import ToolManager
from flowagent.logger import get_logger

from flowagent.workflow.registry import WorkflowRegistry, register_workflow
from flowagent.core.factory import (
    create_react_agent,
    create_plan_execute_agent,
    create_validation_agent,
    create_simple_agent,
    create_vlm_agent,
    create_parallel_agent,
)
from flowagent.skills import Skill, SkillRegistry, get_skill_registry, SkillExecutor

__all__ = [
    # Core
    "BaseAgent",
    "AgentRegistry",
    "register",
    # Configs
    "BaseAgentConfig",
    "SimpleConfig",
    "ReactConfig",
    "GraphConfig",
    "VLMConfig",
    "ParallelConfig",
    "ExecutionMode",
    # State
    "MainState",
    "MainRequest",
    # Infrastructure
    "BaseLLMCaller",
    "BaseParser",
    "ToolManager",
    "get_logger",
    # Workflow
    "WorkflowRegistry",
    "register_workflow",
    # Factory
    "create_react_agent",
    "create_plan_execute_agent",
    "create_validation_agent",
    "create_simple_agent",
    "create_vlm_agent",
    "create_parallel_agent",
    # Skills
    "Skill",
    "SkillRegistry",
    "get_skill_registry",
    "SkillExecutor",
]
