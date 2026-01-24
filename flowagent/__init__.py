"""
FlowAgent - A flexible LLM agent framework

FlowAgent provides a clean, extensible framework for building LLM-powered agents
with support for multiple execution modes, tool integration, and graph-based workflows.
"""

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
]
