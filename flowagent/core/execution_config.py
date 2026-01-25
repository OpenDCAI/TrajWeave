"""
执行配置定义

定义各种执行策略所需的配置参数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ExecutionMode(str, Enum):
    """执行模式枚举"""
    SIMPLE = "simple"
    VALIDATION_RETRY = "validation_retry"
    REACT = "react"
    VLM = "vlm"
    PARALLEL = "parallel"
    PLAN_SOLVE = "plan_solve"
    PLAN_EXECUTE = "plan_execute"


@dataclass
class ExecutionConfig:
    """执行策略配置"""

    # 基础配置
    mode: ExecutionMode = ExecutionMode.SIMPLE
    model_name: Optional[str] = None
    chat_api_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096

    # ValidationRetry模式配置
    max_retries: int = 3
    validators: List[Callable] = field(default_factory=list)

    # React模式配置（工具调用）
    tool_mode: str = "auto"  # auto, required, none

    # VLM模式配置
    vlm_mode: str = "understanding"  # understanding, generation, edit
    image_detail: str = "auto"
    max_image_size: int = 2048
    additional_params: Dict[str, Any] = field(default_factory=dict)

    # Parallel模式配置
    concurrency_limit: int = 5

    # Plan模式通用配置
    planner_model: Optional[str] = None
    planner_temperature: float = 0.7
    executor_model: Optional[str] = None
    executor_temperature: float = 0.3
    executor_tools: List[Any] = field(default_factory=list)
    max_plan_steps: int = 10
    require_plan_approval: bool = False
    continue_on_error: bool = False

    # PlanExecute模式特有配置
    max_replanning_rounds: int = 3
    replanner_model: Optional[str] = None
    replanner_temperature: float = 0.5
    auto_replan_on_error: bool = True
    interrupt_before_step: bool = False
    interrupt_after_step: bool = False
    interrupt_on_replan: bool = False
