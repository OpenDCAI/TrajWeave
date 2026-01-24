"""
DFA Trajectory (TRJ) - Workflow 执行轨迹导出模块

用于捕获、构建和导出 Workflow 执行过程数据，支持：
- ReAct 模式轨迹
- Workflow 模式轨迹
- 多模态数据
"""

from flowagent.trajectory.models import (
    TrajectoryStep,
    Trajectory,
    TrajectoryMode,
    StepRole,
    ActionType,
)
from flowagent.trajectory.collector import TrajectoryCollector
from flowagent.trajectory.builder import TrajectoryBuilder
from flowagent.trajectory.exporter import TrajectoryExporter
from flowagent.trajectory.manager import TrajectoryManager

__all__ = [
    # 数据模型
    "TrajectoryStep",
    "Trajectory",
    "TrajectoryMode",
    "StepRole",
    "ActionType",
    # 核心组件
    "TrajectoryCollector",
    "TrajectoryBuilder",
    "TrajectoryExporter",
    "TrajectoryManager",
]
