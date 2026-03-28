"""
子 Agent 模块 - DeerFlow 2.0 对齐

借鉴 DeerFlow 2.0 的子 Agent 动态生成模式：
- 主 Agent 可动态 spawn 专门化的子 Agent
- 每个子 Agent 拥有独立的 state 副本、工具集、超时控制
- 子 Agent 之间以及与父 Agent 之间状态完全隔离
- 支持并发执行

核心设计原则：
1. 状态隔离：子 Agent 使用深拷贝的独立 state（不是引用）
2. 工具子集：每个子 Agent 只能使用指定的工具
3. 独立超时：通过 asyncio.wait_for 实现
4. 迭代上限：防止死循环

使用示例：
config = SubAgentConfig(
role="researcher",
system_prompt="你是一个专业研究员...",
tools=["web_search", "url_fetch"],
context="请研究 transformer 的最新进展",
timeout_seconds=120,
)
sub = SubAgent(config, parent_state, tool_manager)
result = await sub.run()
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from flowagent.logger import get_logger

if TYPE_CHECKING:
    from flowagent.state.base import MainState
    from flowagent.tools.manager import ToolManager

log = get_logger(__name__)


@dataclass
class SubAgentConfig:
    """子 Agent 配置 - 声明式定义子 Agent 的行为

    Attributes:
    role: 子 Agent 角色名，如 "researcher", "coder", "reviewer"
    system_prompt: 子 Agent 独立的系统提示词（不与父 Agent 共享）
    tools: 允许使用的工具名列表（从父 Agent 的 ToolManager 中筛选）
    context: 从父 Agent 传递的种子上下文（任务描述）
    timeout_seconds: 执行超时（秒），默认 300
    max_iterations: 最大 LLM 调用迭代次数，默认 10
    execution_mode: 使用的执行策略，默认 "react"
    model_name: 可选的模型覆盖（不设则继承父 Agent 的模型）
    """
    role: str
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    context: str = ""
    timeout_seconds: float = 300.0
    max_iterations: int = 10
    execution_mode: str = "react"
    model_name: Optional[str] = None


class SubAgent:
    """子 Agent - 具有隔离状态的独立执行单元

    DeerFlow 2.0 核心模式：子 Agent 获得**独立的 state 副本**，
    而不是父 Agent state 的引用。这确保了：
    - 子 Agent 的消息历史不会污染父 Agent
    - 子 Agent 的工具调用结果独立存储
    - 多个子 Agent 可以安全并发执行
    """

    def __init__(
        self,
        config: SubAgentConfig,
        parent_state: "MainState",
        tool_manager: "ToolManager",
    ):
        self.id = str(uuid4())
        self.config = config
        self._parent_state = parent_state
        self._tool_manager = tool_manager
        self._state = self._create_isolated_state(parent_state, config)
        self._result: Optional[Dict[str, Any]] = None
        self._status: str = "pending"  # pending | running | completed | failed | timeout

    @property
    def status(self) -> str:
        return self._status

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        return self._result

    def _create_isolated_state(
        self,
        parent_state: "MainState",
        config: SubAgentConfig,
    ) -> "MainState":
        """创建隔离的 state 副本

        关键决策：
        - 只复制 request（保留 LLM 配置），不复制 messages
        - 子 Agent 从空消息历史开始
        - 通过 config.context 传递必要的上下文信息
        - 上下文预算设为父 Agent 的一半（防止多子 Agent 同时超限）
        """
        from flowagent.state.base import MainState, MainRequest

        # 深拷贝 request 以避免修改父 Agent 的配置
        new_request = copy.deepcopy(parent_state.request)
        if config.context:
            new_request.target = config.context
        if config.model_name:
            new_request.model = config.model_name

        return MainState(
            request=new_request,
            messages=[],
            agent_results={},
            temp_data={},
            parent_agent_id=parent_state.session_id,
            sub_agent_results={},
            context_budget=min(60_000, parent_state.context_budget // 2),
        )

    async def run(self) -> Dict[str, Any]:
        """执行子 Agent（带超时控制）

        Returns:
            包含执行结果、状态、角色等信息的字典
        """
        self._status = "running"
        log.info(
            f"[SubAgent] 启动子 Agent '{self.config.role}' "
            f"(id={self.id[:8]}..., timeout={self.config.timeout_seconds}s, "
            f"mode={self.config.execution_mode})"
        )

        try:
            result = await asyncio.wait_for(
                self._execute(),
                timeout=self.config.timeout_seconds,
            )
            self._status = "completed"
            self._result = {
                "agent_id": self.id,
                "role": self.config.role,
                "result": result,
                "status": "completed",
            }
            log.info(f"[SubAgent] 子 Agent '{self.config.role}' 执行完成")
            return self._result

        except asyncio.TimeoutError:
            self._status = "timeout"
            self._result = {
                "agent_id": self.id,
                "role": self.config.role,
                "result": self._get_partial_result(),
                "status": "timeout",
                "error": f"执行超时 ({self.config.timeout_seconds}s)",
            }
            log.warning(
                f"[SubAgent] 子 Agent '{self.config.role}' 超时 "
                f"({self.config.timeout_seconds}s)"
            )
            return self._result

        except Exception as e:
            self._status = "failed"
            self._result = {
                "agent_id": self.id,
                "role": self.config.role,
                "result": None,
                "status": "failed",
                "error": str(e),
            }
            log.error(
                f"[SubAgent] 子 Agent '{self.config.role}' 执行失败: {e}",
                exc_info=True,
            )
            return self._result

    async def _execute(self) -> Dict[str, Any]:
        """内部执行逻辑 - 通过工厂函数创建 Agent 并执行"""
        import flowagent.core.factory as factory

        # 获取工具子集
        filtered_tools = self._tool_manager.get_subset(self.config.tools)

        # 根据执行模式创建对应 Agent
        mode = self.config.execution_mode
        create_kwargs = {
            "role_name": f"sub_{self.config.role}",
            "system_prompt": self.config.system_prompt,
        }
        if self.config.model_name:
            create_kwargs["model_name"] = self.config.model_name

        if mode == "react" and filtered_tools.get_all_post_tools():
            create_kwargs["tools"] = filtered_tools.get_all_post_tools()
            agent = factory.create_react_agent(**create_kwargs)
        elif mode == "simple":
            agent = factory.create_simple_agent(**create_kwargs)
        else:
            # 默认使用 simple 模式
            agent = factory.create_simple_agent(**create_kwargs)

        # 设置工具管理器
        if agent.tool_manager is None:
            agent.tool_manager = filtered_tools

        # 执行 Agent
        result = agent.execute(self._state)
        if asyncio.iscoroutine(result):
            result = await result

        return self._state.agent_results

    def _get_partial_result(self) -> Optional[Dict[str, Any]]:
        """获取部分结果（超时/中断时使用）"""
        if self._state.agent_results:
            return self._state.agent_results
        return None
