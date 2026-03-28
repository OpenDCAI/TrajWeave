"""
子 Agent 管理池 - DeerFlow 2.0 对齐

管理多个子 Agent 的并发执行，核心能力：
- spawn: 异步启动单个子 Agent
- spawn_many: 并发启动多个子 Agent，等待全部完成
- collect_results: 收集已完成子 Agent 的结果
- terminate: 终止指定子 Agent

借鉴 DeerFlow 2.0 的设计：
- 使用 asyncio.Semaphore 控制最大并发数
- 每个子 Agent 独立 context、tools 和超时
- 结果自动合并到 parent_state.sub_agent_results

使用示例：
manager = SubAgentManager(tool_manager, max_concurrent=3)

# 方式1: 并发启动多个子 Agent
configs = [
SubAgentConfig(role="researcher", system_prompt="...", context="研究A"),
SubAgentConfig(role="coder", system_prompt="...", context="实现B"),
SubAgentConfig(role="reviewer", system_prompt="...", context="审查C"),
]
results = await manager.spawn_many(configs, parent_state)

# 方式2: 异步启动单个子 Agent
agent_id = await manager.spawn(config, parent_state)
# ... 做其他事情 ...
result = await manager.get_result(agent_id)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from flowagent.core.sub_agent import SubAgent, SubAgentConfig
from flowagent.logger import get_logger

if TYPE_CHECKING:
    from flowagent.state.base import MainState
    from flowagent.tools.manager import ToolManager

log = get_logger(__name__)


class SubAgentManager:
    """子 Agent 管理器 - 编排多个子 Agent 的并发生命周期

    DeerFlow 2.0 模式：主 Agent 通过 SubAgentManager 动态创建
    和管理多个专门化的子 Agent，每个子 Agent 独立运行。

    Attributes:
    max_concurrent: 最大并发子 Agent 数量
    """

    def __init__(
        self,
        tool_manager: "ToolManager",
        max_concurrent: int = 5,
    ):
        self._tool_manager = tool_manager
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active: Dict[str, SubAgent] = {}
        self._completed: Dict[str, Dict[str, Any]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self.max_concurrent = max_concurrent

    @property
    def active_count(self) -> int:
        """当前活跃子 Agent 数量"""
        return len(self._active)

    @property
    def completed_count(self) -> int:
        """已完成子 Agent 数量"""
        return len(self._completed)

    async def spawn(
        self,
        config: SubAgentConfig,
        parent_state: "MainState",
    ) -> str:
        """异步启动单个子 Agent，立即返回其 ID

        子 Agent 在后台执行，可通过 get_result() 获取结果。

        Args:
        config: 子 Agent 配置
        parent_state: 父 Agent 的 state（用于创建隔离副本）

        Returns:
        子 Agent 的唯一 ID
        """
        agent = SubAgent(config, parent_state, self._tool_manager)
        self._active[agent.id] = agent

        task = asyncio.create_task(
            self._run_with_semaphore(agent),
            name=f"sub_agent_{config.role}_{agent.id[:8]}",
        )
        self._tasks[agent.id] = task

        log.info(
            f"[SubAgentManager] 启动子 Agent: role={config.role}, "
            f"id={agent.id[:8]}..., active={self.active_count}"
        )
        return agent.id

    async def spawn_many(
        self,
        configs: List[SubAgentConfig],
        parent_state: "MainState",
    ) -> List[Dict[str, Any]]:
        """并发启动多个子 Agent，等待全部完成后返回结果

        这是最常用的模式 — 将复杂任务分解为多个子 Agent 并行处理。

        Args:
        configs: 子 Agent 配置列表
        parent_state: 父 Agent 的 state

        Returns:
        所有子 Agent 的执行结果列表（顺序与 configs 对应）
        """
        if not configs:
            return []

        log.info(
            f"[SubAgentManager] 并发启动 {len(configs)} 个子 Agent: "
            f"{[c.role for c in configs]}"
        )

        agents = [
            SubAgent(c, parent_state, self._tool_manager) for c in configs
        ]
        for agent in agents:
            self._active[agent.id] = agent

        tasks = [self._run_with_semaphore(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常情况
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "agent_id": agents[i].id,
                    "role": configs[i].role,
                    "result": None,
                    "status": "failed",
                    "error": str(result),
                })
                log.error(
                    f"[SubAgentManager] 子 Agent '{configs[i].role}' "
                    f"异常: {result}"
                )
            else:
                processed_results.append(result)

        # 自动合并结果到 parent_state
        for result in processed_results:
            parent_state.sub_agent_results[result["role"]] = result

        log.info(
            f"[SubAgentManager] 全部 {len(configs)} 个子 Agent 执行完毕, "
            f"成功: {sum(1 for r in processed_results if r['status'] == 'completed')}, "
            f"失败: {sum(1 for r in processed_results if r['status'] != 'completed')}"
        )

        return processed_results

    async def get_result(
        self,
        agent_id: str,
        timeout: float = 300.0,
    ) -> Optional[Dict[str, Any]]:
        """等待并获取指定子 Agent 的结果

        Args:
        agent_id: 子 Agent ID
        timeout: 等待超时（秒）

        Returns:
        子 Agent 的执行结果，如果超时返回 None
        """
        if agent_id in self._completed:
            return self._completed[agent_id]

        task = self._tasks.get(agent_id)
        if task is None:
            log.warning(f"[SubAgentManager] 子 Agent {agent_id[:8]}... 不存在")
            return None

        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            log.warning(
                f"[SubAgentManager] 等待子 Agent {agent_id[:8]}... 超时"
            )
            return None

    async def terminate(self, agent_id: str) -> bool:
        """终止指定子 Agent

        Args:
        agent_id: 子 Agent ID

        Returns:
        是否成功终止
        """
        task = self._tasks.get(agent_id)
        if task and not task.done():
            task.cancel()
            self._active.pop(agent_id, None)
            self._tasks.pop(agent_id, None)
            log.info(f"[SubAgentManager] 终止子 Agent {agent_id[:8]}...")
            return True
        return False

    async def terminate_all(self) -> int:
        """终止所有活跃的子 Agent

        Returns:
        终止的数量
        """
        count = 0
        for agent_id in list(self._tasks.keys()):
            if await self.terminate(agent_id):
                count += 1
        log.info(f"[SubAgentManager] 终止了 {count} 个子 Agent")
        return count

    def collect_results(self) -> Dict[str, Dict[str, Any]]:
        """收集所有已完成子 Agent 的结果

        Returns:
        {role: result_dict} 的字典
        """
        return dict(self._completed)

    def get_status_summary(self) -> Dict[str, Any]:
        """获取所有子 Agent 的状态摘要"""
        statuses = {}
        for agent_id, agent in self._active.items():
            statuses[agent_id] = {
                "role": agent.config.role,
                "status": agent.status,
            }
        return {
            "active": self.active_count,
            "completed": self.completed_count,
            "agents": statuses,
        }

    async def _run_with_semaphore(self, agent: SubAgent) -> Dict[str, Any]:
        """在信号量控制下执行子 Agent"""
        async with self._semaphore:
            result = await agent.run()
            # 完成后从 active 移到 completed
            self._active.pop(agent.id, None)
            self._completed[agent.id] = result
            return result
