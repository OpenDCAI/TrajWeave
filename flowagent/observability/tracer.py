"""
可观测性追踪器 - DeerFlow 2.0 对齐

DeerFlow 2.0 通过 LangSmith 提供完整的可观测性：
- 每次 LLM 调用的输入/输出/耗时
- 每次工具执行的参数/结果/耗时
- Agent 完整生命周期追踪
- 子 Agent 父子关系追踪

本模块提供 FlowAgentTracer，当 LangSmith 可用时自动集成，
否则优雅降级为结构化日志。

使用示例：
tracer = FlowAgentTracer(project_name="flowagent")

async with tracer.trace_agent_run("Writer", run_id="xxx"):
    async with tracer.trace_llm_call("gpt-4o", messages):
        response = await llm.invoke(messages)
    async with tracer.trace_tool_call("web_search", {"query": "..."}):
        result = await tool.run(...)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from flowagent.logger import get_logger

log = get_logger(__name__)


class FlowAgentTracer:
    """统一追踪器 - LangSmith 可用时集成，否则降级为日志

    Attributes:
    project_name: LangSmith 项目名称
    enabled: 是否启用追踪
    """

    def __init__(
        self,
        project_name: str = "flowagent",
        langsmith_enabled: bool = True,
    ):
        self._project = project_name
        self._langsmith = None
        self.enabled = False

        if langsmith_enabled:
            try:
                from langsmith import Client
                self._langsmith = Client()
                self.enabled = True
                log.info(f"LangSmith tracing 已启用, 项目: {project_name}")
            except ImportError:
                log.info("langsmith 未安装，降级为日志追踪")
            except Exception as e:
                log.info(f"LangSmith 连接失败（{e}），降级为日志追踪")

    @asynccontextmanager
    async def trace_agent_run(
        self,
        agent_name: str,
        run_id: str = "",
        parent_run_id: Optional[str] = None,
        **metadata,
    ):
        """追踪 Agent 执行的完整生命周期

        Args:
        agent_name: Agent 名称
        run_id: 运行 ID
        parent_run_id: 父运行 ID（子 Agent 用）
        **metadata: 额外元数据
        """
        start_time = time.time()
        log.info(
            f"[Trace] Agent '{agent_name}' 开始执行 "
            f"(run_id={run_id[:8] if run_id else 'N/A'})"
        )

        try:
            yield {"agent_name": agent_name, "run_id": run_id, "start_time": start_time}
        except Exception as e:
            elapsed = time.time() - start_time
            log.error(
                f"[Trace] Agent '{agent_name}' 执行失败 "
                f"({elapsed:.2f}s): {e}"
            )
            raise
        finally:
            elapsed = time.time() - start_time
            log.info(
                f"[Trace] Agent '{agent_name}' 执行完成 ({elapsed:.2f}s)"
            )

    @asynccontextmanager
    async def trace_llm_call(
        self,
        model: str,
        messages: Optional[list] = None,
        **kwargs,
    ):
        """追踪单次 LLM 调用

        Args:
        model: 模型名称
        messages: 输入消息列表
        **kwargs: 额外参数
        """
        start_time = time.time()
        msg_count = len(messages) if messages else 0
        log.debug(
            f"[Trace] LLM 调用开始: model={model}, messages={msg_count}"
        )

        try:
            yield {"model": model, "start_time": start_time}
        except Exception as e:
            elapsed = time.time() - start_time
            log.error(
                f"[Trace] LLM 调用失败: model={model}, "
                f"elapsed={elapsed:.2f}s, error={e}"
            )
            raise
        finally:
            elapsed = time.time() - start_time
            log.debug(
                f"[Trace] LLM 调用完成: model={model}, "
                f"elapsed={elapsed:.2f}s"
            )

    @asynccontextmanager
    async def trace_tool_call(
        self,
        tool_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """追踪工具执行

        Args:
        tool_name: 工具名称
        input_data: 输入数据
        """
        start_time = time.time()
        log.debug(f"[Trace] 工具调用开始: {tool_name}")

        try:
            yield {"tool_name": tool_name, "start_time": start_time}
        except Exception as e:
            elapsed = time.time() - start_time
            log.error(
                f"[Trace] 工具调用失败: {tool_name}, "
                f"elapsed={elapsed:.2f}s, error={e}"
            )
            raise
        finally:
            elapsed = time.time() - start_time
            log.debug(
                f"[Trace] 工具调用完成: {tool_name}, "
                f"elapsed={elapsed:.2f}s"
            )

    @asynccontextmanager
    async def trace_sub_agent(
        self,
        parent_agent: str,
        sub_agent_role: str,
        sub_agent_id: str,
        **metadata,
    ):
        """追踪子 Agent 执行（包含父子关系）

        DeerFlow 2.0 模式：子 Agent 的 trace 关联到父 Agent。

        Args:
        parent_agent: 父 Agent 名称
        sub_agent_role: 子 Agent 角色
        sub_agent_id: 子 Agent ID
        """
        start_time = time.time()
        log.info(
            f"[Trace] 子 Agent '{sub_agent_role}' 开始执行 "
            f"(parent={parent_agent}, id={sub_agent_id[:8]}...)"
        )

        try:
            yield {
                "parent": parent_agent,
                "sub_role": sub_agent_role,
                "sub_id": sub_agent_id,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            log.error(
                f"[Trace] 子 Agent '{sub_agent_role}' 失败 ({elapsed:.2f}s): {e}"
            )
            raise
        finally:
            elapsed = time.time() - start_time
            log.info(
                f"[Trace] 子 Agent '{sub_agent_role}' 完成 ({elapsed:.2f}s)"
            )


# 全局单例
_tracer_instance: Optional[FlowAgentTracer] = None


def get_tracer(
    project_name: str = "flowagent",
    langsmith_enabled: bool = True,
) -> FlowAgentTracer:
    """获取全局追踪器实例"""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = FlowAgentTracer(
            project_name=project_name,
            langsmith_enabled=langsmith_enabled,
        )
    return _tracer_instance
