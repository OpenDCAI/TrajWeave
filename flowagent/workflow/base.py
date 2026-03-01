
"""Workflow 编排基础设施（Phase 0）

目标：提供一个最小可用的 workflow 装配层：
- 节点（node）：任意可调用函数（后续会适配 BaseAgent.execute）
- 边（edge）：节点之间的执行顺序/条件路由
- 运行（run）：对 StateGraph.compile() 的轻量封装

说明：
- 目前只做骨架与最小 API，不引入配置化（YAML/JSON）
- state_model 既支持 Pydantic BaseModel，也支持你们现有的 dataclass State（BaseAgent 内部已有 StateGraph(type(state)) 用法）
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from langgraph.graph import StateGraph


NodeCallable = Callable[[Any], Any]
AsyncNodeCallable = Callable[[Any], Awaitable[Any]]


def _state_get(state: Any, key: str, default: Any = None) -> Any:
	if isinstance(state, dict):
		return state.get(key, default)
	return getattr(state, key, default)


def _state_set(state: Any, key: str, value: Any) -> None:
	if isinstance(state, dict):
		state[key] = value
	else:
		setattr(state, key, value)


def _ensure_mapping(state: Any, key: str) -> Dict[str, Any]:
	value = _state_get(state, key, None)
	if isinstance(value, dict):
		return value
	new_value: Dict[str, Any] = {}
	_state_set(state, key, new_value)
	return new_value


def agent_node(agent: Any, *, name: Optional[str] = None) -> AsyncNodeCallable:
	"""将 Agent 适配为 LangGraph 节点。

	约定：agent 需要具备
	- role_name: str
	- execute(state, **kwargs) -> awaitable（返回 state 或更新 state）

	返回：一个 async 节点函数，返回 dict 更新，保证 state.agent_results 可见。
	"""

	role = name or getattr(agent, "role_name", None) or getattr(agent, "name", None) or "agent"
	role_key = str(role)

	def _iter_post_tools(tool_manager: Any, target_role: str) -> Iterable[Any]:
		try:
			return list(tool_manager.get_post_tools(target_role))
		except Exception:
			return []

	def _merge_tool_managers(dst: Any, src: Any, target_role: str) -> None:
		"""将 src 的全局/角色后置工具尽力合并到 dst。"""
		try:
			# 全局后置工具
			for t in getattr(src, "global_post_tools", []) or []:
				dst.register_post_tool(t, role=None)
		except Exception:
			pass

		try:
			for t in _iter_post_tools(src, target_role):
				dst.register_post_tool(t, role=target_role)
		except Exception:
			pass

	def _inject_workflow_tools(state: Any) -> None:
		"""从 state.temp_data 注入 workflow 级 ToolManager/工具绑定到 agent。"""
		temp_data = _state_get(state, "temp_data", None)
		if not isinstance(temp_data, dict):
			return

		wf_tool_manager = temp_data.get("_workflow_tool_manager")
		wf_bindings = temp_data.get("_workflow_tool_bindings")
		if wf_tool_manager is None and not isinstance(wf_bindings, dict):
			return

		# 1) 统一 tool_manager 注入路径：workflow 优先（避免各处 new ToolManager 打架）
		if wf_tool_manager is not None:
			try:
				current_tm = getattr(agent, "tool_manager", None)
				if current_tm is None:
					setattr(agent, "tool_manager", wf_tool_manager)
				elif current_tm is not wf_tool_manager:
					# 合并已有工具到 workflow tool_manager，再切换引用
					_merge_tool_managers(wf_tool_manager, current_tm, role_key)
					setattr(agent, "tool_manager", wf_tool_manager)
			except Exception:
				pass

		# 2) 注入 workflow.bind_tools 声明的工具
		if isinstance(wf_bindings, dict):
			try:
				tools = wf_bindings.get(role_key) or []
				agent_tm = getattr(agent, "tool_manager", None)
				if agent_tm is not None:
					for t in tools:
						agent_tm.register_post_tool(t, role=role_key)
			except Exception:
				pass

	async def _node(state: Any):
		agent_results = _ensure_mapping(state, "agent_results")

		# 将上游节点结果追加到 request.target，让下游 agent prompt 自动感知
		if agent_results:
			import json
			req = _state_get(state, "request", None)
			if req is not None:
				orig_target = getattr(req, "target", "") or ""
				upstream_str = json.dumps(
					{k: v.get("results") if isinstance(v, dict) else v for k, v in agent_results.items()},
					ensure_ascii=False,
				)
				req.target = f"{orig_target}\n\n上游节点输出：{upstream_str}"

		try:
			_inject_workflow_tools(state)
			res = agent.execute(state)
			if asyncio.iscoroutine(res):
				res = await res

			agent_results = _ensure_mapping(state, "agent_results")
			if role_key not in agent_results and getattr(agent, "role_name", None):
				agent_results[role_key] = {"results": res}

			entry = agent_results.get(role_key)
			if isinstance(entry, dict):
				if "results" not in entry:
					agent_results[role_key] = {"status": "ok", "results": entry}
				else:
					entry.setdefault("status", "ok")
					entry.setdefault("error", None)
			else:
				agent_results[role_key] = {"status": "ok", "results": entry, "error": None}

			return {"agent_results": agent_results}

		except Exception as e:
			err = str(e)
			agent_results[role_key] = {
				"status": "error",
				"error": err,
				"results": {"error": err},
			}
			return {"agent_results": agent_results}

	return _node


@dataclass
class Workflow:
	"""已编译的工作流。"""

	name: str
	state_model: type
	compiled_graph: Any
	tool_manager: Any = None
	tool_bindings: Dict[str, List[Any]] = field(default_factory=dict)

	async def run_async(self, state: Any, **kwargs: Any) -> Any:
		"""异步运行工作流。"""
		if kwargs:
			# 允许把额外参数塞到 state.temp_data（若存在）作为上下文
			try:
				temp_data = getattr(state, "temp_data", None)
				if temp_data is None:
					setattr(state, "temp_data", {})
					temp_data = state.temp_data
				if isinstance(temp_data, dict):
					temp_data.update(kwargs)
			except Exception:
				pass

		# 注入 workflow 级上下文（工具绑定等），供 agent_node 在执行前读取。
		try:
			temp_data = getattr(state, "temp_data", None)
			if temp_data is None:
				setattr(state, "temp_data", {})
				temp_data = state.temp_data
			if isinstance(temp_data, dict):
				if self.tool_manager is not None:
					temp_data["_workflow_tool_manager"] = self.tool_manager
				if self.tool_bindings:
					temp_data["_workflow_tool_bindings"] = self.tool_bindings
		except Exception:
			pass

		return await self.compiled_graph.ainvoke(state)

	def run(self, state: Any, **kwargs: Any) -> Any:
		"""同步运行工作流（内部自动创建事件循环）。"""
		return asyncio.run(self.run_async(state, **kwargs))


class WorkflowBuilder:
	"""最小工作流构建器（Phase 0）。"""

	def __init__(
		self,
		state_model: type,
		name: str = "workflow",
		entry_point: Optional[str] = None,
		tool_manager: Any = None,
	):
		self.state_model = state_model
		self.name = name
		self._graph = StateGraph(state_model)
		self._entry_point: Optional[str] = entry_point
		self._tool_bindings: Dict[str, List[Any]] = {}
		if tool_manager is None:
			try:
				from flowagent.tools.manager import ToolManager

				tool_manager = ToolManager()
			except Exception:
				tool_manager = None
		self._tool_manager = tool_manager

	def bind_tools(self, role_or_node: str, tools: List[Any]) -> "WorkflowBuilder":
		"""在 workflow 层声明工具（当前先实现后置工具）。

		- role_or_node: 优先按 role_name 绑定（与 agent_node 的 name/agent.role_name 对齐）
		- tools: LangChain Tool 列表
		"""
		key = str(role_or_node)
		existing = self._tool_bindings.get(key, [])
		# 去重（按 tool.name）
		seen = {getattr(t, "name", id(t)) for t in existing}
		for t in tools:
			name = getattr(t, "name", None)
			marker = name if name else id(t)
			if marker in seen:
				continue
			seen.add(marker)
			existing.append(t)
		self._tool_bindings[key] = existing

		# 同步注册到 workflow 的 tool_manager（若可用）
		if self._tool_manager is not None:
			try:
				for t in tools:
					self._tool_manager.register_post_tool(t, role=key)
			except Exception:
				pass

		return self

	def add_node(self, name: str, func: NodeCallable | AsyncNodeCallable) -> "WorkflowBuilder":
		"""添加一个节点。func 可以是同步或异步函数。"""

		async def _wrapped(state: Any):
			if asyncio.iscoroutinefunction(func):
				return await func(state)  # type: ignore[misc]
			return func(state)  # type: ignore[misc]

		self._graph.add_node(name, _wrapped)
		return self

	def add_edge(self, src: str, dst: str) -> "WorkflowBuilder":
		"""添加一条顺序边。"""
		self._graph.add_edge(src, dst)
		return self

	def add_conditional_edges(
		self,
		src: str,
		router: Callable[[Any], Any],
		*,
		path_map: Optional[Dict[Any, str]] = None,
		then: Optional[str] = None,
	) -> "WorkflowBuilder":
		"""添加条件边（路由函数由 LangGraph 解释）。

		Args:
			src: 源节点名称
			router: 路由函数，返回一个 key（或直接返回目标节点名）
			path_map: 将 router 返回的 key 映射到目标节点名
			then: 所有分支结束后统一进入的节点（可选）
		"""
		# 兼容不同版本的 langgraph：部分版本不支持 then 参数
		try:
			if then is None:
				self._graph.add_conditional_edges(src, router, path_map=path_map)
			else:
				self._graph.add_conditional_edges(src, router, path_map=path_map, then=then)
		except TypeError:
			# fallback: 不传 then
			self._graph.add_conditional_edges(src, router, path_map=path_map)
		return self

	def set_entry(self, name: str) -> "WorkflowBuilder":
		self._entry_point = name
		self._graph.set_entry_point(name)
		return self

	def add_agent_node(
		self,
		name: str,
		agent: Any,
		*,
		tools: Optional[List[Any]] = None,
	) -> "WorkflowBuilder":
		"""将 BaseAgent 直接添加为工作流节点（内部自动调用 agent_node 适配）。"""
		node_func = agent_node(agent, name=name)
		self._graph.add_node(name, node_func)
		if tools:
			self.bind_tools(name, tools)
		return self

	def chain(self, *names: str) -> "WorkflowBuilder":
		"""将多个节点串联为线性链，并自动设置入口。"""
		if not names:
			return self
		if not self._entry_point:
			self.set_entry(names[0])
		for i in range(len(names) - 1):
			self.add_edge(names[i], names[i + 1])
		return self

	def compile(self) -> Workflow:
		if not self._entry_point:
			raise ValueError("WorkflowBuilder.compile() 需要先 set_entry(entry_node)")
		compiled = self._graph.compile()
		return Workflow(
			name=self.name,
			state_model=self.state_model,
			compiled_graph=compiled,
			tool_manager=self._tool_manager,
			tool_bindings=self._tool_bindings,
		)

