from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Callable, Set, TYPE_CHECKING
from langchain_core.tools import Tool
from flowagent.logger import get_logger

if TYPE_CHECKING:
    from flowagent.tools.mcp.client import MCPClient

log = get_logger(__name__)


class ToolManager:
    """工具管理器 - 支持不同角色的工具管理"""

    def __init__(self):
        self.role_pre_tools: Dict[str, Dict[str, Callable]] = {}
        self.role_post_tools: Dict[str, List[Tool]] = {}
        self.global_pre_tools: Dict[str, Callable] = {}
        self.global_post_tools: List[Tool] = []

    def register_pre_tool(self, name: str, func: Callable, role: Optional[str] = None):
        """注册前置工具"""
        if role:
            if role not in self.role_pre_tools:
                self.role_pre_tools[role] = {}
            self.role_pre_tools[role][name] = func
            log.info(f"为角色 '{role}' 注册前置工具: {name}")
        else:
            self.global_pre_tools[name] = func
            log.info(f"注册全局前置工具: {name}")

    def register_post_tool(self, tool: Tool, role: Optional[str] = None):
        """注册后置工具"""
        if role:
            if role not in self.role_post_tools:
                self.role_post_tools[role] = []
            self.role_post_tools[role].append(tool)
            log.info(f"为角色 '{role}' 注册后置工具: {tool.name}")
        else:
            self.global_post_tools.append(tool)
            log.info(f"注册全局后置工具: {tool.name}")

    def get_pre_tools(self, role: str) -> Dict[str, Callable]:
        """获取指定角色的前置工具（包含全局工具）"""
        tools = self.global_pre_tools.copy()
        if role in self.role_pre_tools:
            tools.update(self.role_pre_tools[role])
        return tools

    def get_post_tools(self, role: str) -> List[Tool]:
        """获取指定角色的后置工具（包含全局工具）"""
        tools = self.global_post_tools.copy()
        if role in self.role_post_tools:
            tools.extend(self.role_post_tools[role])
        return tools

    async def execute_pre_tools(self, role: str) -> Dict[str, Any]:
        """执行指定角色的前置工具"""
        tools = self.get_pre_tools(role)
        results = {}

        for name, func in tools.items():
            try:
                if asyncio.iscoroutinefunction(func):
                    results[name] = await func()
                else:
                    results[name] = func()
                log.info(f"角色 '{role}' 前置工具 '{name}' 执行成功")
            except Exception as e:
                log.error(f"角色 '{role}' 前置工具 '{name}' 执行失败: {e}")
                results[name] = None

        return results

    def get_available_roles(self) -> Set[str]:
        roles = set(self.role_pre_tools.keys())
        roles.update(self.role_post_tools.keys())
        return roles

    # ==================== Agent-as-Tool 支持 ====================

    def register_agent_as_tool(self, agent, state, role: Optional[str] = None):
        """
        将 agent 注册为后置工具

        Args:
            agent: BaseAgent 实例
            state: DFState 实例
            role: 要注册到哪个角色的后置工具，None 表示全局工具
        """
        tool = agent.as_tool(state)
        self.register_post_tool(tool, role)
        log.info(f"Agent '{agent.role_name}' 注册为工具 '{tool.name}' (角色: {role or 'global'})")
        return tool

    def register_multiple_agents_as_tools(self, agents: List, state, role: Optional[str] = None):
        """
        批量注册多个 agent 作为后置工具

        Args:
            agents: BaseAgent 实例列表
            state: DFState 实例
            role: 目标角色
        """
        tools = []
        for agent in agents:
            tool = self.register_agent_as_tool(agent, state, role)
            tools.append(tool)
        log.info(f"批量注册了 {len(tools)} 个 agent 作为工具")
        return tools

    # ==================== MCP 支持 ====================

    async def register_mcp_server(
        self,
        server_url: str,
        role: Optional[str] = None,
        timeout: int = 30
    ) -> List[Tool]:
        """
        注册MCP服务器的所有工具

        Args:
            server_url: MCP服务器URL
            role: 目标角色，None表示全局
            timeout: 连接超时时间

        Returns:
            注册的工具列表
        """
        from flowagent.tools.mcp.client import MCPClient
        from flowagent.tools.mcp.adapters import MCPToolAdapter

        client = MCPClient(timeout=timeout)
        await client.connect(server_url)

        mcp_tools = await client.list_tools()
        adapter = MCPToolAdapter(client)
        langchain_tools = adapter.to_langchain_tools(mcp_tools)

        for tool in langchain_tools:
            self.register_post_tool(tool, role)

        log.info(f"从MCP服务器 {server_url} 注册了 {len(langchain_tools)} 个工具")
        return langchain_tools

    # ==================== Skills 支持 ====================

    def register_skill(self, skill, role: Optional[str] = None) -> None:
        """
        注册Skill的所有工具

        Args:
            skill: Skill实例
            role: 目标角色，None表示全局
        """
        tools = skill.get_tools()
        for tool in tools:
            self.register_post_tool(tool, role)

        log.info(f"注册Skill '{skill.name}' 的 {len(tools)} 个工具")

    # ==================== 新增方法 ====================

    def get_subset(self, tool_names: List[str]) -> "ToolManager":
        """创建一个新的 ToolManager，只包含指定名称的工具

        Args:
            tool_names: 要保留的工具名称列表

        Returns:
            新的 ToolManager 实例，只包含指定的工具
        """
        subset = ToolManager()
        name_set = set(tool_names)

        # 过滤全局前置工具
        for name, func in self.global_pre_tools.items():
            if name in name_set:
                subset.global_pre_tools[name] = func

        # 过滤全局后置工具
        for tool in self.global_post_tools:
            if tool.name in name_set:
                subset.global_post_tools.append(tool)

        # 过滤角色前置工具
        for role, tools_dict in self.role_pre_tools.items():
            for name, func in tools_dict.items():
                if name in name_set:
                    if role not in subset.role_pre_tools:
                        subset.role_pre_tools[role] = {}
                    subset.role_pre_tools[role][name] = func

        # 过滤角色后置工具
        for role, tools_list in self.role_post_tools.items():
            for tool in tools_list:
                if tool.name in name_set:
                    if role not in subset.role_post_tools:
                        subset.role_post_tools[role] = []
                    subset.role_post_tools[role].append(tool)

        return subset

    def get_all_post_tools(self) -> List[Tool]:
        """返回所有后置工具（全局 + 所有角色），去重

        Returns:
            去重后的所有后置工具列表
        """
        seen_names: Set[str] = set()
        result: List[Tool] = []

        for tool in self.global_post_tools:
            if tool.name not in seen_names:
                seen_names.add(tool.name)
                result.append(tool)

        for role_tools in self.role_post_tools.values():
            for tool in role_tools:
                if tool.name not in seen_names:
                    seen_names.add(tool.name)
                    result.append(tool)

        return result

    def list_tool_names(self) -> List[str]:
        """返回所有已注册工具名称的排序列表

        Returns:
            排序后的工具名称列表
        """
        names: Set[str] = set()

        # 全局前置工具名
        names.update(self.global_pre_tools.keys())

        # 全局后置工具名
        for tool in self.global_post_tools:
            names.add(tool.name)

        # 角色前置工具名
        for tools_dict in self.role_pre_tools.values():
            names.update(tools_dict.keys())

        # 角色后置工具名
        for tools_list in self.role_post_tools.values():
            for tool in tools_list:
                names.add(tool.name)

        return sorted(names)


_tool_manager_instance = None


def get_tool_manager() -> ToolManager:
    global _tool_manager_instance
    if _tool_manager_instance is None:
        _tool_manager_instance = ToolManager()
    return _tool_manager_instance
