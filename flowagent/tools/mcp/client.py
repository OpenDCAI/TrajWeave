"""
MCP客户端实现

提供连接MCP服务器、列出工具、调用工具的功能。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
import httpx

from flowagent.logger import get_logger
from flowagent.tools.mcp.protocol import MCPTool, MCPToolCall, MCPToolResult, MCPServerInfo

log = get_logger(__name__)


class MCPClient:
    """
    MCP客户端 - 连接外部MCP服务

    Example:
        >>> client = MCPClient()
        >>> await client.connect("http://localhost:3000")
        >>> tools = await client.list_tools()
        >>> result = await client.call_tool("search", {"query": "hello"})
    """

    def __init__(self, timeout: int = 30):
        self.server_url: Optional[str] = None
        self.server_info: Optional[MCPServerInfo] = None
        self.timeout = timeout
        self._tools_cache: Optional[List[MCPTool]] = None
        self._connected = False

    async def connect(self, server_url: str) -> MCPServerInfo:
        """
        连接到MCP服务器

        Args:
            server_url: MCP服务器URL

        Returns:
            服务器信息
        """
        self.server_url = server_url.rstrip("/")
        log.info(f"连接MCP服务器: {self.server_url}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 获取服务器信息
                resp = await client.get(f"{self.server_url}/info")
                resp.raise_for_status()
                data = resp.json()

                self.server_info = MCPServerInfo(
                    name=data.get("name", "unknown"),
                    version=data.get("version", "0.0.0"),
                    capabilities=data.get("capabilities", [])
                )
                self._connected = True
                log.info(f"已连接: {self.server_info.name} v{self.server_info.version}")
                return self.server_info

        except Exception as e:
            log.error(f"连接MCP服务器失败: {e}")
            raise ConnectionError(f"无法连接到MCP服务器: {server_url}") from e

    async def list_tools(self, force_refresh: bool = False) -> List[MCPTool]:
        """
        列出服务器提供的所有工具

        Args:
            force_refresh: 是否强制刷新缓存

        Returns:
            工具列表
        """
        if not self._connected:
            raise RuntimeError("未连接到MCP服务器，请先调用connect()")

        if self._tools_cache and not force_refresh:
            return self._tools_cache

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(f"{self.server_url}/tools")
                resp.raise_for_status()
                data = resp.json()

                self._tools_cache = [
                    MCPTool(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {})
                    )
                    for t in data.get("tools", [])
                ]
                log.info(f"获取到 {len(self._tools_cache)} 个MCP工具")
                return self._tools_cache

        except Exception as e:
            log.error(f"获取MCP工具列表失败: {e}")
            raise

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """
        调用MCP工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具调用结果
        """
        if not self._connected:
            raise RuntimeError("未连接到MCP服务器")

        log.info(f"调用MCP工具: {name}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.server_url}/tools/{name}/call",
                    json={"arguments": arguments}
                )
                resp.raise_for_status()
                data = resp.json()

                return MCPToolResult(
                    content=data.get("content"),
                    is_error=data.get("isError", False),
                    error_message=data.get("errorMessage")
                )

        except httpx.HTTPStatusError as e:
            log.error(f"MCP工具调用HTTP错误: {e}")
            return MCPToolResult(
                content=None,
                is_error=True,
                error_message=str(e)
            )
        except Exception as e:
            log.error(f"MCP工具调用失败: {e}")
            return MCPToolResult(
                content=None,
                is_error=True,
                error_message=str(e)
            )

    async def disconnect(self):
        """断开与MCP服务器的连接"""
        self._connected = False
        self._tools_cache = None
        self.server_info = None
        log.info("已断开MCP服务器连接")

    @property
    def is_connected(self) -> bool:
        return self._connected
