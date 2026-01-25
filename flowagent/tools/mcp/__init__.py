"""
FlowAgent MCP (Model Context Protocol) 支持模块

提供MCP协议的客户端实现，支持连接外部MCP服务器并将其工具转换为LangChain Tool。
"""
from flowagent.tools.mcp.client import MCPClient
from flowagent.tools.mcp.adapters import MCPToolAdapter

__all__ = [
    "MCPClient",
    "MCPToolAdapter",
]
