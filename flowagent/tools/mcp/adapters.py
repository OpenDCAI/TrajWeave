"""
MCP工具适配器

将MCP工具转换为LangChain Tool格式。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from langchain_core.tools import Tool
from pydantic import BaseModel, create_model

from flowagent.logger import get_logger
from flowagent.tools.mcp.protocol import MCPTool

if TYPE_CHECKING:
    from flowagent.tools.mcp.client import MCPClient

log = get_logger(__name__)


class MCPToolAdapter:
    """
    MCP工具适配器 - 将MCP工具转换为LangChain Tool

    Example:
        >>> adapter = MCPToolAdapter(mcp_client)
        >>> tools = adapter.to_langchain_tools()
    """

    def __init__(self, client: "MCPClient"):
        self.client = client

    def to_langchain_tool(self, mcp_tool: MCPTool) -> Tool:
        """
        将单个MCP工具转换为LangChain Tool

        Args:
            mcp_tool: MCP工具定义

        Returns:
            LangChain Tool实例
        """
        # 创建工具调用函数
        async def tool_func(**kwargs) -> str:
            result = await self.client.call_tool(mcp_tool.name, kwargs)
            if result.is_error:
                return f"错误: {result.error_message}"
            return str(result.content)

        # 从input_schema创建Pydantic模型
        args_schema = self._create_args_schema(mcp_tool)

        tool = Tool(
            name=mcp_tool.name,
            description=mcp_tool.description,
            func=lambda **kwargs: None,  # 同步占位
            coroutine=tool_func,
            args_schema=args_schema
        )

        log.debug(f"转换MCP工具: {mcp_tool.name}")
        return tool

    def to_langchain_tools(self, mcp_tools: List[MCPTool]) -> List[Tool]:
        """
        批量转换MCP工具为LangChain Tool

        Args:
            mcp_tools: MCP工具列表

        Returns:
            LangChain Tool列表
        """
        return [self.to_langchain_tool(t) for t in mcp_tools]

    def _create_args_schema(self, mcp_tool: MCPTool) -> type:
        """从MCP input_schema创建Pydantic模型"""
        schema = mcp_tool.input_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        fields = {}
        for name, prop in properties.items():
            field_type = self._json_type_to_python(prop.get("type", "string"))
            default = ... if name in required else None
            fields[name] = (field_type, default)

        if not fields:
            fields["input"] = (str, ...)

        model = create_model(f"{mcp_tool.name}Args", **fields)
        return model

    def _json_type_to_python(self, json_type: str) -> type:
        """JSON Schema类型转Python类型"""
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_map.get(json_type, str)
