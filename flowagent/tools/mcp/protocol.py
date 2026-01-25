"""
MCP协议数据结构定义
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class MCPMessageType(str, Enum):
    """MCP消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPTool:
    """MCP工具定义"""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolCall:
    """MCP工具调用"""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolResult:
    """MCP工具调用结果"""
    content: Any
    is_error: bool = False
    error_message: Optional[str] = None


@dataclass
class MCPServerInfo:
    """MCP服务器信息"""
    name: str
    version: str
    capabilities: List[str] = field(default_factory=list)
