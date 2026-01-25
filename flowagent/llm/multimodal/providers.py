"""
AI Provider策略模式实现

参考Paper2Any的providers.py，支持多个AI服务商的统一接口。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import httpx

from flowagent.logger import get_logger

log = get_logger(__name__)


class AIProviderStrategy(ABC):
    """AI提供商策略基类"""

    @abstractmethod
    def match(self, api_url: str, model: str) -> bool:
        """判断当前策略是否适用"""
        pass

    @abstractmethod
    def build_generation_request(
        self,
        prompt: str,
        api_url: str,
        model: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        """构造文生图请求，返回 (url, payload, is_stream)"""
        pass

    def build_chat_request(
        self,
        messages: List[Dict[str, Any]],
        api_url: str,
        model: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """构造对话/理解请求（默认OpenAI格式）"""
        url = f"{api_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        return url, payload

    @abstractmethod
    def parse_generation_response(self, response_data: Dict) -> str:
        """解析生图响应，返回Base64字符串"""
        pass

    def parse_chat_response(self, response_data: Dict) -> str:
        """解析对话响应（默认OpenAI格式）"""
        return response_data["choices"][0]["message"]["content"]


class GeminiProvider(AIProviderStrategy):
    """Gemini模型提供商"""

    def match(self, api_url: str, model: str) -> bool:
        return "gemini" in model.lower()

    def build_generation_request(
        self,
        prompt: str,
        api_url: str,
        model: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        url = f"{api_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
        }
        return url, payload, False

    def parse_generation_response(self, response_data: Dict) -> str:
        content = response_data["choices"][0]["message"]["content"]
        # 尝试提取base64
        if "base64" in str(response_data):
            return self._extract_base64(response_data)
        return content

    def _extract_base64(self, data: Dict) -> str:
        """从响应中提取base64"""
        import re
        content = str(data)
        match = re.search(r'[A-Za-z0-9+/]{100,}={0,2}', content)
        return match.group() if match else ""


class OpenAIProvider(AIProviderStrategy):
    """OpenAI/DALL-E模型提供商"""

    def match(self, api_url: str, model: str) -> bool:
        return "dall-e" in model.lower() or "gpt-image" in model.lower()

    def build_generation_request(
        self,
        prompt: str,
        api_url: str,
        model: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        url = f"{api_url.rstrip('/')}/images/generations"
        payload = {
            "model": model,
            "prompt": prompt,
            "size": kwargs.get("size", "1024x1024"),
            "quality": kwargs.get("quality", "standard"),
            "response_format": "b64_json",
        }
        return url, payload, False

    def parse_generation_response(self, response_data: Dict) -> str:
        return response_data["data"][0]["b64_json"]


class DefaultProvider(AIProviderStrategy):
    """默认OpenAI兼容提供商"""

    def match(self, api_url: str, model: str) -> bool:
        return True  # 作为后备

    def build_generation_request(
        self,
        prompt: str,
        api_url: str,
        model: str,
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        url = f"{api_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        return url, payload, False

    def parse_generation_response(self, response_data: Dict) -> str:
        return response_data["choices"][0]["message"]["content"]


# 策略列表（按优先级排序）
STRATEGIES = [
    GeminiProvider(),
    OpenAIProvider(),
    DefaultProvider(),  # 后备
]


def get_provider(api_url: str, model: str) -> AIProviderStrategy:
    """根据URL和模型获取合适的Provider"""
    for strategy in STRATEGIES:
        if strategy.match(api_url, model):
            return strategy
    return STRATEGIES[-1]  # 返回默认
