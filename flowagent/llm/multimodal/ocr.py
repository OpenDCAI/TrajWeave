"""
OCR识别功能

参考Paper2Any的req_ocr.py实现。
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional
import httpx

from flowagent.logger import get_logger
from flowagent.llm.multimodal.providers import get_provider

log = get_logger(__name__)


async def call_ocr_async(
    model: str,
    messages: List[Dict[str, Any]],
    api_url: str,
    api_key: str,
    image_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.01,
    timeout: int = 120,
    **kwargs,
) -> str:
    """
    调用OCR识别

    Args:
        model: 模型名称
        messages: 消息列表
        api_url: API URL
        api_key: API密钥
        image_path: 图像路径
        max_tokens: 最大token数
        temperature: 温度
        timeout: 超时时间

    Returns:
        OCR识别结果
    """
    log.info(f"调用OCR: model={model}")

    # 准备消息
    processed_messages = [m.copy() for m in messages]

    # 如果有图像，注入到消息中
    if image_path:
        b64, fmt = _encode_image(image_path)
        if processed_messages:
            last_msg = processed_messages[-1]
            if last_msg.get("role") == "user":
                # 图像优先于文本
                last_msg["content"] = [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/{fmt};base64,{b64}"}},
                    {"type": "text", "text": last_msg.get("content", "")},
                ]

    # 获取Provider并构建请求
    provider = get_provider(api_url, model)
    url, payload = provider.build_chat_request(
        processed_messages, api_url, model,
        max_tokens=max_tokens, temperature=temperature
    )

    # 发送请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return provider.parse_chat_response(data)


def _encode_image(image_path: str) -> tuple:
    """编码图像为base64"""
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")

    ext = image_path.rsplit(".", 1)[-1].lower()
    fmt = "jpeg" if ext in {"jpg", "jpeg"} else ext

    return b64, fmt
