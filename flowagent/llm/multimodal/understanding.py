"""
图像理解功能

参考Paper2Any的req_understanding.py实现。
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional
import httpx

from flowagent.logger import get_logger
from flowagent.llm.multimodal.providers import get_provider

log = get_logger(__name__)


async def call_image_understanding_async(
    model: str,
    messages: List[Dict[str, Any]],
    api_url: str,
    api_key: str,
    image_path: Optional[str] = None,
    max_tokens: int = 16384,
    temperature: float = 0.1,
    timeout: int = 120,
    **kwargs,
) -> str:
    """
    调用通用图像理解模型

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
        图像理解结果
    """
    log.info(f"调用图像理解: model={model}")

    # 准备消息
    processed_messages = [m.copy() for m in messages]

    # 处理图像
    if image_path:
        b64, fmt = _encode_image(image_path)

        target_msg = None
        if processed_messages:
            last_msg = processed_messages[-1]
            if last_msg["role"] == "user":
                target_msg = last_msg

        if target_msg:
            original_content = target_msg["content"]

            if isinstance(original_content, str):
                target_msg["content"] = [
                    {"type": "text", "text": original_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
                ]
            elif isinstance(original_content, list):
                target_msg["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
                )
        else:
             # 如果没有 user 消息或列表为空，追加一条
            processed_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
                ]
            })

    # 使用 Provider 构造请求
    provider = get_provider(api_url, model)
    url, payload = provider.build_chat_request(
        api_url=api_url,
        model=model,
        messages=processed_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )

    # 发送请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # 解析响应
    return provider.parse_chat_response(data)


def _encode_image(image_path: str) -> tuple:
    """编码图像为base64"""
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")

    ext = image_path.rsplit(".", 1)[-1].lower()
    if ext in {"jpg", "jpeg"}:
        fmt = "jpeg"
    elif ext == "png":
        fmt = "png"
    else:
        fmt = ext

    return b64, fmt
