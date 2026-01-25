"""
文本转语音功能

参考Paper2Any的req_tts.py实现。
"""
from __future__ import annotations

import os
from typing import Optional
import httpx

from flowagent.logger import get_logger

log = get_logger(__name__)


async def generate_speech_async(
    text: str,
    save_path: str,
    api_url: str,
    api_key: str,
    model: str = "tts-1",
    voice: str = "alloy",
    timeout: int = 120,
    **kwargs,
) -> str:
    """
    生成语音

    Args:
        text: 要转换的文本
        save_path: 保存路径
        api_url: API URL
        api_key: API密钥
        model: 模型名称
        voice: 声音类型
        timeout: 超时时间

    Returns:
        保存的文件路径
    """
    log.info(f"生成语音: model={model}, voice={voice}")

    url = f"{api_url.rstrip('/')}/audio/speech"
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(resp.content)

    log.info(f"语音已保存: {save_path}")
    return save_path
