"""
视频处理功能

参考Paper2Any的req_videos.py实现。
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional
import httpx

from flowagent.logger import get_logger
from flowagent.llm.multimodal.providers import get_provider

log = get_logger(__name__)


async def call_video_understanding_async(
    model: str,
    messages: List[Dict[str, Any]],
    api_url: str,
    api_key: str,
    video_path: str,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    timeout: int = 300,
    **kwargs,
) -> str:
    """
    调用视频理解

    Args:
        model: 模型名称
        messages: 消息列表
        api_url: API URL
        api_key: API密钥
        video_path: 视频路径
        max_tokens: 最大token数
        temperature: 温度
        timeout: 超时时间

    Returns:
        视频理解结果
    """
    log.info(f"调用视频理解: model={model}, video={video_path}")

    # 编码视频
    b64, mime = _encode_video(video_path)

    # 准备消息
    processed_messages = [m.copy() for m in messages]
    if processed_messages:
        last_msg = processed_messages[-1]
        if last_msg.get("role") == "user":
            last_msg["content"] = [
                {"type": "video_url",
                 "video_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": last_msg.get("content", "")},
            ]

    # 获取Provider并构建请求
    provider = get_provider(api_url, model)
    url, payload = provider.build_chat_request(
        processed_messages, api_url, model,
        max_tokens=max_tokens, temperature=temperature
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return provider.parse_chat_response(data)


def _encode_video(video_path: str) -> tuple:
    """编码视频为base64，大文件自动压缩"""
    # 检查文件大小
    size_mb = os.path.getsize(video_path) / (1024 * 1024)

    if size_mb > 20:
        log.info(f"视频文件过大({size_mb:.1f}MB)，进行压缩")
        video_path = _compress_video(video_path)

    with open(video_path, "rb") as f:
        raw = f.read()

    b64 = base64.b64encode(raw).decode("utf-8")
    ext = video_path.rsplit(".", 1)[-1].lower()
    mime = f"video/{ext}"

    return b64, mime


def _compress_video(input_path: str) -> str:
    """使用FFmpeg压缩视频"""
    output_path = tempfile.mktemp(suffix=".mp4")

    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", "scale=-2:720",
        "-crf", "28",
        "-preset", "fast",
        "-y", output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except Exception as e:
        log.warning(f"视频压缩失败: {e}")
        return input_path
