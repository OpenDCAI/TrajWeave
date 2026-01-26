"""
图像生成/编辑功能

参考Paper2Any的req_img.py实现，简化版本。
"""
from __future__ import annotations

import base64
from typing import Optional
import httpx

from flowagent.logger import get_logger
from flowagent.llm.multimodal.providers import get_provider

log = get_logger(__name__)


async def generate_or_edit_and_save_image_async(
    prompt: str,
    save_path: str,
    api_url: str,
    api_key: str,
    model: str,
    *,
    image_path: Optional[str] = None,
    use_edit: bool = False,
    size: str = "1024x1024",
    aspect_ratio: str = '16:9',
    quality: str = "standard",
    style: str = "vivid",
    timeout: int = 120,
    **kwargs,
) -> str:
    """
    图像生成/编辑

    Args:
        prompt: 提示词
        save_path: 保存路径
        api_url: API URL
        api_key: API密钥
        model: 模型名称
        image_path: 输入图像路径（编辑模式）
        use_edit: 是否使用编辑模式
        size: 图像尺寸
        aspect_ratio: 宽高比
        quality: 质量
        style: 风格
        timeout: 超时时间

    Returns:
        Base64编码的图像数据
    """
    log.info(f"图像生成: model={model}, edit={use_edit}")

    # 获取Provider
    provider = get_provider(api_url, model)

    # 构建请求
    if use_edit and image_path:
        # 编辑模式：读取并编码输入图像
        b64_input, fmt = _encode_image(image_path)
        # 注意：简化版本暂不支持编辑模式，使用生成模式
        log.warning("简化版本暂不支持编辑模式，使用生成模式")
        url, payload, is_stream = provider.build_generation_request(
            prompt=prompt,
            api_url=api_url,
            model=model,
            size=size,
            quality=quality,
            style=style,
            aspect_ratio=aspect_ratio,
            **kwargs
        )
    else:
        # 生成模式
        url, payload, is_stream = provider.build_generation_request(
            prompt=prompt,
            api_url=api_url,
            model=model,
            size=size,
            quality=quality,
            style=style,
            aspect_ratio=aspect_ratio,
            **kwargs
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

    # 解析响应
    b64_result = provider.parse_generation_response(data)

    # 保存图像
    _save_base64_image(b64_result, save_path)

    return b64_result


def _encode_image(image_path: str) -> tuple:
    """编码图像为base64"""
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")

    ext = image_path.rsplit(".", 1)[-1].lower()
    fmt = "jpeg" if ext in {"jpg", "jpeg"} else ext

    return b64, fmt


def _save_base64_image(b64_data: str, save_path: str):
    """保存base64图像到文件"""
    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    image_data = base64.b64decode(b64_data)
    with open(save_path, "wb") as f:
        f.write(image_data)

    log.info(f"图像已保存: {save_path}")
