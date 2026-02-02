"""VLM Agent 最小跑通示例

用法1（自动从 .env 加载）：
  python examples/vlm_agent_example.py
  
用法2（PowerShell 手动设置）：
  $env:DF_API_URL = "http://123.129.219.111:3000/v1"
  $env:DF_API_KEY = "..."
  $env:DF_VLM_MODEL = "gpt-4o"   # 可选
  python examples/vlm_agent_example.py

说明：
- 脚本会在 examples/.tmp_vlm/ 下生成一张 64x64 PNG（红色方块）作为输入。
- 通过 create_vlm_agent 创建 VLM 模式 Agent，走 VLMStrategy -> BaseAgent._execute_vlm -> VisionLLMCaller。
"""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

# 尝试从 .env 文件加载环境变量（如果安装了 python-dotenv）
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        print(f"已加载环境变量: {_env_path}")
except ImportError:
    pass  # 没有 python-dotenv，使用系统环境变量

from flowagent.core.factory import create_vlm_agent
from flowagent.state import MainRequest, MainState


def _require_env_any(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value.lower() != "test":
            return value
    raise RuntimeError(f"缺少环境变量（或仍为默认 test）：{', '.join(names)}")


def _write_test_image(out_dir: Path) -> Path:
    """生成一个 64x64 的纯红色方块 PNG（更适合 VLM 测试）"""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "red_square.png"
    
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        img.save(img_path, "PNG")
    except ImportError:
        # 如果没有 PIL，使用预编码的 64x64 红色方块 PNG
        # (手动生成并 base64 编码，或者用更大的测试图)
        _RED_64x64_PNG_B64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAH0lEQVR4nO3BAQEAAADC"
            "oPdPbQ43oAAAAAAAAAAAvg0hAAABaKT6fQAAAABJRU5ErkJggg=="
        )
        img_path.write_bytes(base64.b64decode(_RED_64x64_PNG_B64))
    
    return img_path


async def main() -> None:
    # 兼容两套环境变量：
    # - DF_*：项目默认
    # - AIHUBMIX_*：你当前要测试的站点
    api_url = _require_env_any("AIHUBMIX_BASE_URL", "DF_API_URL")
    api_key = _require_env_any("AIHUBMIX_API_KEY", "DF_API_KEY")
    model = (
        os.getenv("AIHUBMIX_MODEL", "").strip()
        or os.getenv("DF_VLM_MODEL", "").strip()
        or "gpt-4o"
    )
    timeout_s = int(os.getenv("DF_VLM_TIMEOUT", os.getenv("DF_API_TIMEOUT", "30")))

    tmp_dir = Path(__file__).resolve().parent / ".tmp_vlm"
    img_path = _write_test_image(tmp_dir)

    # 这里的 target 会进入 DynamicAgent.build_messages 的任务提示词中
    target = "请描述这张图片的主色是什么？只回答一个英文颜色词。"

    state = MainState(
        request=MainRequest(
            language="zh",
            chat_api_url=api_url.rstrip("/"),
            api_key=api_key,
            model=model,
            target=target,
        )
    )

    print("=== VLM Smoke Test Starting ===")
    print(f"image: {img_path}")
    print(f"model: {model}")
    print(f"api_url: {api_url.rstrip('/')}")
    print(f"timeout_s: {timeout_s}")

    agent = create_vlm_agent(
        role_name="vlm_smoke_test",
        system_prompt="你是一个图像理解助手，必须基于图像内容回答。",
        vlm_mode="understanding",
        image_detail="auto",
        max_image_size=2048,
        # 关键：VisionLLMCaller/understanding.py 从 vlm_config['input_image'] 取图像路径
        additional_params={"input_image": str(img_path), "timeout": timeout_s},
        # 避免默认 JSON 解析器导致的“解析失败”噪音
        parser_type="text",
        temperature=0.0,
        max_tokens=256,
    )

    final_state = await agent.execute(state)
    result = final_state.agent_results.get(agent.role_name, {}).get("results")

    print("=== VLM Smoke Test Result ===")
    print(f"image: {img_path}")
    print(f"model: {model}")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
