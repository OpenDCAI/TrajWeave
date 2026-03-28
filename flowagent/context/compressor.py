"""
对话压缩器 - DeerFlow 2.0 对齐

使用 LLM 将旧对话消息压缩为简洁摘要，释放上下文空间。
DeerFlow 2.0 的做法：分段压缩，保留关键决策和用户偏好。

使用示例：
compressor = ContextCompressor(state)
summary, remaining_msgs = await compressor.compress(
    messages=state.messages,
    keep_recent=10,
)
state.context_summary = summary
state.messages = remaining_msgs
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from flowagent.logger import get_logger

if TYPE_CHECKING:
    from flowagent.state.base import MainState

log = get_logger(__name__)

# 压缩提示词模板
_COMPRESSION_PROMPT = """请简洁地总结以下对话内容。

要求：
1. 保留所有关键决策和结论
2. 保留用户表达的偏好和修正
3. 保留重要的工具调用结果
4. 丢弃寒暄、失败的尝试和重复信息
5. 使用简洁的要点形式，不超过 500 字

对话内容：
{conversation}"""


class ContextCompressor:
    """LLM 驱动的对话压缩器

    通过调用一个便宜/快速的 LLM 模型将旧对话消息
    压缩为结构化摘要，保留关键信息。

    Attributes:
    state: MainState 实例（用于获取 LLM 配置）
    """

    def __init__(self, state: "MainState"):
        self._state = state

    async def compress(
        self,
        messages: list,
        keep_recent: int = 10,
    ) -> Tuple[str, list]:
        """压缩旧消息为摘要，保留最近 N 条

        DeerFlow 2.0 模式：分段压缩，不是一次性压缩全部。

        Args:
        messages: 完整消息列表
        keep_recent: 保留最近 N 条消息（不压缩）

        Returns:
        (summary_text, remaining_messages) 元组
        """
        if len(messages) <= keep_recent:
            log.info("[ContextCompressor] 消息数不足，无需压缩")
            return "", messages

        to_compress = messages[:-keep_recent]
        to_keep = messages[-keep_recent:]

        log.info(
            f"[ContextCompressor] 压缩 {len(to_compress)} 条消息, "
            f"保留最近 {len(to_keep)} 条"
        )

        # 格式化待压缩的对话
        conversation_text = self._format_messages(to_compress)

        # 调用 LLM 生成摘要
        summary = await self._generate_summary(conversation_text)

        log.info(f"[ContextCompressor] 生成摘要: {len(summary)} 字符")
        return summary, to_keep

    async def _generate_summary(self, conversation_text: str) -> str:
        """调用 LLM 生成对话摘要"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            # 使用与 state 相同的 LLM 配置，但降低温度
            llm = ChatOpenAI(
                openai_api_base=self._state.request.chat_api_url,
                openai_api_key=self._state.request.api_key,
                model_name=self._state.request.model,
                temperature=0.0,
                max_tokens=2000,
            )

            prompt = _COMPRESSION_PROMPT.format(conversation=conversation_text)
            response = await llm.ainvoke([
                SystemMessage(content="你是一个对话摘要专家，擅长提取关键信息。"),
                HumanMessage(content=prompt),
            ])

            return response.content

        except Exception as e:
            log.error(f"[ContextCompressor] LLM 摘要生成失败: {e}")
            # 降级方案：简单截断
            return self._fallback_summary(conversation_text)

    def _format_messages(self, messages: list) -> str:
        """将消息列表格式化为文本"""
        parts = []
        for msg in messages:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            if isinstance(content, str) and len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)

    @staticmethod
    def _fallback_summary(text: str) -> str:
        """降级摘要方案：截取开头和结尾"""
        if len(text) <= 1000:
            return text
        return (
            text[:400]
            + "\n\n... [中间内容已省略] ...\n\n"
            + text[-400:]
        )
