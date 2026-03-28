"""
Token 预算管理 - DeerFlow 2.0 对齐

管理 Agent 会话的上下文 Token 预算，核心能力：
- 计算消息列表的 Token 数量
- 判断是否需要触发压缩
- 基于优先级选择保留哪些消息

DeerFlow 2.0 的上下文工程理念：
- 系统提示词永远保留
- 历史摘要优先于原始消息
- 最近的消息优先于旧消息
- "pinned" 消息（关键决策）优先保留

使用示例：
budget = ContextBudgetManager(max_tokens=120_000)

if budget.needs_compression(state.messages):
    # 需要压缩，调用 ContextCompressor
    compressor = ContextCompressor(llm)
    summary, trimmed = await compressor.compress(state.messages)
    state.context_summary = summary
    state.messages = trimmed

# 选择在预算内的消息
selected = budget.select_messages(state.messages, state.context_summary)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from flowagent.logger import get_logger

log = get_logger(__name__)


def _get_encoder():
    """获取 tiktoken 编码器（懒加载，优雅降级）"""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        return None


class ContextBudgetManager:
    """Token 预算管理器

    Attributes:
    max_tokens: 最大允许 Token 数
    reserve_for_response: 为 LLM 响应预留的 Token 数
    compression_threshold: 触发压缩的阈值（已用/可用比例）
    """

    def __init__(
        self,
        max_tokens: int = 120_000,
        reserve_for_response: int = 4_000,
        compression_threshold: float = 0.75,
    ):
        self.max_tokens = max_tokens
        self.reserve_for_response = reserve_for_response
        self.compression_threshold = compression_threshold
        self._encoder = _get_encoder()

    @property
    def available_tokens(self) -> int:
        """可用于消息的 Token 数"""
        return self.max_tokens - self.reserve_for_response

    def count_tokens(self, text: str) -> int:
        """计算文本的 Token 数

        如果 tiktoken 不可用，使用粗略估算（字符数 / 4）。
        """
        if self._encoder:
            return len(self._encoder.encode(text))
        # 粗略估算: 英文 ~4 chars/token, 中文 ~2 chars/token
        return len(text) // 3

    def count_messages_tokens(self, messages: list) -> int:
        """计算消息列表的总 Token 数

        Args:
        messages: BaseMessage 列表或 dict 列表

        Returns:
        总 Token 数
        """
        total = 0
        for msg in messages:
            if hasattr(msg, "content"):
                content = str(msg.content)
            elif isinstance(msg, dict):
                content = str(msg.get("content", ""))
            else:
                content = str(msg)
            total += self.count_tokens(content) + 4  # 每条消息的 overhead
        return total

    def needs_compression(self, messages: list) -> bool:
        """判断当前消息是否需要压缩

        当已使用 Token 超过 available * threshold 时返回 True。

        Args:
        messages: 当前消息列表

        Returns:
        是否需要压缩
        """
        used = self.count_messages_tokens(messages)
        threshold = self.available_tokens * self.compression_threshold
        needs = used > threshold
        if needs:
            log.info(
                f"[ContextBudget] 需要压缩: used={used}, "
                f"threshold={threshold:.0f}, available={self.available_tokens}"
            )
        return needs

    def select_messages(
        self,
        messages: list,
        summary: Optional[str] = None,
    ) -> list:
        """基于优先级选择在预算内的消息

        优先级顺序：
        1. 系统消息（永远保留）
        2. 历史摘要（如果有）
        3. 最近的消息（从最新向最旧填充）

        Args:
        messages: 完整消息列表
        summary: 压缩后的历史摘要（可选）

        Returns:
        筛选后的消息列表
        """
        budget = self.available_tokens
        selected = []

        # 1. 系统消息永远保留
        system_msgs = []
        other_msgs = []
        for msg in messages:
            role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
            if role == "system":
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)

        for msg in system_msgs:
            cost = self.count_tokens(str(getattr(msg, "content", msg)))
            budget -= cost
            selected.append(msg)

        # 2. 注入历史摘要
        if summary:
            from langchain_core.messages import SystemMessage
            summary_msg = SystemMessage(content=f"## 上一轮对话摘要\n{summary}")
            cost = self.count_tokens(summary)
            if cost <= budget:
                budget -= cost
                selected.append(summary_msg)

        # 3. 从最近的消息开始填充
        for msg in reversed(other_msgs):
            content = str(getattr(msg, "content", msg))
            cost = self.count_tokens(content) + 4
            if cost <= budget:
                budget -= cost
                selected.insert(len(system_msgs) + (1 if summary else 0), msg)
            else:
                break

        log.info(
            f"[ContextBudget] 选择了 {len(selected)}/{len(messages)} 条消息, "
            f"剩余预算: {budget}"
        )
        return selected

    def get_usage_stats(self, messages: list) -> Dict[str, Any]:
        """获取 Token 使用统计

        Returns:
        包含 used, available, utilization 等信息的字典
        """
        used = self.count_messages_tokens(messages)
        return {
            "used_tokens": used,
            "max_tokens": self.max_tokens,
            "available_tokens": self.available_tokens,
            "utilization": used / self.available_tokens if self.available_tokens > 0 else 0,
            "needs_compression": used > self.available_tokens * self.compression_threshold,
            "message_count": len(messages),
        }
