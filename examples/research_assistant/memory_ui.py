"""记忆面板 UI 辅助函数"""
import os
import chainlit as cl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowagent.memory.context import MemoryContext
    from flowagent.context.budget import ContextBudgetManager

# 确保 .files 目录存在（Chainlit 在 Windows 上不会自动创建）
_files_dir = os.path.join(os.path.dirname(__file__), ".files")
os.makedirs(_files_dir, exist_ok=True)


async def display_memory_panel(memory_ctx: "MemoryContext") -> None:
    """在 Chainlit 侧边栏展示用户记忆"""
    if not memory_ctx or not memory_ctx.is_loaded:
        return

    sections = []

    if memory_ctx.profile and memory_ctx.profile.preferences:
        prefs = "\n".join(
            f"- **{k}**: {v}"
            for k, v in memory_ctx.profile.preferences.items()
        )
        sections.append(f"### User Preferences\n{prefs}")

    if memory_ctx.profile and memory_ctx.profile.technical_background:
        bg = ", ".join(memory_ctx.profile.technical_background)
        sections.append(f"### Technical Background\n{bg}")

    if memory_ctx.facts:
        facts = "\n".join(
            f"- [{f.category}] {f.content}"
            for f in memory_ctx.facts[:10]
        )
        sections.append(f"### Known Facts\n{facts}")

    if memory_ctx.session_summaries:
        summaries = "\n".join(
            f"- {s}" for s in memory_ctx.session_summaries[:5]
        )
        sections.append(f"### Recent Sessions\n{summaries}")

    if sections:
        content = "## Agent Memory\n\n" + "\n\n".join(sections)
    else:
        content = "## Agent Memory\n\n_No memory yet. I'll learn your preferences as we interact._"

    elements = [cl.Text(name="memory", content=content, display="side")]
    await cl.Message(content="", elements=elements).send()


async def display_budget_stats(budget_manager: "ContextBudgetManager", messages: list) -> None:
    """展示 Token 使用统计"""
    if not budget_manager:
        return

    stats = budget_manager.get_usage_stats(messages)
    utilization_pct = stats["utilization"] * 100

    content = (
        f"**Token Usage**: {stats['used_tokens']:,} / {stats['max_tokens']:,} "
        f"({utilization_pct:.1f}%)\n"
        f"**Messages**: {stats['message_count']}\n"
        f"**Needs Compression**: {'Yes' if stats['needs_compression'] else 'No'}"
    )

    elements = [cl.Text(name="budget", content=content, display="side")]
    await cl.Message(content="", elements=elements).send()
