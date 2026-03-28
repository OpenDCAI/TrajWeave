"""
记忆上下文 - DeerFlow 2.0 对齐

MemoryContext 是持久化记忆与活跃 Agent 状态之间的桥梁。
- 会话开始时加载用户画像和近期会话摘要
- 执行过程中学习新的知识事实
- 会话结束时持久化检查点和更新画像
- 提供 to_system_prompt_section() 将记忆注入系统提示词

使用示例：
store = SQLiteMemoryStore()
await store.initialize()

ctx = MemoryContext(store, user_id="user_123")
await ctx.load()

# 注入到系统提示词
memory_section = ctx.to_system_prompt_section()
system_prompt = f"{base_prompt}\\n\\n{memory_section}"

# 在交互中学习新事实
await ctx.learn("preference", "用户偏好使用 PyTorch 而非 TensorFlow")

# 会话结束时保存
await ctx.save_session(session_id, summary, task_description)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from flowagent.logger import get_logger
from flowagent.memory.models import KnowledgeFact, SessionCheckpoint, UserProfile

if TYPE_CHECKING:
    from flowagent.memory.store import MemoryBackend

log = get_logger(__name__)


class MemoryContext:
    """记忆上下文 - 持久化记忆与 Agent 运行时的桥梁

    Attributes:
    user_id: 用户标识
    profile: 加载的用户画像
    facts: 加载的知识事实列表
    session_summaries: 近期会话摘要列表
    """

    def __init__(self, store: "MemoryBackend", user_id: str):
        self._store = store
        self.user_id = user_id
        self.profile: Optional[UserProfile] = None
        self.facts: List[KnowledgeFact] = []
        self.session_summaries: List[str] = []
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        """会话开始时加载用户记忆

        加载内容：
        1. 用户画像（偏好、背景、模式）
        2. 知识事实（最近 50 条）
        3. 近期会话摘要（最近 5 个）
        """
        try:
            self.profile = await self._store.load_profile(self.user_id)
            self.facts = await self._store.get_facts(self.user_id, limit=50)
            recent = await self._store.get_recent_sessions(self.user_id, limit=5)
            self.session_summaries = [s.summary for s in recent if s.summary]
            self._loaded = True
            log.info(
                f"加载用户记忆: user={self.user_id}, "
                f"facts={len(self.facts)}, sessions={len(self.session_summaries)}"
            )
        except Exception as e:
            log.warning(f"加载用户记忆失败: {e}")
            self._loaded = True  # 标记已尝试加载，避免重复

    def to_system_prompt_section(self) -> str:
        """将记忆渲染为系统提示词注入段

        DeerFlow 2.0 模式：通过系统提示词让 Agent "记住"用户。
        仅当有实际记忆内容时才生成注入段。

        Returns:
        Markdown 格式的记忆上下文字符串，为空时返回空字符串
        """
        parts: List[str] = []

        if self.profile:
            if self.profile.preferences:
                prefs = "\n".join(
                    f"  - {k}: {v}" for k, v in self.profile.preferences.items()
                )
                parts.append(f"## 用户偏好\n{prefs}")
            if self.profile.technical_background:
                bg = ", ".join(self.profile.technical_background)
                parts.append(f"## 技术背景\n  {bg}")

        if self.facts:
            facts_str = "\n".join(
                f"  - [{f.category}] {f.content}"
                for f in self.facts[:20]  # 最多注入 20 条
            )
            parts.append(f"## 已知事实\n{facts_str}")

        if self.session_summaries:
            summaries = "\n".join(
                f"  - {s}" for s in self.session_summaries[:5]
            )
            parts.append(f"## 近期会话\n{summaries}")

        if not parts:
            return ""

        return "# 用户记忆上下文\n\n" + "\n\n".join(parts)

    async def learn(self, category: str, content: str, confidence: float = 1.0) -> None:
        """在交互过程中学习新的知识事实

        Args:
        category: 分类 - "preference" | "context" | "correction" | "pattern"
        content: 事实内容
        confidence: 置信度 0.0-1.0
        """
        fact = KnowledgeFact(
            fact_id=str(uuid4()),
            user_id=self.user_id,
            category=category,
            content=content,
            confidence=confidence,
        )
        self.facts.append(fact)
        try:
            await self._store.add_fact(fact)
            log.info(f"学习新事实: [{category}] {content[:50]}...")
        except Exception as e:
            log.warning(f"保存知识事实失败: {e}")

    async def update_profile(self, **kwargs) -> None:
        """更新用户画像

        Args:
        **kwargs: 要更新的字段，如 preferences={"language": "python"}
        """
        if self.profile is None:
            self.profile = UserProfile(user_id=self.user_id)

        for key, value in kwargs.items():
            if key == "preferences" and isinstance(value, dict):
                self.profile.preferences.update(value)
            elif key == "technical_background" and isinstance(value, list):
                for tag in value:
                    self.profile.add_background(tag)
            elif key == "work_patterns" and isinstance(value, dict):
                self.profile.work_patterns.update(value)

        try:
            await self._store.save_profile(self.profile)
        except Exception as e:
            log.warning(f"更新用户画像失败: {e}")

    async def save_session(
        self,
        session_id: str,
        summary: str,
        task_description: str = "",
        key_decisions: Optional[List[str]] = None,
        agent_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """会话结束时保存检查点

        Args:
        session_id: 会话 ID
        summary: 会话摘要（LLM 生成或手动提供）
        task_description: 任务描述
        key_decisions: 重要决策列表
        agent_results: Agent 执行结果快照
        """
        checkpoint = SessionCheckpoint(
            session_id=session_id,
            user_id=self.user_id,
            summary=summary,
            key_decisions=key_decisions or [],
            task_description=task_description,
            agent_results_snapshot=agent_results or {},
        )
        try:
            await self._store.save_checkpoint(checkpoint)
            log.info(f"保存会话检查点: session={session_id}")
        except Exception as e:
            log.warning(f"保存会话检查点失败: {e}")
