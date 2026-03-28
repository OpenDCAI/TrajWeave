"""
记忆数据模型 - DeerFlow 2.0 对齐

定义跨会话持久化的数据结构：
- UserProfile: 用户画像（偏好、技术背景、工作模式）
- SessionCheckpoint: 会话检查点（状态快照 + 摘要）
- KnowledgeFact: 从交互中提取的知识事实
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class UserProfile:
    """用户画像 - 跨会话积累的用户信息

    DeerFlow 2.0 模式：Agent 通过多次交互逐步学习用户的偏好、
    技术水平和工作习惯，提供越来越个性化的服务。

    Attributes:
    user_id: 用户唯一标识
    preferences: 用户偏好，如 {"language": "python", "style": "concise"}
    technical_background: 技术背景标签，如 ["ML engineer", "uses PyTorch"]
    work_patterns: 工作模式，如 {"typical_tasks": ["code review"]}
    """
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    technical_background: List[str] = field(default_factory=list)
    work_patterns: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def update_preference(self, key: str, value: Any) -> None:
        """更新单个偏好"""
        self.preferences[key] = value
        self.updated_at = datetime.utcnow().isoformat()

    def add_background(self, tag: str) -> None:
        """添加技术背景标签（去重）"""
        if tag not in self.technical_background:
            self.technical_background.append(tag)
            self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "technical_background": self.technical_background,
            "work_patterns": self.work_patterns,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SessionCheckpoint:
    """会话检查点 - 保存会话状态用于恢复

    Attributes:
    session_id: 会话唯一标识
    user_id: 关联的用户标识
    summary: LLM 生成的会话摘要
    key_decisions: 重要决策列表
    task_description: 任务描述
    agent_results_snapshot: agent_results 的快照
    """
    session_id: str
    user_id: str
    summary: str = ""
    key_decisions: List[str] = field(default_factory=list)
    task_description: str = ""
    agent_results_snapshot: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "summary": self.summary,
            "key_decisions": self.key_decisions,
            "task_description": self.task_description,
            "agent_results_snapshot": self.agent_results_snapshot,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionCheckpoint":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class KnowledgeFact:
    """知识事实 - 从交互中提取的离散知识

    DeerFlow 2.0 模式：Agent 在交互过程中主动提取和记录
    有价值的信息，形成可复用的知识库。

    Attributes:
    fact_id: 事实唯一标识
    user_id: 关联的用户标识
    category: 分类 - "preference" | "context" | "correction" | "pattern"
    content: 事实内容
    confidence: 置信度 0.0-1.0
    source_session: 来源会话 ID
    """
    fact_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    category: str = "context"  # "preference" | "context" | "correction" | "pattern"
    content: str = ""
    confidence: float = 1.0
    source_session: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "user_id": self.user_id,
            "category": self.category,
            "content": self.content,
            "confidence": self.confidence,
            "source_session": self.source_session,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeFact":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
