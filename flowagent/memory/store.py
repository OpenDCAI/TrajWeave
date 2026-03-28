"""
记忆存储后端 - DeerFlow 2.0 对齐

提供可插拔的持久化存储，默认使用 SQLite（轻量级、零配置）。
DeerFlow 2.0 支持 SQLite（本地）和 PostgreSQL（生产），
我们通过 MemoryBackend 协议实现相同的灵活性。

使用示例：
store = SQLiteMemoryStore("~/.flowagent/memory.db")
await store.initialize()

# 保存/加载用户画像
await store.save_profile(profile)
profile = await store.load_profile("user_123")

# 记录知识事实
await store.add_fact(fact)
facts = await store.get_facts("user_123", category="preference")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from flowagent.logger import get_logger
from flowagent.memory.models import KnowledgeFact, SessionCheckpoint, UserProfile

log = get_logger(__name__)

# SQLite 建表 SQL
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_profiles (
user_id TEXT PRIMARY KEY,
preferences TEXT DEFAULT '{}',
technical_background TEXT DEFAULT '[]',
work_patterns TEXT DEFAULT '{}',
created_at TEXT,
updated_at TEXT
);

CREATE TABLE IF NOT EXISTS session_checkpoints (
session_id TEXT PRIMARY KEY,
user_id TEXT NOT NULL,
summary TEXT DEFAULT '',
key_decisions TEXT DEFAULT '[]',
task_description TEXT DEFAULT '',
agent_results_snapshot TEXT DEFAULT '{}',
created_at TEXT
);

CREATE TABLE IF NOT EXISTS knowledge_facts (
fact_id TEXT PRIMARY KEY,
user_id TEXT NOT NULL,
category TEXT DEFAULT 'context',
content TEXT NOT NULL,
confidence REAL DEFAULT 1.0,
source_session TEXT DEFAULT '',
created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON knowledge_facts(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_category ON knowledge_facts(user_id, category);
CREATE INDEX IF NOT EXISTS idx_checkpoints_user ON session_checkpoints(user_id);
"""


@runtime_checkable
class MemoryBackend(Protocol):
    """记忆存储后端协议 - 可插拔的持久化接口

    实现此协议即可替换底层存储（SQLite → PostgreSQL, Redis 等）。
    """

    async def initialize(self) -> None:
        """初始化存储（创建表等）"""
        ...

    async def save_profile(self, profile: UserProfile) -> None:
        ...

    async def load_profile(self, user_id: str) -> Optional[UserProfile]:
        ...

    async def save_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        ...

    async def load_latest_checkpoint(self, user_id: str) -> Optional[SessionCheckpoint]:
        ...

    async def get_recent_sessions(
        self, user_id: str, limit: int = 5
    ) -> List[SessionCheckpoint]:
        ...

    async def add_fact(self, fact: KnowledgeFact) -> None:
        ...

    async def get_facts(
        self, user_id: str, category: Optional[str] = None, limit: int = 50
    ) -> List[KnowledgeFact]:
        ...


class SQLiteMemoryStore:
    """SQLite 记忆存储 - 默认的本地存储后端

    特点：
    - 零配置，开箱即用
    - 异步 I/O（通过 aiosqlite）
    - 数据保存在 ~/.flowagent/memory.db
    """

    def __init__(self, db_path: str = "~/.flowagent/memory.db"):
        self._db_path = Path(db_path).expanduser()
        self._initialized = False

    async def initialize(self) -> None:
        """创建数据库和表结构"""
        try:
            import aiosqlite
        except ImportError:
            raise ImportError(
                "aiosqlite is required for SQLiteMemoryStore. "
                "Install it with: pip install flowagent[memory]"
            )

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(str(self._db_path)) as db:
            await db.executescript(_SCHEMA_SQL)
            await db.commit()

        self._initialized = True
        log.info(f"SQLiteMemoryStore 初始化完成: {self._db_path}")

    def _get_db(self):
        import aiosqlite
        return aiosqlite.connect(str(self._db_path))

    async def save_profile(self, profile: UserProfile) -> None:
        async with self._get_db() as db:
            await db.execute(
                """INSERT OR REPLACE INTO user_profiles
                (user_id, preferences, technical_background, work_patterns,
                created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                profile.user_id,
                json.dumps(profile.preferences, ensure_ascii=False),
                json.dumps(profile.technical_background, ensure_ascii=False),
                json.dumps(profile.work_patterns, ensure_ascii=False),
                profile.created_at,
                profile.updated_at,
                ),
            )
            await db.commit()
            log.info(f"保存用户画像: {profile.user_id}")

    async def load_profile(self, user_id: str) -> Optional[UserProfile]:
        async with self._get_db() as db:
            db.row_factory = _dict_factory
            cursor = await db.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return UserProfile(
                user_id=row["user_id"],
                preferences=json.loads(row["preferences"]),
                technical_background=json.loads(row["technical_background"]),
                work_patterns=json.loads(row["work_patterns"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    async def save_checkpoint(self, checkpoint: SessionCheckpoint) -> None:
        async with self._get_db() as db:
            await db.execute(
                """INSERT OR REPLACE INTO session_checkpoints
                (session_id, user_id, summary, key_decisions,
                task_description, agent_results_snapshot, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                checkpoint.session_id,
                checkpoint.user_id,
                checkpoint.summary,
                json.dumps(checkpoint.key_decisions, ensure_ascii=False),
                checkpoint.task_description,
                json.dumps(checkpoint.agent_results_snapshot, ensure_ascii=False),
                checkpoint.created_at,
                ),
            )
            await db.commit()
            log.info(f"保存会话检查点: {checkpoint.session_id}")

    async def load_latest_checkpoint(
        self, user_id: str
    ) -> Optional[SessionCheckpoint]:
        async with self._get_db() as db:
            db.row_factory = _dict_factory
            cursor = await db.execute(
                """SELECT * FROM session_checkpoints
                WHERE user_id = ? ORDER BY created_at DESC LIMIT 1""",
                (user_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return SessionCheckpoint(
                session_id=row["session_id"],
                user_id=row["user_id"],
                summary=row["summary"],
                key_decisions=json.loads(row["key_decisions"]),
                task_description=row["task_description"],
                agent_results_snapshot=json.loads(row["agent_results_snapshot"]),
                created_at=row["created_at"],
            )

    async def get_recent_sessions(
        self, user_id: str, limit: int = 5
    ) -> List[SessionCheckpoint]:
        async with self._get_db() as db:
            db.row_factory = _dict_factory
            cursor = await db.execute(
                """SELECT * FROM session_checkpoints
                WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit),
            )
            rows = await cursor.fetchall()
            return [
                SessionCheckpoint(
                    session_id=r["session_id"],
                    user_id=r["user_id"],
                    summary=r["summary"],
                    key_decisions=json.loads(r["key_decisions"]),
                    task_description=r["task_description"],
                    agent_results_snapshot=json.loads(r["agent_results_snapshot"]),
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    async def add_fact(self, fact: KnowledgeFact) -> None:
        async with self._get_db() as db:
            await db.execute(
                """INSERT OR REPLACE INTO knowledge_facts
                (fact_id, user_id, category, content, confidence,
                source_session, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                fact.fact_id,
                fact.user_id,
                fact.category,
                fact.content,
                fact.confidence,
                fact.source_session,
                fact.created_at,
                ),
            )
            await db.commit()

    async def get_facts(
        self,
        user_id: str,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> List[KnowledgeFact]:
        async with self._get_db() as db:
            db.row_factory = _dict_factory
            if category:
                cursor = await db.execute(
                    """SELECT * FROM knowledge_facts
                    WHERE user_id = ? AND category = ?
                    ORDER BY created_at DESC LIMIT ?""",
                    (user_id, category, limit),
                )
            else:
                cursor = await db.execute(
                    """SELECT * FROM knowledge_facts
                    WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
                    (user_id, limit),
                )
            rows = await cursor.fetchall()
            return [KnowledgeFact.from_dict(r) for r in rows]


def _dict_factory(cursor, row):
    """SQLite row → dict 转换器"""
    columns = [col[0] for col in cursor.description]
    return dict(zip(columns, row))
