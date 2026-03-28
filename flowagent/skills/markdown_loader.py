"""
Markdown 技能加载器 - DeerFlow 2.0 对齐

DeerFlow 2.0 的关键创新：用 Markdown 文件定义 Skill。
好处：
1. 非开发者也能扩展 Agent 能力（只需写 Markdown）
2. 渐进加载 — 先加载元数据，选中后才加载全文（节省上下文）
3. 版本控制友好（纯文本文件）
4. 可读性好（Markdown 自身就是文档）

Markdown Skill 格式：

---
name: research
description: 深度研究某个主题
tools_required: [web_search, url_fetch]
strategy: react
max_iterations: 15
---
# System Prompt
你是一个专业研究员...

# User Prompt Template
请研究以下主题：{input}

# Process
1. 理解研究问题
2. 搜索一手资料
...

使用示例：
loader = MarkdownSkillLoader("skills/library")
skills_meta = loader.scan()  # 只加载元数据
full = loader.load_full("research") # 选中后加载全文
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from flowagent.logger import get_logger

log = get_logger(__name__)


class MarkdownSkillLoader:
    """​Markdown 技能加载器 - 支持渐进式加载

    Attributes:
    skills_dir: 技能文件目录
    """

    def __init__(self, skills_dir: str = ""):
        if skills_dir:
            self._dir = Path(skills_dir)
        else:
            # 默认使用 flowagent/skills/library
            self._dir = Path(__file__).parent / "library"
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._full_content_cache: Dict[str, str] = {}

    def scan(self) -> Dict[str, Dict[str, Any]]:
        """扫描所有 Markdown 技能文件，只加载 YAML frontmatter（渐进加载 Stage 1）

        这是轻量级操作 — 只解析文件头部的元数据，不加载完整内容。
        用于向 LLM 展示可用技能列表。

        Returns:
        {skill_name: metadata_dict} 字典
        """
        if not self._dir.exists():
            log.info(f"技能目录不存在: {self._dir}")
            return {}

        count = 0
        for md_file in self._dir.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                if not content.startswith("---"):
                    continue

                parts = content.split("---", 2)
                if len(parts) < 3:
                    continue

                frontmatter = yaml.safe_load(parts[1]) or {}
                name = frontmatter.get("name", md_file.stem)

                # 元数据 + 文件路径 + Token 粗估
                frontmatter["_file"] = str(md_file)
                frontmatter["_token_estimate"] = len(parts[2]) // 4
                frontmatter.setdefault("description", "")
                frontmatter.setdefault("tools_required", [])
                frontmatter.setdefault("strategy", "react")
                frontmatter.setdefault("max_iterations", 10)

                self._metadata_cache[name] = frontmatter
                count += 1

            except Exception as e:
                log.warning(f"解析技能文件失败 {md_file}: {e}")

        log.info(f"扫描到 {count} 个 Markdown 技能")
        return dict(self._metadata_cache)

    def load_full(self, skill_name: str) -> str:
        """加载指定技能的完整 Markdown 内容（渐进加载 Stage 2）

        只在技能被选中执行时调用，避免不必要的上下文占用。

        Args:
        skill_name: 技能名称

        Returns:
        完整的 Markdown 内容
        """
        if skill_name in self._full_content_cache:
            return self._full_content_cache[skill_name]

        meta = self._metadata_cache.get(skill_name)
        if meta is None:
            raise KeyError(f"技能 '{skill_name}' 未找到，请先调用 scan()")

        file_path = meta["_file"]
        content = Path(file_path).read_text(encoding="utf-8")
        self._full_content_cache[skill_name] = content

        log.info(f"加载技能完整内容: {skill_name} ({len(content)} chars)")
        return content

    def get_skill_descriptions(self) -> List[Dict[str, Any]]:
        """获取所有技能的轻量级描述（供 LLM 选择技能用）

        Returns:
        [{"name": ..., "description": ..., "tools": ..., "token_estimate": ...}, ...]
        """
        return [
            {
                "name": name,
                "description": meta.get("description", ""),
                "tools": meta.get("tools_required", []),
                "strategy": meta.get("strategy", "react"),
                "token_estimate": meta.get("_token_estimate", 0),
            }
            for name, meta in self._metadata_cache.items()
        ]

    def parse_sections(self, skill_name: str) -> Dict[str, str]:
        """解析技能文件的各个 section

        从 Markdown 的 `# 标题` 分段中提取：
        - system_prompt
        - user_prompt_template
        - process
        - output_format
        - examples

        Returns:
        {section_name: content} 字典
        """
        content = self.load_full(skill_name)

        # 跳过 YAML frontmatter
        parts = content.split("---", 2)
        body = parts[2] if len(parts) >= 3 else content

        sections: Dict[str, str] = {}
        current_section = None
        current_lines: List[str] = []

        for line in body.strip().split("\n"):
            if line.startswith("# "):
                if current_section:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = line[2:].strip().lower().replace(" ", "_")
                current_lines = []
            else:
                current_lines.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_lines).strip()

        return sections

    @property
    def available_skills(self) -> List[str]:
        """已扫描到的技能名列表"""
        return list(self._metadata_cache.keys())
