"""
Plan Agent 使用示例

展示如何使用Plan策略实现类似Claude Code的功能：
给一个需求，生成执行计划。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.tools import Tool

from flowagent.logger import get_logger

log = get_logger(__name__)


async def generate_plan_for_task(
    task: str,
    api_url: str,
    api_key: str,
    model: str = "gpt-4",
    available_tools: Optional[List[str]] = None,
    max_steps: int = 7,
) -> Dict[str, Any]:
    """
    为任务生成执行计划（类似Claude Code的Plan模式）

    Args:
        task: 任务描述
        api_url: API URL
        api_key: API密钥
        model: 模型名称
        available_tools: 可用工具列表
        max_steps: 最大步骤数

    Returns:
        包含计划的字典

    Example:
        >>> plan = await generate_plan_for_task(
        ...     task="实现一个用户登录功能",
        ...     api_url="http://localhost:3000/v1",
        ...     api_key="sk-xxx",
        ...     available_tools=["read_file", "write_file", "run_tests"]
        ... )
        >>> print(plan["steps"])
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    # 构建系统提示词
    tools_info = ""
    if available_tools:
        tools_info = f"\n\n可用工具:\n" + "\n".join(f"- {t}" for t in available_tools)

    system_prompt = f"""你是一个任务规划专家，类似Claude Code的Plan模式。

你的职责是分析用户的任务需求，生成一个清晰、可执行的分步计划。

规则:
1. 每个步骤应该是具体的、可执行的操作
2. 步骤之间有清晰的逻辑顺序
3. 步骤描述要具体，包含文件路径、函数名等细节
4. 标注每个步骤需要使用的工具
5. 步骤数量控制在3-{max_steps}步
6. 考虑边界情况和错误处理
{tools_info}"""

    task_prompt = f"""请为以下任务生成执行计划:

任务: {task}

请以JSON格式返回，包含:
- summary: 任务概述（1-2句话）
- steps: 步骤列表，每个步骤包含:
  - description: 步骤描述
  - tool: 使用的工具（如果有）
  - files: 涉及的文件（如果有）
- verification: 如何验证任务完成"""

    # 创建LLM
    llm = ChatOpenAI(
        openai_api_base=api_url,
        openai_api_key=api_key,
        model_name=model,
        temperature=0.7,
    )

    # 定义输出结构
    class PlanStep(BaseModel):
        description: str = Field(description="步骤描述")
        tool: Optional[str] = Field(default=None, description="使用的工具")
        files: List[str] = Field(default_factory=list, description="涉及的文件")

    class PlanOutput(BaseModel):
        summary: str = Field(description="任务概述")
        steps: List[PlanStep] = Field(description="执行步骤")
        verification: str = Field(description="验证方法")

    try:
        structured_llm = llm.with_structured_output(PlanOutput)
        response = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt)
        ])

        return {
            "status": "success",
            "task": task,
            "summary": response.summary,
            "steps": [
                {
                    "index": i + 1,
                    "description": step.description,
                    "tool": step.tool,
                    "files": step.files
                }
                for i, step in enumerate(response.steps[:max_steps])
            ],
            "verification": response.verification,
            "total_steps": len(response.steps[:max_steps])
        }

    except Exception as e:
        log.error(f"生成计划失败: {e}")
        return {
            "status": "error",
            "error": str(e),
            "task": task
        }


def format_plan_markdown(plan: Dict[str, Any]) -> str:
    """将计划格式化为Markdown"""
    if plan.get("status") != "success":
        return f"生成计划失败: {plan.get('error', '未知错误')}"

    lines = [
        f"# 执行计划",
        f"",
        f"## 任务概述",
        f"{plan['summary']}",
        f"",
        f"## 执行步骤",
    ]

    for step in plan["steps"]:
        lines.append(f"")
        lines.append(f"### 步骤 {step['index']}")
        lines.append(f"{step['description']}")
        if step.get("tool"):
            lines.append(f"- **工具**: {step['tool']}")
        if step.get("files"):
            lines.append(f"- **文件**: {', '.join(step['files'])}")

    lines.extend([
        f"",
        f"## 验证方法",
        f"{plan['verification']}",
    ])

    return "\n".join(lines)
