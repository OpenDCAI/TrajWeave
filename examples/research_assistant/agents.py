"""
Research Assistant - Agent 定义与编排逻辑

编排流程：
  1. Orchestrator: 分解研究任务
  2. Researcher: 搜索信息（顺序执行，使用 web_search 工具）
  3. Analyzer + Writer: 并行执行（分析+撰写报告）

展示 FlowAgent DeerFlow 2.0 全部能力：
  - SubAgentManager (spawn / spawn_many)
  - MemoryContext (load / learn / to_system_prompt_section)
  - ContextBudgetManager
  - FlowAgentTracer
  - ModelRouter
  - ToolManager.get_subset
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional, Protocol

from flowagent.core.factory import create_simple_agent
from flowagent.core.sub_agent import SubAgentConfig
from flowagent.core.agent_pool import SubAgentManager
from flowagent.state.base import MainState, MainRequest
from flowagent.tools.manager import ToolManager
from flowagent.memory.context import MemoryContext
from flowagent.context.budget import ContextBudgetManager
from flowagent.observability.tracer import get_tracer
from flowagent.llm.router import ModelRouter
from flowagent.logger import get_logger

log = get_logger(__name__)


# ============================================================
# Progress Reporter Protocol
# ============================================================

class ProgressReporter(Protocol):
    """Agent 执行进度回调协议"""
    async def on_agent_start(self, agent_name: str, input_text: str) -> None: ...
    async def on_agent_end(self, agent_name: str, output_text: str) -> None: ...
    async def on_tool_call(self, tool_name: str, input_data: str, output_data: str) -> None: ...


class NullReporter:
    """空实现，用于无 UI 时"""
    async def on_agent_start(self, agent_name: str, input_text: str) -> None:
        log.info(f"[{agent_name}] Start: {input_text[:100]}")
    async def on_agent_end(self, agent_name: str, output_text: str) -> None:
        log.info(f"[{agent_name}] End: {output_text[:100]}")
    async def on_tool_call(self, tool_name: str, input_data: str, output_data: str) -> None:
        log.info(f"[Tool:{tool_name}] {input_data[:50]} -> {output_data[:50]}")


# ============================================================
# System Prompts
# ============================================================

ORCHESTRATOR_PROMPT = """You are a Research Orchestrator. Your job is to analyze the user's research request and create a brief research plan.

{memory_section}

Output a concise research plan with 2-3 key aspects to investigate. Be specific about what to search for.
Keep your response under 200 words. Write in the same language as the user's query."""

RESEARCHER_PROMPT = """You are a Research Specialist. Your task is to search for information on the given topic using the web_search tool.

{memory_section}

Instructions:
1. Use the web_search tool to find relevant information
2. Search for 1-2 different aspects of the topic
3. Compile your findings into a structured summary
4. Include key facts, dates, and sources when available

Write your findings in the same language as the original query."""

ANALYZER_PROMPT = """You are an Analysis Specialist. Your task is to synthesize research findings and identify key themes, patterns, and insights.

{memory_section}

You will receive research data collected by the Researcher agent. Your job is to:
1. Identify the main themes and patterns
2. Highlight the most important findings
3. Note any contradictions or gaps
4. Provide a confidence assessment

Write your analysis in the same language as the research context."""

WRITER_PROMPT = """You are a Report Writer. Your task is to compose a well-structured research report based on the analysis provided.

{memory_section}

Structure your report as:
## Research Report: [Topic]
### Executive Summary (2-3 sentences)
### Key Findings (bullet points)
### Detailed Analysis (2-3 paragraphs)
### Conclusion

Write in the same language as the provided context. Make the report informative and well-organized."""


# ============================================================
# Tool Manager Setup
# ============================================================

def create_tool_manager() -> ToolManager:
    """创建并注册所有工具"""
    from tools import web_search, analyze_data

    tm = ToolManager()
    tm.register_post_tool(web_search)
    tm.register_post_tool(analyze_data)
    return tm


# ============================================================
# Sub-Agent Config Builder
# ============================================================

def build_sub_agent_configs(
    query: str,
    researcher_context: str,
    memory_section: str,
    model_name: str,
) -> Dict[str, SubAgentConfig]:
    """构建各子 Agent 的配置"""
    return {
        "researcher": SubAgentConfig(
            role="researcher",
            system_prompt=RESEARCHER_PROMPT.format(memory_section=memory_section),
            tools=["web_search"],
            context=f"Research the following topic thoroughly: {query}",
            timeout_seconds=120,
            execution_mode="react",
            model_name=model_name,
        ),
        "analyzer": SubAgentConfig(
            role="analyzer",
            system_prompt=ANALYZER_PROMPT.format(memory_section=memory_section),
            tools=["analyze_data"],
            context=f"Analyze these research findings:\n\n{researcher_context}",
            timeout_seconds=120,
            execution_mode="react",
            model_name=model_name,
        ),
        "writer": SubAgentConfig(
            role="writer",
            system_prompt=WRITER_PROMPT.format(memory_section=memory_section),
            tools=[],
            context=f"Write a research report based on:\n\nQuery: {query}\n\nFindings:\n{researcher_context}",
            timeout_seconds=120,
            execution_mode="simple",
            model_name=model_name,
        ),
    }


# ============================================================
# Main Orchestration
# ============================================================

async def run_research(
    query: str,
    user_id: str,
    session_id: str,
    memory_ctx: MemoryContext,
    model_name: str = "gpt-4o",
    reporter: Optional[ProgressReporter] = None,
) -> Dict[str, Any]:
    """执行完整的研究流程

    编排步骤：
      1. Orchestrator 分解任务
      2. Researcher 顺序执行（需要 web_search）
      3. Analyzer + Writer 并行执行

    Args:
        query: 用户研究请求
        user_id: 用户 ID
        session_id: 会话 ID
        memory_ctx: 记忆上下文
        model_name: LLM 模型名称
        reporter: 进度报告回调

    Returns:
        包含各 Agent 结果的字典
    """
    if reporter is None:
        reporter = NullReporter()

    tracer = get_tracer()
    results = {}

    # -- 获取记忆注入段 --
    memory_section = ""
    if memory_ctx and memory_ctx.is_loaded:
        section = memory_ctx.to_system_prompt_section()
        if section:
            memory_section = f"\n\n{section}"

    # -- 初始化工具管理器 --
    tool_manager = create_tool_manager()

    # -- 创建基础 State --
    state = MainState(
        request=MainRequest(
            chat_api_url=os.getenv("DF_API_URL", ""),
            api_key=os.getenv("DF_API_KEY", ""),
            model=model_name,
            target=query,
            language="zh" if any('\u4e00' <= c <= '\u9fff' for c in query) else "en",
        ),
        session_id=session_id,
        memory_context=memory_ctx,
    )

    # ===== Phase 1: Orchestrator =====
    async with tracer.trace_agent_run("Orchestrator", run_id=session_id):
        await reporter.on_agent_start("Orchestrator", query)

        orchestrator = create_simple_agent(
            role_name="orchestrator",
            system_prompt=ORCHESTRATOR_PROMPT.format(memory_section=memory_section),
            model_name=model_name,
            parser_type="text",
        )
        orch_state = MainState(
            request=MainRequest(
                chat_api_url=os.getenv("DF_API_URL", ""),
                api_key=os.getenv("DF_API_KEY", ""),
                model=model_name,
                target=query,
            ),
        )
        await orchestrator.execute(orch_state)
        plan = orch_state.agent_results.get("orchestrator", {})
        plan_text = str(plan.get("raw", plan.get("results", str(plan))))
        results["plan"] = plan_text

        await reporter.on_agent_end("Orchestrator", plan_text[:300])

    # ===== Phase 2: Researcher (顺序执行) =====
    async with tracer.trace_sub_agent("Orchestrator", "researcher", session_id):
        await reporter.on_agent_start("Researcher", f"Searching: {query}")

        sub_manager = SubAgentManager(tool_manager, max_concurrent=3)
        configs = build_sub_agent_configs(query, "", memory_section, model_name)

        researcher_id = await sub_manager.spawn(configs["researcher"], state)
        researcher_result = await sub_manager.get_result(researcher_id, timeout=120)

        researcher_text = ""
        if researcher_result and researcher_result.get("status") == "completed":
            r = researcher_result.get("result", {})
            researcher_text = str(r.get("sub_researcher", r))
        else:
            researcher_text = f"Research on '{query}' completed with limited results."

        results["researcher"] = researcher_text
        await reporter.on_agent_end("Researcher", researcher_text[:300])

    # ===== Phase 3: Analyzer + Writer (并行执行) =====
    configs = build_sub_agent_configs(query, researcher_text, memory_section, model_name)

    # Spawn analyzer and writer in parallel
    async with tracer.trace_sub_agent("Orchestrator", "analyzer", session_id):
        await reporter.on_agent_start("Analyzer", "Analyzing findings...")

    async with tracer.trace_sub_agent("Orchestrator", "writer", session_id):
        await reporter.on_agent_start("Writer", "Composing report...")

    parallel_results = await sub_manager.spawn_many(
        [configs["analyzer"], configs["writer"]],
        state,
    )

    # Collect results
    for pr in parallel_results:
        role = pr.get("role", "unknown")
        if pr.get("status") == "completed":
            r = pr.get("result", {})
            text = str(r.get(f"sub_{role}", r))
        else:
            text = f"{role} completed with limited results."
        results[role] = text

    await reporter.on_agent_end("Analyzer", results.get("analyzer", "")[:300])
    await reporter.on_agent_end("Writer", results.get("writer", "")[:300])

    # ===== Memory: 记录本次研究 =====
    if memory_ctx:
        await memory_ctx.learn("context", f"Researched: {query}")

    log.info(f"Research completed for query: {query[:50]}")
    return results


# ============================================================
# Scene 2: Code Review (uses SkillExecutor + code_review.md)
# ============================================================

async def run_code_review(
    code: str,
    model_name: str = "gpt-4o",
    memory_ctx: Optional[MemoryContext] = None,
) -> str:
    """使用 Markdown Skill 执行代码审查"""
    memory_section = ""
    if memory_ctx and memory_ctx.is_loaded:
        section = memory_ctx.to_system_prompt_section()
        if section:
            memory_section = section

    system_prompt = (
        "You are a senior code reviewer. Review the code for:\n"
        "1. Correctness and bugs\n"
        "2. Performance issues\n"
        "3. Code style and best practices\n"
        "4. Security vulnerabilities\n\n"
        "Provide a structured review with:\n"
        "## Code Review Report\n"
        "### Overall Rating: [Good/Fair/Needs Work]\n"
        "### Issues Found\n"
        "### Suggestions\n"
        "### Highlights (good parts)\n\n"
        "Write in the same language as the code comments or variable names.\n"
        f"{memory_section}"
    )

    agent = create_simple_agent(
        role_name="code_reviewer",
        system_prompt=system_prompt,
        model_name=model_name,
        parser_type="text",
    )

    state = MainState(
        request=MainRequest(
            chat_api_url=os.getenv("DF_API_URL", ""),
            api_key=os.getenv("DF_API_KEY", ""),
            model=model_name,
            target=f"Please review this code:\n\n{code}",
        ),
    )

    await agent.execute(state)
    result = state.agent_results.get("code_reviewer", {})
    return str(result.get("raw", result.get("results", str(result))))


# ============================================================
# Scene 3: Smart Chat with Memory
# ============================================================

async def run_smart_chat(
    query: str,
    history: list = None,
    memory_ctx: Optional[MemoryContext] = None,
    model_name: str = "gpt-4o",
) -> str:
    """带记忆的多轮对话"""
    memory_section = ""
    if memory_ctx and memory_ctx.is_loaded:
        section = memory_ctx.to_system_prompt_section()
        if section:
            memory_section = f"\n\n{section}"

    system_prompt = (
        "You are a helpful, friendly AI assistant with long-term memory. "
        "You remember user preferences and past conversations.\n\n"
        "Be concise but thorough. Match the user's language.\n"
        f"{memory_section}"
    )

    # Build conversation context from history
    context_parts = []
    if history:
        for msg in history[-10:]:  # Last 10 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_parts.append(f"[{role}]: {content}")

    full_query = query
    if context_parts:
        conversation = "\n".join(context_parts)
        full_query = f"Previous conversation:\n{conversation}\n\nUser's new message: {query}"

    agent = create_simple_agent(
        role_name="chat_assistant",
        system_prompt=system_prompt,
        model_name=model_name,
        parser_type="text",
    )

    state = MainState(
        request=MainRequest(
            chat_api_url=os.getenv("DF_API_URL", ""),
            api_key=os.getenv("DF_API_KEY", ""),
            model=model_name,
            target=full_query,
        ),
    )

    await agent.execute(state)
    result = state.agent_results.get("chat_assistant", {})
    response = str(result.get("raw", result.get("results", str(result))))

    # Learn from interaction
    if memory_ctx:
        if any(kw in query.lower() for kw in ["i prefer", "i like", "i use", "i'm a", "i am a", "我喜欢", "我用", "我是"]):
            await memory_ctx.learn("preference", f"User said: {query[:100]}")

    return response


# ============================================================
# Scene 4: Image Analysis (VLM)
# ============================================================

async def run_image_analysis(
    image_path: str,
    query: str = "Describe this image in detail.",
    model_name: str = "gpt-4o",
    memory_ctx: Optional[MemoryContext] = None,
) -> str:
    """使用 VLM 分析图片"""
    import base64
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Determine mime type
        ext = Path(image_path).suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
        mime_type = mime_map.get(ext, "image/png")

        llm = ChatOpenAI(
            openai_api_base=os.getenv("DF_API_URL", ""),
            openai_api_key=os.getenv("DF_API_KEY", ""),
            model_name=model_name,
            temperature=0.0,
            max_tokens=2000,
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
            ]
        )

        response = await llm.ainvoke([message])
        return response.content

    except Exception as e:
        log.error(f"Image analysis failed: {e}", exc_info=True)
        return f"Image analysis failed: {str(e)}\n\nThis feature requires a model that supports vision (e.g., gpt-4o, gpt-4-vision)."


# ============================================================
# Scene 5: Task Planner (Plan-Execute)
# ============================================================

async def run_task_planner(
    task: str,
    model_name: str = "gpt-4o",
    memory_ctx: Optional[MemoryContext] = None,
    reporter: Optional[ProgressReporter] = None,
) -> str:
    """使用 Plan-Execute 模式规划和执行任务"""
    if reporter is None:
        reporter = NullReporter()

    memory_section = ""
    if memory_ctx and memory_ctx.is_loaded:
        section = memory_ctx.to_system_prompt_section()
        if section:
            memory_section = f"\n\n{section}"

    # Step 1: Plan
    planner_prompt = (
        "You are an expert task planner. Break down the user's request into clear, actionable steps.\n\n"
        "Output format:\n"
        "## Task Plan: [Task Title]\n\n"
        "### Steps:\n"
        "1. **Step name**: Description\n"
        "2. **Step name**: Description\n"
        "...\n\n"
        "### Estimated complexity: [Low/Medium/High]\n"
        "### Key considerations:\n"
        "- point 1\n"
        "- point 2\n\n"
        "Write in the same language as the user's request.\n"
        f"{memory_section}"
    )

    planner = create_simple_agent(
        role_name="planner",
        system_prompt=planner_prompt,
        model_name=model_name,
        parser_type="text",
    )

    state = MainState(
        request=MainRequest(
            chat_api_url=os.getenv("DF_API_URL", ""),
            api_key=os.getenv("DF_API_KEY", ""),
            model=model_name,
            target=task,
        ),
    )

    await reporter.on_agent_start("Planning", task[:200])
    await planner.execute(state)
    plan_result = state.agent_results.get("planner", {})
    plan_text = str(plan_result.get("raw", plan_result.get("results", str(plan_result))))
    await reporter.on_agent_end("Planning", plan_text[:300])

    # Step 2: Execute/Elaborate
    executor_prompt = (
        "You are a task execution specialist. Given a plan, provide detailed guidance "
        "for executing each step, including specific commands, code snippets, or resources needed.\n\n"
        "For each step:\n"
        "- Provide concrete implementation details\n"
        "- Include example code or commands where relevant\n"
        "- Note potential pitfalls and solutions\n\n"
        "Write in the same language as the plan.\n"
        f"{memory_section}"
    )

    executor = create_simple_agent(
        role_name="executor",
        system_prompt=executor_prompt,
        model_name=model_name,
        parser_type="text",
    )

    exec_state = MainState(
        request=MainRequest(
            chat_api_url=os.getenv("DF_API_URL", ""),
            api_key=os.getenv("DF_API_KEY", ""),
            model=model_name,
            target=f"Execute this plan:\n\n{plan_text}\n\nOriginal task: {task}",
        ),
    )

    await reporter.on_agent_start("Executing", "Elaborating plan steps...")
    await executor.execute(exec_state)
    exec_result = exec_state.agent_results.get("executor", {})
    exec_text = str(exec_result.get("raw", exec_result.get("results", str(exec_result))))
    await reporter.on_agent_end("Executing", exec_text[:300])

    return f"{plan_text}\n\n---\n\n## Execution Details\n\n{exec_text}"


# ============================================================
# Scene 6: Data Analysis (Parallel Sub-Agents)
# ============================================================

async def run_data_analysis(
    query: str,
    model_name: str = "gpt-4o",
    memory_ctx: Optional[MemoryContext] = None,
    reporter: Optional[ProgressReporter] = None,
) -> Dict[str, Any]:
    """使用并行子Agent进行多维度数据分析"""
    if reporter is None:
        reporter = NullReporter()

    memory_section = ""
    if memory_ctx and memory_ctx.is_loaded:
        section = memory_ctx.to_system_prompt_section()
        if section:
            memory_section = f"\n\n{section}"

    tool_manager = create_tool_manager()

    state = MainState(
        request=MainRequest(
            chat_api_url=os.getenv("DF_API_URL", ""),
            api_key=os.getenv("DF_API_KEY", ""),
            model=model_name,
            target=query,
        ),
    )

    # Define 3 parallel analysis sub-agents
    configs = [
        SubAgentConfig(
            role="trend_analyst",
            system_prompt=(
                "You are a trend analyst. Identify and describe the key trends "
                "related to the given topic. Focus on: timeline, growth patterns, "
                "and future projections. Write in the user's language.\n"
                f"{memory_section}"
            ),
            tools=[],
            context=f"Analyze trends for: {query}",
            timeout_seconds=90,
            execution_mode="simple",
            model_name=model_name,
        ),
        SubAgentConfig(
            role="impact_analyst",
            system_prompt=(
                "You are an impact analyst. Analyze the practical impact and "
                "implications of the given topic. Focus on: industry effects, "
                "user impact, and economic considerations. Write in the user's language.\n"
                f"{memory_section}"
            ),
            tools=[],
            context=f"Analyze impact of: {query}",
            timeout_seconds=90,
            execution_mode="simple",
            model_name=model_name,
        ),
        SubAgentConfig(
            role="risk_analyst",
            system_prompt=(
                "You are a risk analyst. Identify risks, challenges, and opportunities "
                "related to the given topic. Focus on: potential pitfalls, mitigation "
                "strategies, and competitive landscape. Write in the user's language.\n"
                f"{memory_section}"
            ),
            tools=[],
            context=f"Analyze risks and opportunities for: {query}",
            timeout_seconds=90,
            execution_mode="simple",
            model_name=model_name,
        ),
    ]

    await reporter.on_agent_start("Trend Analyst", "Analyzing trends...")
    await reporter.on_agent_start("Impact Analyst", "Analyzing impact...")
    await reporter.on_agent_start("Risk Analyst", "Analyzing risks...")

    sub_manager = SubAgentManager(tool_manager, max_concurrent=3)
    parallel_results = await sub_manager.spawn_many(configs, state)

    results = {}
    for pr in parallel_results:
        role = pr.get("role", "unknown")
        if pr.get("status") == "completed":
            r = pr.get("result", {})
            text = str(r.get(f"sub_{role}", r))
        else:
            text = f"{role}: analysis completed with limited results."
        results[role] = text
        await reporter.on_agent_end(
            role.replace("_", " ").title(),
            text[:300]
        )

    # Synthesize report
    synthesizer = create_simple_agent(
        role_name="synthesizer",
        system_prompt=(
            "You are a data analysis synthesizer. Combine the findings from multiple "
            "analysts into a comprehensive report.\n\n"
            "Format:\n"
            "## Data Analysis Report: [Topic]\n"
            "### Key Trends\n"
            "### Impact Assessment\n"
            "### Risks & Opportunities\n"
            "### Recommendations\n\n"
            "Write in the user's language.\n"
            f"{memory_section}"
        ),
        model_name=model_name,
        parser_type="text",
    )

    synth_state = MainState(
        request=MainRequest(
            chat_api_url=os.getenv("DF_API_URL", ""),
            api_key=os.getenv("DF_API_KEY", ""),
            model=model_name,
            target=(
                f"Synthesize these analysis results for '{query}':\n\n"
                f"Trend Analysis:\n{results.get('trend_analyst', 'N/A')}\n\n"
                f"Impact Analysis:\n{results.get('impact_analyst', 'N/A')}\n\n"
                f"Risk Analysis:\n{results.get('risk_analyst', 'N/A')}"
            ),
        ),
    )

    await reporter.on_agent_start("Synthesizer", "Combining analysis...")
    await synthesizer.execute(synth_state)
    synth_result = synth_state.agent_results.get("synthesizer", {})
    report = str(synth_result.get("raw", synth_result.get("results", str(synth_result))))
    await reporter.on_agent_end("Synthesizer", report[:300])

    results["report"] = report
    return results
