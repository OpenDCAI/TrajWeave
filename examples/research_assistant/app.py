"""
FlowAgent Multi-Agent Demo - Chainlit 多场景前端

支持 6 个场景：
1. 深度研究 - 多子Agent协作研究
2. 代码审查 - Markdown Skill 驱动的代码审查
3. 智能对话 - 带跨会话记忆的多轮对话
4. 图片理解 - VLM 视觉语言模型
5. 任务规划 - Plan-Execute 自动规划执行
6. 数据分析 - Workflow 并行分析

运行: cd examples/research_assistant && python -m chainlit run app.py
"""
import os
import sys
from pathlib import Path
from uuid import uuid4

# API config
os.environ.setdefault("DF_API_KEY", "sk-seXl8SEklmjOj1pMQVbNiDTaLpGmaqbgK3XqCVQdl7Yc8Asx")
os.environ.setdefault("DF_API_URL", "http://123.129.219.111:3000/v1")

# Ensure .files dir exists (Windows compat)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".files"), exist_ok=True)

import chainlit as cl

# Paths
project_root = Path(__file__).resolve().parent.parent.parent
local_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(local_dir))

from flowagent.memory.store import SQLiteMemoryStore
from flowagent.memory.context import MemoryContext
from flowagent.context.budget import ContextBudgetManager
from flowagent.observability.tracer import get_tracer
from flowagent.llm.router import ModelRouter
from flowagent.logger import get_logger

log = get_logger(__name__)

MEMORY_DB_PATH = os.path.expanduser("~/.flowagent/demo_memory.db")
DEFAULT_MODEL = os.getenv("DF_MODEL", "gpt-4o")

# ============================================================
# Progress Reporter
# ============================================================

class ChainlitProgressReporter:
    def __init__(self):
        self._steps = {}

    async def on_agent_start(self, agent_name, input_text):
        step = cl.Step(name=agent_name, type="llm")
        step.input = input_text[:300]
        await step.__aenter__()
        self._steps[agent_name] = step

    async def on_agent_end(self, agent_name, output_text):
        step = self._steps.pop(agent_name, None)
        if step:
            step.output = output_text[:500] if output_text else "Done"
            await step.__aexit__(None, None, None)

    async def on_tool_call(self, tool_name, input_data, output_data):
        async with cl.Step(name=f"Tool: {tool_name}", type="tool") as step:
            step.input = input_data[:200]
            step.output = output_data[:300]

    async def close_all(self):
        for name in list(self._steps.keys()):
            await self.on_agent_end(name, "Interrupted")


# ============================================================
# Chat Profiles - 6 scenarios
# ============================================================

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="Deep Research",
            markdown_description="Multi-agent collaborative research with web search, analysis, and report writing.",
            starters=[
                cl.Starter(label="Research Transformers", message="Research the latest advances in Transformer architecture"),
                cl.Starter(label="Research AI Agents", message="Research the development trends of AI Agents in 2025-2026"),
            ],
        ),
        cl.ChatProfile(
            name="Code Review",
            markdown_description="Professional code review powered by Markdown Skills. Paste your code and get detailed feedback.",
            starters=[
                cl.Starter(label="Review Python code", message="Please review this code:\n```python\ndef fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"),
            ],
        ),
        cl.ChatProfile(
            name="Smart Chat",
            markdown_description="Multi-turn conversation with cross-session memory. I remember your preferences!",
            starters=[
                cl.Starter(label="Introduce yourself", message="Hi! Tell me about yourself and what you can do."),
            ],
        ),
        cl.ChatProfile(
            name="Image Analysis",
            markdown_description="Upload an image and I'll analyze it using Vision Language Model (VLM).",
            starters=[
                cl.Starter(label="How to upload", message="How do I upload an image for analysis?"),
            ],
        ),
        cl.ChatProfile(
            name="Task Planner",
            markdown_description="Break down complex tasks into actionable steps using Plan-Execute pattern.",
            starters=[
                cl.Starter(label="Plan a project", message="Help me plan building a REST API with user authentication"),
            ],
        ),
        cl.ChatProfile(
            name="Data Analysis",
            markdown_description="Analyze data with parallel agent workflows. Describe your analysis needs.",
            starters=[
                cl.Starter(label="Analyze trends", message="Analyze the key trends in the AI industry for 2025-2026"),
            ],
        ),
    ]


# ============================================================
# Welcome messages per profile
# ============================================================

WELCOME_MESSAGES = {
    "Deep Research": (
        "# Deep Research Assistant\n\n"
        "I'm a **multi-agent research team**:\n"
        "- Orchestrator decomposes your request\n"
        "- Researcher searches for information\n"
        "- Analyzer synthesizes findings\n"
        "- Writer composes the report\n\n"
        "Tell me a topic to research!"
    ),
    "Code Review": (
        "# Code Review Assistant\n\n"
        "I'll review your code for:\n"
        "- Correctness and potential bugs\n"
        "- Performance issues\n"
        "- Code style and best practices\n"
        "- Security vulnerabilities\n\n"
        "Paste your code and I'll analyze it!"
    ),
    "Smart Chat": (
        "# Smart Chat\n\n"
        "I'm a conversational assistant with **cross-session memory**.\n\n"
        "I remember your preferences and past conversations. "
        "The more we chat, the better I understand you!\n\n"
        "What would you like to talk about?"
    ),
    "Image Analysis": (
        "# Image Analysis\n\n"
        "Upload an image and I'll analyze it using **Vision Language Model**.\n\n"
        "I can:\n"
        "- Describe image content\n"
        "- Answer questions about images\n"
        "- Extract text (OCR)\n\n"
        "Upload an image or drag & drop to start!"
    ),
    "Task Planner": (
        "# Task Planner\n\n"
        "I use **Plan-Execute** pattern to break down complex tasks:\n\n"
        "1. Analyze your request\n"
        "2. Create a step-by-step plan\n"
        "3. Execute each step\n"
        "4. Adapt the plan if needed\n\n"
        "Describe a complex task and I'll plan it out!"
    ),
    "Data Analysis": (
        "# Data Analysis\n\n"
        "I use **parallel agent workflows** to analyze data from multiple angles simultaneously:\n"
        "- Trend analysis\n"
        "- Pattern recognition\n"
        "- Statistical insights\n\n"
        "Describe what you'd like to analyze!"
    ),
}


# ============================================================
# Shared initialization
# ============================================================

async def init_session():
    """Initialize shared components for all profiles."""
    api_url = os.getenv("DF_API_URL", "")
    api_key = os.getenv("DF_API_KEY", "")

    if not api_url or api_url == "test" or not api_key or api_key == "test":
        await cl.Message(content=(
            "## Configuration Required\n\n"
            "```bash\n"
            "export DF_API_URL=\"https://api.openai.com/v1\"\n"
            "export DF_API_KEY=\"sk-your-key\"\n"
            "```\n\nThen restart: `python -m chainlit run app.py`"
        )).send()
        return False

    session_id = str(uuid4())
    user_id = "default_user"

    # Memory
    try:
        store = SQLiteMemoryStore(MEMORY_DB_PATH)
        await store.initialize()
        memory_ctx = MemoryContext(store, user_id)
        await memory_ctx.load()
    except Exception as e:
        log.warning(f"Memory init failed: {e}")
        memory_ctx = None

    cl.user_session.set("session_id", session_id)
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("memory_ctx", memory_ctx)
    cl.user_session.set("budget_manager", ContextBudgetManager(max_tokens=120_000))
    cl.user_session.set("tracer", get_tracer(langsmith_enabled=False))
    cl.user_session.set("model_router", ModelRouter({"default": DEFAULT_MODEL}))

    return True


# ============================================================
# Lifecycle handlers
# ============================================================

@cl.on_chat_start
async def on_chat_start():
    profile = cl.user_session.get("chat_profile")
    if not await init_session():
        return

    welcome = WELCOME_MESSAGES.get(profile, "Welcome! How can I help you?")
    await cl.Message(content=welcome).send()

    # Show memory for profiles that use it
    memory_ctx = cl.user_session.get("memory_ctx")
    if memory_ctx and profile in ("Smart Chat", "Deep Research"):
        from memory_ui import display_memory_panel
        await display_memory_panel(memory_ctx)


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    if not session_id:
        await cl.Message(content="Session not initialized. Please refresh.").send()
        return

    profile = cl.user_session.get("chat_profile")
    memory_ctx = cl.user_session.get("memory_ctx")
    reporter = ChainlitProgressReporter()

    try:
        if profile == "Deep Research":
            await handle_research(message, memory_ctx, reporter)
        elif profile == "Code Review":
            await handle_code_review(message, memory_ctx, reporter)
        elif profile == "Smart Chat":
            await handle_smart_chat(message, memory_ctx, reporter)
        elif profile == "Image Analysis":
            await handle_image_analysis(message, memory_ctx, reporter)
        elif profile == "Task Planner":
            await handle_task_planner(message, memory_ctx, reporter)
        elif profile == "Data Analysis":
            await handle_data_analysis(message, memory_ctx, reporter)
        else:
            await handle_smart_chat(message, memory_ctx, reporter)
    except Exception as e:
        log.error(f"Handler failed: {e}", exc_info=True)
        await reporter.close_all()
        await cl.Message(content=f"## Error\n\n{str(e)}").send()


# ============================================================
# Profile handlers
# ============================================================

async def handle_research(message, memory_ctx, reporter):
    from agents import run_research
    result = await run_research(
        query=message.content,
        user_id=cl.user_session.get("user_id"),
        session_id=cl.user_session.get("session_id"),
        memory_ctx=memory_ctx,
        model_name=DEFAULT_MODEL,
        reporter=reporter,
    )
    report = result.get("writer", result.get("analyzer", "No results."))
    if isinstance(report, dict):
        report = str(report)
    msg = cl.Message(content="")
    for i in range(0, len(report), 30):
        await msg.stream_token(report[i:i+30])
    await msg.send()

    if memory_ctx:
        await memory_ctx.learn("context", f"Researched: {message.content}")
        from memory_ui import display_memory_panel
        await display_memory_panel(memory_ctx)


async def handle_code_review(message, memory_ctx, reporter):
    from agents import run_code_review
    await reporter.on_agent_start("Code Reviewer", message.content[:200])
    result = await run_code_review(
        code=message.content,
        model_name=DEFAULT_MODEL,
        memory_ctx=memory_ctx,
    )
    await reporter.on_agent_end("Code Reviewer", result[:300])
    await cl.Message(content=result).send()


async def handle_smart_chat(message, memory_ctx, reporter):
    from agents import run_smart_chat
    history = cl.user_session.get("chat_history") or []
    await reporter.on_agent_start("Assistant", message.content[:200])
    result = await run_smart_chat(
        query=message.content,
        history=history,
        memory_ctx=memory_ctx,
        model_name=DEFAULT_MODEL,
    )
    await reporter.on_agent_end("Assistant", result[:300])

    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": result})
    cl.user_session.set("chat_history", history)

    await cl.Message(content=result).send()

    if memory_ctx:
        from memory_ui import display_memory_panel
        await display_memory_panel(memory_ctx)


async def handle_image_analysis(message, memory_ctx, reporter):
    from agents import run_image_analysis
    images = [f for f in (message.elements or []) if "image" in (f.mime or "")]
    if not images:
        await cl.Message(content="Please upload an image to analyze. You can drag & drop or click the attachment button.").send()
        return

    image_path = images[0].path
    await reporter.on_agent_start("VLM Analyzer", f"Analyzing: {images[0].name}")
    result = await run_image_analysis(
        image_path=image_path,
        query=message.content or "Describe this image in detail.",
        model_name=DEFAULT_MODEL,
        memory_ctx=memory_ctx,
    )
    await reporter.on_agent_end("VLM Analyzer", result[:300])

    elements = [cl.Image(name="uploaded", path=image_path, display="inline")]
    await cl.Message(content=result, elements=elements).send()


async def handle_task_planner(message, memory_ctx, reporter):
    from agents import run_task_planner
    await reporter.on_agent_start("Planner", message.content[:200])
    result = await run_task_planner(
        task=message.content,
        model_name=DEFAULT_MODEL,
        memory_ctx=memory_ctx,
        reporter=reporter,
    )
    await reporter.on_agent_end("Planner", result[:300])
    await cl.Message(content=result).send()


async def handle_data_analysis(message, memory_ctx, reporter):
    from agents import run_data_analysis
    result = await run_data_analysis(
        query=message.content,
        model_name=DEFAULT_MODEL,
        memory_ctx=memory_ctx,
        reporter=reporter,
    )
    report = result.get("report", str(result))
    if isinstance(report, dict):
        report = str(report)
    msg = cl.Message(content="")
    for i in range(0, len(report), 30):
        await msg.stream_token(report[i:i+30])
    await msg.send()


@cl.on_chat_end
async def on_chat_end():
    memory_ctx = cl.user_session.get("memory_ctx")
    session_id = cl.user_session.get("session_id")
    if memory_ctx and session_id:
        try:
            await memory_ctx.save_session(session_id=session_id, summary="Session ended")
        except Exception:
            pass
