"""
FlowAgent Research Assistant - Chainlit Demo

展示 FlowAgent DeerFlow 2.0 全部能力的端到端 Demo：
- 子 Agent 动态生成 (SubAgentManager)
- 跨会话长期记忆 (MemoryContext + SQLiteMemoryStore)
- 上下文工程 (ContextBudgetManager)
- 可观测性追踪 (FlowAgentTracer)
- 多模型路由 (ModelRouter)

运行方式：
    cd examples/research_assistant
    chainlit run app.py
    # 浏览器打开 http://localhost:8000
"""
import os
import sys
from pathlib import Path
from uuid import uuid4

# API 配置
os.environ["DF_API_KEY"] = "sk-seXl8SEklmjOj1pMQVbNiDTaLpGmaqbgK3XqCVQdl7Yc8Asx"
os.environ["DF_API_URL"] = "http://123.129.219.111:3000/v1"

# 确保 Chainlit .files 目录存在（Windows 兼容）
os.makedirs(os.path.join(os.path.dirname(__file__), ".files"), exist_ok=True)

import chainlit as cl

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Ensure local directory is in path (for agents.py, tools.py, memory_ui.py)
local_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(local_dir))

from flowagent.memory.store import SQLiteMemoryStore
from flowagent.memory.context import MemoryContext
from flowagent.context.budget import ContextBudgetManager
from flowagent.observability.tracer import get_tracer, FlowAgentTracer
from flowagent.llm.router import ModelRouter
from flowagent.logger import get_logger

log = get_logger(__name__)

# Constants
MEMORY_DB_PATH = os.path.expanduser("~/.flowagent/research_assistant_memory.db")
DEFAULT_MODEL = os.getenv("DF_MODEL", "gpt-4o")


# ============================================================
# Chainlit Progress Reporter
# ============================================================

class ChainlitProgressReporter:
    """将 Agent 执行进度映射到 Chainlit Step UI"""

    def __init__(self):
        self._steps = {}

    async def on_agent_start(self, agent_name: str, input_text: str) -> None:
        step = cl.Step(name=agent_name, type="llm")
        step.input = input_text
        await step.__aenter__()
        self._steps[agent_name] = step

    async def on_agent_end(self, agent_name: str, output_text: str) -> None:
        step = self._steps.pop(agent_name, None)
        if step:
            step.output = output_text[:500] if output_text else "Done"
            await step.__aexit__(None, None, None)

    async def on_tool_call(self, tool_name: str, input_data: str, output_data: str) -> None:
        async with cl.Step(name=f"Tool: {tool_name}", type="tool") as step:
            step.input = input_data[:200]
            step.output = output_data[:300]


# ============================================================
# Chainlit Lifecycle
# ============================================================

@cl.on_chat_start
async def on_chat_start():
    """新会话开始时初始化所有组件"""

    # Check API configuration
    api_url = os.getenv("DF_API_URL", "")
    api_key = os.getenv("DF_API_KEY", "")

    if not api_url or api_url == "test" or not api_key or api_key == "test":
        await cl.Message(
            content=(
                "## ⚙️ Configuration Required\n\n"
                "Please set environment variables before starting:\n\n"
                "```bash\n"
                "export DF_API_URL=\"https://api.openai.com/v1\"  # or compatible endpoint\n"
                "export DF_API_KEY=\"sk-your-key-here\"\n"
                "export DF_MODEL=\"gpt-4o\"  # optional\n"
                "```\n\n"
                "Then restart: `chainlit run app.py`"
            )
        ).send()
        return

    session_id = str(uuid4())
    user_id = "default_user"

    # Initialize memory
    try:
        store = SQLiteMemoryStore(MEMORY_DB_PATH)
        await store.initialize()
        memory_ctx = MemoryContext(store, user_id)
        await memory_ctx.load()
    except Exception as e:
        log.warning(f"Memory initialization failed: {e}")
        memory_ctx = None

    # Initialize other components
    budget_manager = ContextBudgetManager(max_tokens=120_000)
    tracer = get_tracer(project_name="research_assistant", langsmith_enabled=False)
    model_router = ModelRouter({
        "reasoning": DEFAULT_MODEL,
        "summarization": DEFAULT_MODEL,
        "default": DEFAULT_MODEL,
    })

    # Store in session
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("memory_ctx", memory_ctx)
    cl.user_session.set("budget_manager", budget_manager)
    cl.user_session.set("tracer", tracer)
    cl.user_session.set("model_router", model_router)
    cl.user_session.set("store", store if memory_ctx else None)

    # Welcome message
    welcome = (
        "# 🔍 Research Assistant\n\n"
        "I'm a **multi-agent research assistant** powered by FlowAgent.\n\n"
        "Tell me a topic and my team will research it for you!\n\n"
        "**My agents:**\n"
        "- 🎯 **Orchestrator** — decomposes your request\n"
        "- 🔎 **Researcher** — searches for information\n"
        "- 📊 **Analyzer** — synthesizes findings\n"
        "- ✍️ **Writer** — composes the final report\n\n"
        "I remember your preferences across sessions. Try asking me something!"
    )
    await cl.Message(content=welcome).send()

    # Show memory panel
    if memory_ctx:
        from memory_ui import display_memory_panel
        await display_memory_panel(memory_ctx)


@cl.on_message
async def on_message(message: cl.Message):
    """处理用户消息，执行研究流程"""

    session_id = cl.user_session.get("session_id")
    if not session_id:
        await cl.Message(content="Session not initialized. Please refresh the page.").send()
        return

    user_id = cl.user_session.get("user_id")
    memory_ctx = cl.user_session.get("memory_ctx")
    query = message.content

    # Create progress reporter
    reporter = ChainlitProgressReporter()

    # Execute research
    try:
        from agents import run_research

        result = await run_research(
            query=query,
            user_id=user_id,
            session_id=session_id,
            memory_ctx=memory_ctx,
            model_name=DEFAULT_MODEL,
            reporter=reporter,
        )

        # Stream the final report
        report = result.get("writer", "")
        if not report or report == "{}":
            report = result.get("analyzer", "No results generated.")

        msg = cl.Message(content="")
        # Simulate streaming
        chunk_size = 20
        for i in range(0, len(report), chunk_size):
            chunk = report[i:i + chunk_size]
            await msg.stream_token(chunk)
        await msg.send()

    except Exception as e:
        log.error(f"Research failed: {e}", exc_info=True)
        await cl.Message(
            content=f"## Error\n\nResearch failed: {str(e)}\n\nPlease check your API configuration and try again."
        ).send()
        # Close any open steps
        for step in list(reporter._steps.values()):
            try:
                step.output = f"Error: {str(e)}"
                await step.__aexit__(None, None, None)
            except Exception:
                pass
        return

    # Save to memory
    if memory_ctx:
        try:
            await memory_ctx.learn("context", f"Researched: {query}")
            await memory_ctx.save_session(
                session_id=session_id,
                summary=f"Research on: {query}",
                task_description=query,
                agent_results={"topics": [query]},
            )
        except Exception as e:
            log.warning(f"Memory save failed: {e}")

        # Update memory panel
        from memory_ui import display_memory_panel
        await display_memory_panel(memory_ctx)


@cl.on_chat_end
async def on_chat_end():
    """会话结束时保存状态"""
    memory_ctx = cl.user_session.get("memory_ctx")
    session_id = cl.user_session.get("session_id")
    if memory_ctx and session_id:
        try:
            await memory_ctx.save_session(
                session_id=session_id,
                summary="Session ended normally",
            )
        except Exception as e:
            log.warning(f"Session save failed: {e}")
