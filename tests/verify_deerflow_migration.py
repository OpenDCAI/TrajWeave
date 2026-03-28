"""
DeerFlow 2.0 迁移 - 全量验证脚本

验证所有 Phase (0~5) 的实现完整性。
运行方式：python tests/verify_deerflow_migration.py
"""

import asyncio
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_phase(phase, description):
    print(f"\n{'='*60}")
    print(f"  {phase}: {description}")
    print(f"{'='*60}")


def print_pass(msg):
    print(f"  PASS: {msg}")


def print_fail(msg, error=""):
    print(f"  FAIL: {msg}")
    if error:
        print(f"         {error}")


# ============================================================
# Phase 0: MainState 扩展 + 依赖更新
# ============================================================
def verify_phase0():
    print_phase("Phase 0", "State extension + dependency update")
    passed = 0
    failed = 0

    # 1. MainState new fields
    try:
        from flowagent.state.base import MainState, MainRequest
        state = MainState()
        assert state.parent_agent_id is None
        assert isinstance(state.sub_agent_results, dict)
        assert isinstance(state.session_id, str) and len(state.session_id) > 0
        assert state.memory_context is None
        assert state.context_budget == 120_000
        assert state.context_summary is None
        print_pass("MainState 6 new fields exist with correct defaults")
        passed += 1
    except Exception as e:
        print_fail("MainState new fields", str(e))
        failed += 1

    # 2. Backward compat - DFState still works
    try:
        from flowagent.state.base import DFState, DFRequest
        df_state = DFState()
        assert hasattr(df_state, "parent_agent_id")
        assert hasattr(df_state, "session_id")
        assert hasattr(df_state, "category")
        print_pass("DFState backward compatible, new fields inherited")
        passed += 1
    except Exception as e:
        print_fail("DFState backward compat", str(e))
        failed += 1

    # 3. pyproject.toml new dependency groups
    try:
        pyproject_path = project_root / "pyproject.toml"
        content = pyproject_path.read_text(encoding="utf-8")
        assert "[project.optional-dependencies]" in content
        assert "memory" in content
        assert "observability" in content
        assert "context" in content
        assert "deerflow" in content
        print_pass("pyproject.toml has 4 new optional dependency groups")
        passed += 1
    except Exception as e:
        print_fail("pyproject.toml dependency groups", str(e))
        failed += 1

    # 4. uuid4 unique session_id
    try:
        from flowagent.state.base import MainState
        s1 = MainState()
        s2 = MainState()
        assert s1.session_id != s2.session_id
        print_pass("session_id auto-generates unique UUID")
        passed += 1
    except Exception as e:
        print_fail("session_id UUID", str(e))
        failed += 1

    print(f"\n  Phase 0 result: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# Phase 1: Sub-Agent dynamic spawning
# ============================================================
def verify_phase1():
    print_phase("Phase 1", "Sub-Agent dynamic spawning system")
    passed = 0
    failed = 0

    # 1. SubAgentConfig
    try:
        from flowagent.core.sub_agent import SubAgentConfig
        config = SubAgentConfig(
            role="researcher",
            system_prompt="You are a researcher",
            tools=["web_search"],
            context="Research transformers",
            timeout_seconds=60,
            max_iterations=5,
            execution_mode="react",
        )
        assert config.role == "researcher"
        assert config.timeout_seconds == 60
        assert config.execution_mode == "react"
        print_pass("SubAgentConfig creation OK")
        passed += 1
    except Exception as e:
        print_fail("SubAgentConfig creation", str(e))
        failed += 1

    # 2. SubAgent state isolation
    try:
        from flowagent.core.sub_agent import SubAgent, SubAgentConfig
        from flowagent.state.base import MainState, MainRequest
        from flowagent.tools.manager import ToolManager

        parent_state = MainState(
            request=MainRequest(target="parent task", model="gpt-4o"),
        )

        config = SubAgentConfig(role="coder", context="child task")
        tm = ToolManager()
        sub = SubAgent(config, parent_state, tm)

        assert len(sub._state.messages) == 0, "Sub-agent messages should be empty"
        assert sub._state.request.target == "child task"
        assert sub._state.parent_agent_id == parent_state.session_id
        assert sub._state.session_id != parent_state.session_id
        print_pass("SubAgent state isolation verified")
        passed += 1
    except Exception as e:
        print_fail("SubAgent state isolation", str(e))
        failed += 1

    # 3. SubAgentManager
    try:
        from flowagent.core.agent_pool import SubAgentManager
        from flowagent.tools.manager import ToolManager
        tm = ToolManager()
        manager = SubAgentManager(tm, max_concurrent=3)
        assert manager.max_concurrent == 3
        assert manager.active_count == 0
        assert manager.completed_count == 0
        print_pass("SubAgentManager creation OK")
        passed += 1
    except Exception as e:
        print_fail("SubAgentManager creation", str(e))
        failed += 1

    # 4. ToolManager.get_subset
    try:
        from flowagent.tools.manager import ToolManager
        from langchain_core.tools import Tool

        tm = ToolManager()
        tool1 = Tool(name="web_search", func=lambda x: x, description="search")
        tool2 = Tool(name="url_fetch", func=lambda x: x, description="fetch")
        tool3 = Tool(name="code_run", func=lambda x: x, description="run code")
        tm.register_post_tool(tool1)
        tm.register_post_tool(tool2)
        tm.register_post_tool(tool3)

        subset = tm.get_subset(["web_search", "url_fetch"])
        assert len(subset.global_post_tools) == 2
        names = {t.name for t in subset.global_post_tools}
        assert names == {"web_search", "url_fetch"}
        print_pass("ToolManager.get_subset correct")
        passed += 1
    except Exception as e:
        print_fail("ToolManager.get_subset", str(e))
        failed += 1

    # 5. ToolManager.get_all_post_tools and list_tool_names
    try:
        all_tools = tm.get_all_post_tools()
        assert len(all_tools) == 3
        names = tm.list_tool_names()
        assert "web_search" in names
        assert "code_run" in names
        print_pass("ToolManager.get_all_post_tools and list_tool_names OK")
        passed += 1
    except Exception as e:
        print_fail("ToolManager new methods", str(e))
        failed += 1

    # 6. BaseAgent.sub_agents property
    try:
        from flowagent.core.base_agent import BaseAgent
        assert hasattr(BaseAgent, "sub_agents")
        print_pass("BaseAgent.sub_agents property exists")
        passed += 1
    except Exception as e:
        print_fail("BaseAgent.sub_agents", str(e))
        failed += 1

    print(f"\n  Phase 1 result: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# Phase 2: Long-term memory
# ============================================================
def verify_phase2():
    print_phase("Phase 2", "Cross-session long-term memory")
    passed = 0
    failed = 0

    # 1. Data models
    try:
        from flowagent.memory.models import UserProfile, SessionCheckpoint, KnowledgeFact
        profile = UserProfile(user_id="test_user")
        profile.update_preference("language", "python")
        profile.add_background("ML engineer")
        assert profile.preferences["language"] == "python"
        assert "ML engineer" in profile.technical_background

        fact = KnowledgeFact(user_id="test_user", category="preference", content="Prefers PyTorch")
        assert fact.category == "preference"

        checkpoint = SessionCheckpoint(session_id="sess1", user_id="test_user", summary="Test summary")
        assert checkpoint.summary == "Test summary"

        d = profile.to_dict()
        restored = UserProfile.from_dict(d)
        assert restored.user_id == "test_user"
        assert restored.preferences["language"] == "python"

        print_pass("Memory data models (UserProfile, SessionCheckpoint, KnowledgeFact) OK")
        passed += 1
    except Exception as e:
        print_fail("Memory data models", str(e))
        failed += 1

    # 2. SQLiteMemoryStore CRUD
    try:
        import aiosqlite

        async def test_store():
            import tempfile
            import os
            from flowagent.memory.store import SQLiteMemoryStore
            from flowagent.memory.models import UserProfile, KnowledgeFact, SessionCheckpoint

            db_path = os.path.join(tempfile.mkdtemp(), "test_memory.db")
            store = SQLiteMemoryStore(db_path)
            await store.initialize()

            profile = UserProfile(user_id="u1")
            profile.update_preference("theme", "dark")
            await store.save_profile(profile)
            loaded = await store.load_profile("u1")
            assert loaded is not None
            assert loaded.preferences["theme"] == "dark"

            fact = KnowledgeFact(user_id="u1", category="preference", content="Likes Python")
            await store.add_fact(fact)
            facts = await store.get_facts("u1")
            assert len(facts) >= 1
            assert facts[0].content == "Likes Python"

            cp = SessionCheckpoint(session_id="s1", user_id="u1", summary="Test session")
            await store.save_checkpoint(cp)
            latest = await store.load_latest_checkpoint("u1")
            assert latest is not None
            assert latest.summary == "Test session"

            sessions = await store.get_recent_sessions("u1", limit=5)
            assert len(sessions) >= 1

            os.unlink(db_path)

        asyncio.run(test_store())
        print_pass("SQLiteMemoryStore full CRUD cycle OK")
        passed += 1
    except ImportError:
        print_fail("SQLiteMemoryStore test skipped (aiosqlite not installed)")
        failed += 1
    except Exception as e:
        print_fail("SQLiteMemoryStore CRUD", str(e))
        traceback.print_exc()
        failed += 1

    # 3. MemoryContext system prompt injection
    try:
        from flowagent.memory.context import MemoryContext
        from flowagent.memory.models import UserProfile, KnowledgeFact

        class FakeStore:
            async def load_profile(self, uid):
                return UserProfile(user_id=uid, preferences={"lang": "zh"})

            async def get_facts(self, uid, limit=50):
                return [KnowledgeFact(user_id=uid, category="context", content="Uses PyTorch")]

            async def get_recent_sessions(self, uid, limit=5):
                return []

            async def add_fact(self, f):
                pass

        async def test_context():
            ctx = MemoryContext(FakeStore(), "test_user")
            await ctx.load()
            assert ctx.is_loaded
            section = ctx.to_system_prompt_section()
            assert len(section) > 0
            return section

        section = asyncio.run(test_context())
        print_pass(f"MemoryContext system prompt injection OK ({len(section)} chars)")
        passed += 1
    except Exception as e:
        print_fail("MemoryContext system prompt injection", str(e))
        failed += 1

    print(f"\n  Phase 2 result: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# Phase 3: Context engineering
# ============================================================
def verify_phase3():
    print_phase("Phase 3", "Context engineering (token budget + compression)")
    passed = 0
    failed = 0

    # 1. ContextBudgetManager basics
    try:
        from flowagent.context.budget import ContextBudgetManager

        budget = ContextBudgetManager(max_tokens=1000, reserve_for_response=100, compression_threshold=0.5)
        assert budget.available_tokens == 900
        assert budget.count_tokens("hello world") > 0

        short_msgs = [type("M", (), {"content": "hi", "type": "human"})()]
        assert not budget.needs_compression(short_msgs)

        long_msgs = [type("M", (), {"content": "x" * 400, "type": "human"})() for _ in range(10)]
        assert budget.needs_compression(long_msgs)

        print_pass("ContextBudgetManager token counting and compression check OK")
        passed += 1
    except Exception as e:
        print_fail("ContextBudgetManager", str(e))
        failed += 1

    # 2. select_messages priority
    try:
        from flowagent.context.budget import ContextBudgetManager
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        budget = ContextBudgetManager(max_tokens=500, reserve_for_response=50)

        msgs = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="old message 1"),
            AIMessage(content="old response 1"),
            HumanMessage(content="latest question"),
            AIMessage(content="latest answer"),
        ]

        selected = budget.select_messages(msgs, summary="Previously discussed A and B")
        assert selected[0].type == "system"
        contents = [str(m.content) for m in selected]
        has_summary = any("Previously" in c or "摘要" in c for c in contents)
        assert has_summary, "Should contain history summary"
        assert any("latest" in str(m.content) for m in selected)

        print_pass("select_messages priority selection OK (system > summary > recent)")
        passed += 1
    except Exception as e:
        print_fail("select_messages", str(e))
        failed += 1

    # 3. ContextCompressor
    try:
        from flowagent.context.compressor import ContextCompressor
        from flowagent.state.base import MainState
        state = MainState()
        compressor = ContextCompressor(state)
        assert compressor is not None

        fallback = ContextCompressor._fallback_summary("a" * 2000)
        assert len(fallback) < 2000

        print_pass("ContextCompressor creation and fallback summary OK")
        passed += 1
    except Exception as e:
        print_fail("ContextCompressor", str(e))
        failed += 1

    # 4. get_usage_stats
    try:
        from flowagent.context.budget import ContextBudgetManager
        budget = ContextBudgetManager(max_tokens=10000)
        msgs = [type("M", (), {"content": "hello world", "type": "human"})()]
        stats = budget.get_usage_stats(msgs)
        assert "used_tokens" in stats
        assert "utilization" in stats
        assert "needs_compression" in stats
        assert stats["message_count"] == 1
        print_pass("get_usage_stats returns correct statistics")
        passed += 1
    except Exception as e:
        print_fail("get_usage_stats", str(e))
        failed += 1

    print(f"\n  Phase 3 result: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# Phase 4A: Markdown skill system
# ============================================================
def verify_phase4a():
    print_phase("Phase 4A", "Markdown skill system")
    passed = 0
    failed = 0

    # 1. MarkdownSkillLoader scan
    try:
        from flowagent.skills.markdown_loader import MarkdownSkillLoader
        loader = MarkdownSkillLoader()
        skills = loader.scan()
        assert "research" in skills, f"Should find research skill, got: {list(skills.keys())}"
        assert "code_review" in skills, f"Should find code_review skill, got: {list(skills.keys())}"
        assert skills["research"]["strategy"] == "react"
        print_pass(f"MarkdownSkillLoader scanned {len(skills)} skills: {list(skills.keys())}")
        passed += 1
    except Exception as e:
        print_fail("MarkdownSkillLoader scan", str(e))
        failed += 1

    # 2. Progressive loading - full content
    try:
        full = loader.load_full("research")
        assert "System Prompt" in full
        print_pass(f"Progressive loading: research full content {len(full)} chars")
        passed += 1
    except Exception as e:
        print_fail("Progressive loading full", str(e))
        failed += 1

    # 3. Section parsing
    try:
        sections = loader.parse_sections("research")
        assert "system_prompt" in sections, f"Should have system_prompt section, got: {list(sections.keys())}"
        assert "user_prompt_template" in sections
        print_pass(f"Section parsing: {list(sections.keys())}")
        passed += 1
    except Exception as e:
        print_fail("Section parsing", str(e))
        failed += 1

    # 4. Lightweight descriptions
    try:
        descriptions = loader.get_skill_descriptions()
        assert len(descriptions) >= 2
        assert all("name" in d and "description" in d for d in descriptions)
        print_pass(f"Lightweight descriptions: {len(descriptions)} skills")
        passed += 1
    except Exception as e:
        print_fail("Lightweight descriptions", str(e))
        failed += 1

    # 5. SkillRegistry.discover_all integration
    try:
        from flowagent.skills.registry import SkillRegistry
        registry = SkillRegistry()
        all_skills = registry.discover_all()
        assert "research" in all_skills
        assert all_skills["research"]["type"] == "markdown"
        descs = registry.get_all_skill_descriptions()
        assert len(descs) >= 2
        print_pass(f"SkillRegistry.discover_all integrated: {len(all_skills)} skills")
        passed += 1
    except Exception as e:
        print_fail("SkillRegistry.discover_all", str(e))
        failed += 1

    print(f"\n  Phase 4A result: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# Phase 4B: LangSmith observability
# ============================================================
def verify_phase4b():
    print_phase("Phase 4B", "LangSmith observability")
    passed = 0
    failed = 0

    # 1. FlowAgentTracer creation
    try:
        from flowagent.observability.tracer import FlowAgentTracer
        tracer = FlowAgentTracer(langsmith_enabled=False)
        assert tracer is not None
        assert not tracer.enabled
        print_pass("FlowAgentTracer creation OK (LangSmith disabled mode)")
        passed += 1
    except Exception as e:
        print_fail("FlowAgentTracer creation", str(e))
        failed += 1

    # 2. trace_agent_run context manager
    try:
        async def test_trace():
            from flowagent.observability.tracer import FlowAgentTracer
            tracer = FlowAgentTracer(langsmith_enabled=False)
            async with tracer.trace_agent_run("TestAgent", run_id="test123") as span:
                assert span["agent_name"] == "TestAgent"
                assert span["run_id"] == "test123"
            return True

        result = asyncio.run(test_trace())
        assert result
        print_pass("trace_agent_run context manager OK")
        passed += 1
    except Exception as e:
        print_fail("trace_agent_run", str(e))
        failed += 1

    # 3. trace_llm_call and trace_tool_call
    try:
        async def test_traces():
            from flowagent.observability.tracer import FlowAgentTracer
            tracer = FlowAgentTracer(langsmith_enabled=False)

            async with tracer.trace_llm_call("gpt-4o", messages=[]) as span:
                assert span["model"] == "gpt-4o"

            async with tracer.trace_tool_call("web_search", {"query": "test"}) as span:
                assert span["tool_name"] == "web_search"

            return True

        assert asyncio.run(test_traces())
        print_pass("trace_llm_call and trace_tool_call OK")
        passed += 1
    except Exception as e:
        print_fail("trace_llm_call / trace_tool_call", str(e))
        failed += 1

    # 4. trace_sub_agent (DeerFlow 2.0 parent-child tracing)
    try:
        async def test_sub_trace():
            from flowagent.observability.tracer import FlowAgentTracer
            tracer = FlowAgentTracer(langsmith_enabled=False)
            async with tracer.trace_sub_agent("MainAgent", "researcher", "abc123") as span:
                assert span["parent"] == "MainAgent"
                assert span["sub_role"] == "researcher"
            return True

        assert asyncio.run(test_sub_trace())
        print_pass("trace_sub_agent parent-child tracing OK")
        passed += 1
    except Exception as e:
        print_fail("trace_sub_agent", str(e))
        failed += 1

    # 5. Global singleton
    try:
        from flowagent.observability.tracer import get_tracer
        t1 = get_tracer(langsmith_enabled=False)
        t2 = get_tracer()
        assert t1 is t2
        print_pass("get_tracer global singleton OK")
        passed += 1
    except Exception as e:
        print_fail("get_tracer singleton", str(e))
        failed += 1

    print(f"\n  Phase 4B result: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# Phase 5: Multi-LLM Provider routing
# ============================================================
def verify_phase5():
    print_phase("Phase 5", "Multi-LLM Provider routing")
    passed = 0
    failed = 0

    # 1. ModelRouter default routing
    try:
        from flowagent.llm.router import ModelRouter
        router = ModelRouter()
        assert router.get_model_name("reasoning") is None
        assert router.get_model_name("default") is None
        print_pass("ModelRouter default routing OK (None when unconfigured)")
        passed += 1
    except Exception as e:
        print_fail("ModelRouter default routing", str(e))
        failed += 1

    # 2. Custom routing config
    try:
        from flowagent.llm.router import ModelRouter
        router = ModelRouter({
            "reasoning": "gpt-4o",
            "summarization": "gpt-4o-mini",
            "code_generation": "claude-sonnet",
            "default": "gpt-4o",
        })
        assert router.get_model_name("reasoning") == "gpt-4o"
        assert router.get_model_name("summarization") == "gpt-4o-mini"
        assert router.get_model_name("code_generation") == "claude-sonnet"
        assert router.get_model_name("unknown_task") == "gpt-4o"
        print_pass("Custom routing config OK")
        passed += 1
    except Exception as e:
        print_fail("Custom routing config", str(e))
        failed += 1

    # 3. Dynamic routing update
    try:
        router.update_routing("reasoning", "deepseek-r1")
        assert router.get_model_name("reasoning") == "deepseek-r1"
        table = router.get_routing_table()
        assert "reasoning" in table
        assert table["reasoning"] == "deepseek-r1"
        print_pass("Dynamic routing update OK")
        passed += 1
    except Exception as e:
        print_fail("Dynamic routing update", str(e))
        failed += 1

    # 4. get_llm creates LLM instance
    try:
        from flowagent.llm.router import ModelRouter
        from flowagent.state.base import MainState, MainRequest

        router = ModelRouter({"summarization": "gpt-4o-mini", "default": "gpt-4o"})
        state = MainState(request=MainRequest(
            chat_api_url="https://api.test.com",
            api_key="test-key",
            model="gpt-4o",
        ))

        llm = router.get_llm("summarization", state)
        assert llm is not None
        print_pass("get_llm creates LLM instance with routed model")
        passed += 1
    except Exception as e:
        print_fail("get_llm", str(e))
        failed += 1

    # 5. Global singleton
    try:
        from flowagent.llm.router import get_model_router
        r1 = get_model_router({"default": "test"})
        r2 = get_model_router()
        assert r1 is r2
        print_pass("get_model_router global singleton OK")
        passed += 1
    except Exception as e:
        print_fail("get_model_router singleton", str(e))
        failed += 1

    print(f"\n  Phase 5 result: {passed} passed, {failed} failed")
    return failed == 0


# ============================================================
# Main
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  FlowAgent <- DeerFlow 2.0 Migration Verification")
    print("=" * 60)

    results = {}
    results["Phase 0"] = verify_phase0()
    results["Phase 1"] = verify_phase1()
    results["Phase 2"] = verify_phase2()
    results["Phase 3"] = verify_phase3()
    results["Phase 4A"] = verify_phase4a()
    results["Phase 4B"] = verify_phase4b()
    results["Phase 5"] = verify_phase5()

    print("\n" + "=" * 60)
    print("  OVERALL RESULTS")
    print("=" * 60)
    all_pass = True
    for phase, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {phase}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  All {len(results)} phases verified successfully!")
    else:
        failed_phases = [p for p, v in results.items() if not v]
        print(f"\n  {len(failed_phases)} phase(s) failed: {', '.join(failed_phases)}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

