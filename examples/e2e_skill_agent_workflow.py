"""端到端 LLM 集成测试: Skill → Agent → Workflow 完整流程

使用前设置环境变量（PowerShell）:
    $env:DF_API_URL = "http://172.96.160.199:3000/v1"
    $env:DF_API_KEY = "sk-xxx"
"""
import asyncio
import os
import sys
import tempfile

# 确保 API URL 带 /v1
api_url = os.getenv("DF_API_URL", "")
api_key = os.getenv("DF_API_KEY", "")
if not api_url or not api_key or api_key == "test":
    print("ERROR: 请先设置 DF_API_URL 和 DF_API_KEY 环境变量")
    print('  PowerShell: $env:DF_API_URL = "http://172.96.160.199:3000/v1"')
    print('  PowerShell: $env:DF_API_KEY = "sk-xxx"')
    sys.exit(1)

from flowagent import (
    create_simple_agent, Skill, SkillRegistry, SkillExecutor,
)
from flowagent.state import MainState, MainRequest
from flowagent.workflow.base import WorkflowBuilder

passed, failed = [], []


def report(name: str, ok: bool, detail: str = ""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    (passed if ok else failed).append(name)


async def test_1_single_agent():
    """测试 1: 单个 Agent 直接调用 LLM"""
    print("\n" + "=" * 60)
    print("Test 1: 单 Agent 调用 LLM")
    print("=" * 60)

    agent = create_simple_agent(
        role_name="greeter",
        system_prompt="你是一个友好的助手，用一句话回复。",
    )
    state = MainState(request=MainRequest(target="用中文说你好", language="zh"))
    result = await agent.execute(state)

    agent_out = state.agent_results.get("greeter", {})
    has_result = bool(agent_out.get("results"))
    no_error = "error" not in str(agent_out.get("results", "")).lower() or agent_out.get("results", {}).get("error") is None
    ok = has_result and no_error
    report("单 Agent LLM 调用", ok, f"结果: {agent_out.get('results', {})}")


async def test_2_skill_single_step():
    """测试 2: 单步 Skill → Agent → LLM"""
    print("\n" + "=" * 60)
    print("Test 2: 单步 Skill 执行 (Skill → Agent)")
    print("=" * 60)

    skill = Skill(
        name="translator",
        description="翻译技能",
        system_prompt="你是一个翻译专家，将用户输入翻译成英文，只输出翻译结果。",
        user_prompt_template="{input}",
        execution_mode="simple",
    )

    executor = SkillExecutor()
    result = await executor.execute(skill, {"input": "今天天气真好"})

    has_result = bool(result)
    report("单步 Skill 执行", has_result, f"结果: {result}")


async def test_3_skill_multi_step():
    """测试 3: 多步 Skill → Workflow (chain 多个 Agent)"""
    print("\n" + "=" * 60)
    print("Test 3: 多步 Skill 执行 (Skill → Workflow)")
    print("=" * 60)

    skill = Skill(
        name="write_and_review",
        description="先写作再审校",
        system_prompt="你是助手。",
        execution_mode="simple",
        user_prompt_template="{input}",
        steps=[
            {
                "name": "writer",
                "system_prompt": "你是一个写作助手，根据主题写一段50字以内的短文。",
                "execution_mode": "simple",
            },
            {
                "name": "reviewer",
                "system_prompt": "你是一个审校员，对上一步的内容给出一句话评价。",
                "execution_mode": "simple",
            },
        ],
    )

    executor = SkillExecutor()
    result = await executor.execute(skill, {"input": "春天"})

    has_writer = "writer" in result
    has_reviewer = "reviewer" in result
    ok = has_writer and has_reviewer
    report("多步 Skill Workflow", ok, f"包含节点: {list(result.keys())}")
    if has_writer:
        print(f"    writer 输出: {result['writer']}")
    if has_reviewer:
        print(f"    reviewer 输出: {result['reviewer']}")


async def test_4_skill_from_markdown():
    """测试 4: 从 Markdown 加载 Skill 并执行"""
    print("\n" + "=" * 60)
    print("Test 4: Markdown Skill 加载 + 执行")
    print("=" * 60)

    md_content = """---
name: md_poet
description: 从Markdown加载的诗人技能
execution_mode: simple
---
# System Prompt
你是一个诗人，用一句古诗回复用户的主题。只输出古诗，不要解释。

# User Prompt Template
主题：{input}
"""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
    tmp.write(md_content)
    tmp.close()

    try:
        registry = SkillRegistry()
        skill = registry.load_from_markdown(tmp.name)
        report("Markdown 解析", skill.name == "md_poet", f"name={skill.name}")

        executor = SkillExecutor()
        result = await executor.execute(skill, {"input": "月亮"})
        has_result = bool(result)
        report("Markdown Skill 执行", has_result, f"结果: {result}")
    finally:
        os.unlink(tmp.name)


async def test_5_workflow_chain_agents():
    """测试 5: WorkflowBuilder 直接编排多个 Agent"""
    print("\n" + "=" * 60)
    print("Test 5: WorkflowBuilder chain 编排")
    print("=" * 60)

    agent_a = create_simple_agent(role_name="idea", system_prompt="你是创意助手，给出一个10字以内的创意点子。")
    agent_b = create_simple_agent(role_name="slogan", system_prompt="你是广告文案，根据上下文写一句10字以内的广告语。")

    builder = WorkflowBuilder(MainState, name="chain_test")
    builder.add_agent_node("idea", agent_a)
    builder.add_agent_node("slogan", agent_b)
    builder.chain("idea", "slogan")
    workflow = builder.compile()

    state = MainState(request=MainRequest(target="咖啡店", language="zh"))
    final = await workflow.run_async(state)

    results = final.agent_results if hasattr(final, "agent_results") else final.get("agent_results", {})
    has_idea = "idea" in results
    has_slogan = "slogan" in results
    ok = has_idea and has_slogan
    report("Workflow chain 编排", ok, f"节点: {list(results.keys())}")
    if has_idea:
        print(f"    idea 输出: {results['idea']}")
    if has_slogan:
        print(f"    slogan 输出: {results['slogan']}")


async def main():
    print("FlowAgent 端到端 LLM 集成测试")
    print(f"API URL: {api_url}")
    print(f"Model: gpt-4o")

    await test_1_single_agent()
    await test_2_skill_single_step()
    await test_3_skill_multi_step()
    await test_4_skill_from_markdown()
    await test_5_workflow_chain_agents()

    print("\n" + "=" * 60)
    print(f"结果: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"失败项: {failed}")
    else:
        print("全部通过!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
