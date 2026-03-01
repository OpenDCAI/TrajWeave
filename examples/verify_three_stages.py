"""三阶段改动本地验证脚本（不依赖 LLM）

运行：
  D:/Anaconda/envs/flowagent/python.exe examples/verify_three_stages.py
"""
from __future__ import annotations
import asyncio
import tempfile, os


# ============================================================
# Stage 1: Workflow 无缝组合 Agent（add_agent_node + chain）
# ============================================================
def verify_stage1():
    print("=" * 60)
    print("Stage 1: Workflow 无缝组合 Agent")
    print("=" * 60)

    from flowagent.workflow.base import WorkflowBuilder, agent_node
    from flowagent.state import MainState, MainRequest

    # 1) 验证方法存在
    builder = WorkflowBuilder(MainState, name="s1_test")
    assert hasattr(builder, "add_agent_node"), "add_agent_node 方法缺失"
    assert hasattr(builder, "chain"), "chain 方法缺失"
    print("[PASS] add_agent_node / chain 方法存在")

    # 2) 验证 chain 能串联节点并自动 set_entry
    async def node_a(state):
        from flowagent.workflow.base import _ensure_mapping
        ar = _ensure_mapping(state, "agent_results")
        ar["a"] = {"status": "ok", "results": {"val": 1}, "error": None}
        return {"agent_results": ar}

    async def node_b(state):
        from flowagent.workflow.base import _ensure_mapping
        ar = _ensure_mapping(state, "agent_results")
        prev = ar.get("a", {}).get("results", {}).get("val", 0)
        ar["b"] = {"status": "ok", "results": {"val": prev + 1}, "error": None}
        return {"agent_results": ar}

    builder2 = WorkflowBuilder(MainState, name="s1_chain")
    builder2.add_node("a", node_a).add_node("b", node_b).chain("a", "b")
    wf = builder2.compile()
    state = MainState(request=MainRequest(language="zh", target="test"))
    result = asyncio.run(wf.run_async(state))
    ar = result.agent_results if hasattr(result, "agent_results") else result.get("agent_results", {})
    assert ar["b"]["results"]["val"] == 2, f"chain 传递失败: {ar}"
    print("[PASS] chain 串联 + 状态传递正确")

    # 3) 验证工厂函数可从顶层导入
    from flowagent import (
        create_react_agent, create_simple_agent, create_plan_execute_agent,
        create_validation_agent, create_vlm_agent, create_parallel_agent,
    )
    print("[PASS] 6 个工厂函数均可从 flowagent 顶层导入")

    print(">>> Stage 1 全部通过 <<<\n")


# ============================================================
# Stage 2: 可继承的策略与配置
# ============================================================
def verify_stage2():
    print("=" * 60)
    print("Stage 2: 可继承的策略与配置")
    print("=" * 60)

    # 1) ExecutionMode 统一验证
    from flowagent.core.execution_config import ExecutionMode as EM1
    from flowagent.core.configs import ExecutionMode as EM2
    assert EM1 is EM2, "ExecutionMode 未统一，configs.py 和 execution_config.py 指向不同类"
    assert hasattr(EM1, "CUSTOM"), "缺少 CUSTOM 成员"
    assert hasattr(EM1, "GRAPH"), "缺少 GRAPH 成员"
    assert EM1.CUSTOM.value == "custom"
    print("[PASS] ExecutionMode 已统一，包含 GRAPH + CUSTOM")

    # 2) register_strategy 装饰器
    from flowagent.core.strategies import StrategyFactory, ExecutionStrategy

    @StrategyFactory.register_strategy("test_verify")
    class TestVerifyStrategy(ExecutionStrategy):
        async def execute(self, state, **kwargs):
            return {"verified": True}

    assert "test_verify" in StrategyFactory._strategies
    assert StrategyFactory._strategies["test_verify"] is TestVerifyStrategy
    print("[PASS] @register_strategy 装饰器注册成功")

    # 3) ExecutionConfig.extra 字段
    from flowagent.core.execution_config import ExecutionConfig
    cfg = ExecutionConfig(extra={"my_param": 42})
    assert cfg.extra["my_param"] == 42
    print("[PASS] ExecutionConfig.extra 字段可用")

    # 4) dataclass 继承扩展
    from dataclasses import dataclass

    @dataclass
    class CustomConfig(ExecutionConfig):
        my_custom_field: int = 10

    cc = CustomConfig(mode=EM1.CUSTOM, my_custom_field=99)
    assert cc.my_custom_field == 99
    assert cc.mode == EM1.CUSTOM
    print("[PASS] ExecutionConfig 可通过 dataclass 继承扩展")

    # 清理注册
    del StrategyFactory._strategies["test_verify"]

    print(">>> Stage 2 全部通过 <<<\n")


# ============================================================
# Stage 3: Skills 集成（skill.md + 打包 Workflow）
# ============================================================
def verify_stage3():
    print("=" * 60)
    print("Stage 3: Skills 集成")
    print("=" * 60)

    # 1) Skill 新字段
    from flowagent.skills.base import Skill
    s = Skill(
        name="test", description="test",
        execution_mode="react", model_name="gpt-4o",
        steps=[{"name": "step1", "system_prompt": "hi"}],
    )
    assert s.execution_mode == "react"
    assert s.model_name == "gpt-4o"
    assert len(s.steps) == 1
    print("[PASS] Skill 新字段 (execution_mode / model_name / steps) 可用")

    # 2) load_from_markdown
    from flowagent.skills.registry import SkillRegistry

    md_content = """---
name: verify_skill
description: 验证用技能
execution_mode: simple
model_name: test-model
---
# System Prompt
你是一个验证助手，请回答问题。

# User Prompt Template
请处理以下内容：{input}
"""
    path = os.path.join(tempfile.gettempdir(), "verify_skill.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md_content)

    registry = SkillRegistry()
    skill = registry.load_from_markdown(path)
    assert skill.name == "verify_skill"
    assert skill.execution_mode == "simple"
    assert skill.model_name == "test-model"
    assert "验证助手" in skill.system_prompt
    assert "{input}" in skill.user_prompt_template
    os.unlink(path)
    print("[PASS] load_from_markdown 解析正确")

    # 3) load_from_yaml 新字段
    import yaml
    yaml_content = {
        "name": "yaml_skill",
        "description": "yaml test",
        "execution_mode": "react",
        "model_name": "gpt-4",
        "system_prompt": "test prompt",
        "steps": [{"name": "s1", "system_prompt": "step1 prompt"}],
    }
    yaml_path = os.path.join(tempfile.gettempdir(), "verify_skill.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f)

    registry2 = SkillRegistry()
    skill2 = registry2.load_from_yaml(yaml_path)
    assert skill2.execution_mode == "react"
    assert skill2.model_name == "gpt-4"
    assert len(skill2.steps) == 1
    os.unlink(yaml_path)
    print("[PASS] load_from_yaml 支持新字段")

    # 4) SkillExecutor 结构验证（不实际调 LLM）
    from flowagent.skills.executor import SkillExecutor, _MODE_FACTORY_MAP
    assert "simple" in _MODE_FACTORY_MAP
    assert "react" in _MODE_FACTORY_MAP
    executor = SkillExecutor()
    assert hasattr(executor, "_execute_single")
    assert hasattr(executor, "_execute_workflow")
    print("[PASS] SkillExecutor 结构完整 (single + workflow 两条路径)")

    # 5) 顶层导出
    from flowagent import Skill, SkillRegistry, get_skill_registry, SkillExecutor
    print("[PASS] Skills API 可从 flowagent 顶层导入")

    print(">>> Stage 3 全部通过 <<<\n")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    verify_stage1()
    verify_stage2()
    verify_stage3()
    print("=" * 60)
    print("所有三个阶段验证全部通过！")
    print("=" * 60)
