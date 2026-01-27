"""
Plan Agent 完整使用示例

展示如何使用PlanSolveStrategy和PlanExecuteStrategy
"""
import asyncio
import os
import re
from pathlib import Path
from langchain_core.tools import tool


# ============ 定义工具 ============

@tool
def read_file(path: str) -> str:
    """读取文件内容"""
    try:
        p = Path(path)
        with p.open('r', encoding='utf-8') as f:
            return f.read()[:1000]  # 限制长度
    except Exception as e:
        return f"读取失败: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """写入文件内容"""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8') as f:
            f.write(content)
        return f"已写入: {path}"
    except Exception as e:
        return f"写入失败: {e}"


@tool
def list_dir(path: str = ".") -> str:
    """列出目录下的文件和子目录"""
    try:
        p = Path(path)
        if not p.exists():
            return f"目录不存在: {path}"
        if not p.is_dir():
            return f"不是目录: {path}"
        items = []
        for child in p.iterdir():
            suffix = "/" if child.is_dir() else ""
            items.append(child.name + suffix)
        return "\n".join(items[:200])
    except Exception as e:
        return f"列目录失败: {e}"


@tool
def search_code(pattern: str, directory: str = ".") -> str:
    """搜索代码"""
    try:
        root = Path(directory)
        if not root.exists():
            return f"目录不存在: {directory}"

        exts = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}
        regex = re.compile(pattern)
        hits = []

        for file_path in root.rglob("*"):
            if file_path.suffix.lower() not in exts:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{file_path}:{i}: {line.strip()}")
                    if len(hits) >= 20:
                        break
            if len(hits) >= 20:
                break

        return "\n".join(hits) if hits else "未找到匹配"
    except Exception as e:
        return f"搜索失败: {e}"


# ============ 使用示例 ============

async def example_plan_solve():
    """PlanSolve策略示例 - 一次性计划，顺序执行"""
    from flowagent.core.base_agent import BaseAgent
    from flowagent.core.execution_config import ExecutionConfig, ExecutionMode
    from flowagent.state import PlanningState, PlanningRequest

    api_url = os.getenv("DF_API_URL")
    api_key = os.getenv("DF_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_url or api_url == "test":
        raise RuntimeError("Missing DF_API_URL. Put it in .env or set it in the shell.")
    if not api_key or api_key == "test":
        raise RuntimeError("Missing DF_API_KEY (or OPENAI_API_KEY). Put it in .env or set it in the shell.")

    class PlanSolveAgent(BaseAgent):
        @property
        def role_name(self) -> str:
            return "planner"

        @property
        def system_prompt_template_name(self) -> str:
            return "greeter_system"

        @property
        def task_prompt_template_name(self) -> str:
            return "greeter_task"

    # 创建执行配置
    config = ExecutionConfig(
        mode=ExecutionMode.PLAN_SOLVE,
        model_name="gpt-4o-mini",
        chat_api_url=api_url,
        executor_tools=[list_dir, read_file, write_file, search_code],
        max_plan_steps=10,
        tool_mode="required",
        require_plan_approval=False,  # 不需要人工审批
    )

    # 创建Agent
    agent = PlanSolveAgent(execution_config=config)

    # 模拟state（实际使用时从你的应用获取）
    state = PlanningState(
        request=PlanningRequest(
            target="分析 flowagent/core 目录下的代码结构",
            chat_api_url=api_url,
            api_key=api_key,
            model="gpt-4o-mini",
        ),
        original_task="分析 flowagent/core 目录下的代码结构",
        past_steps=[],
        executor_tools=[t.name for t in config.executor_tools],
    )

    # 执行
    result_state = await agent.execute(state)
    print("执行结果:", result_state.agent_results.get("planner", {}))


async def example_plan_execute():
    """PlanExecute策略示例 - 动态计划，支持重规划"""
    from flowagent.core.base_agent import BaseAgent
    from flowagent.core.execution_config import ExecutionConfig, ExecutionMode

    config = ExecutionConfig(
        mode=ExecutionMode.PLAN_EXECUTE,
        model_name="gpt-4",
        chat_api_url="http://localhost:3000/v1",
        executor_tools=[read_file, write_file, search_code],
        max_plan_steps=7,
        max_replanning_rounds=3,
        auto_replan_on_error=True,
    )

    agent = BaseAgent(
        role_name="smart_planner",
        system_prompt="你是一个智能代码助手",
        execution_config=config
    )

    # ... 同上


# ============ 简化API ============

async def plan_and_execute(
    task: str,
    tools: list,
    api_url: str,
    api_key: str,
    model: str = "gpt-4",
    use_replan: bool = True
):
    """
    简化的Plan-and-Execute API

    Args:
        task: 任务描述
        tools: 工具列表
        api_url: API URL
        api_key: API密钥
        model: 模型名称
        use_replan: 是否使用重规划

    Returns:
        执行结果
    """
    from flowagent.core.base_agent import BaseAgent
    from flowagent.core.execution_config import ExecutionConfig, ExecutionMode
    from types import SimpleNamespace

    mode = ExecutionMode.PLAN_EXECUTE if use_replan else ExecutionMode.PLAN_SOLVE

    config = ExecutionConfig(
        mode=mode,
        model_name=model,
        chat_api_url=api_url,
        executor_tools=tools,
        max_plan_steps=7,
    )

    agent = BaseAgent(
        role_name="task_executor",
        system_prompt="你是一个任务执行专家",
        execution_config=config
    )

    state = SimpleNamespace(
        request=SimpleNamespace(
            target=task,
            chat_api_url=api_url,
            api_key=api_key,
            model=model
        ),
        original_task=task,
        past_steps=[]
    )

    return await agent.execute(state)


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_plan_solve())
    #pass
