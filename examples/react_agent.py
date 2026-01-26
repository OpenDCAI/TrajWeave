"""
ReAct Agent Example - Demonstrates agent with tool use

This example shows how to create an agent that uses tools with the ReAct pattern.
The agent can perform calculations using provided tools.
"""

import asyncio
import os

from flowagent import BaseAgent, register
from flowagent.core import ReactConfig
from flowagent.state import MainState, MainRequest
from flowagent.tools.manager import ToolManager
from langchain_core.tools import tool


@register("calculator")
class CalculatorAgent(BaseAgent):
    """Agent that can perform calculations using tools"""

    @property
    def role_name(self) -> str:
        return "Calculator"

    @property
    def system_prompt_template_name(self) -> str:
        return "calculator_system"

    @property
    def task_prompt_template_name(self) -> str:
        return "calculator_task"

    def get_tools(self, state: MainState):
        """Define LangChain tools for this agent"""

        @tool
        def add(a: float, b: float) -> float:
            """Add two numbers together"""
            return a + b

        @tool
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers together"""
            return a * b

        @tool
        def subtract(a: float, b: float) -> float:
            """Subtract b from a"""
            return a - b

        @tool
        def divide(a: float, b: float) -> float:
            """Divide a by b"""
            if b == 0:
                raise ValueError("Division by zero")
            return a / b

        return [add, multiply, subtract, divide]


async def main():
    api_url = os.getenv("DF_API_URL")
    api_key = os.getenv("DF_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_url or api_url == "test":
        raise RuntimeError("Missing DF_API_URL. Put it in .env or set it in the shell.")
    if not api_key or api_key == "test":
        raise RuntimeError("Missing DF_API_KEY (or OPENAI_API_KEY). Put it in .env or set it in the shell.")

    # ReAct(tool-calling) mode
    def result_must_be_number(content: str, parsed_result: dict):
        v = parsed_result.get("result") if isinstance(parsed_result, dict) else None
        if isinstance(v, (int, float)):
            return True, None
        return False, "JSON 字段 result 必须是 number 且不可为 null"

    config = ReactConfig(
        model_name="gpt-4o-mini",
        temperature=0.0,
        chat_api_url=api_url,
        tool_mode="auto",
        validators=[result_must_be_number],
    )

    tool_manager = ToolManager()
    agent = CalculatorAgent(tool_manager=tool_manager, execution_config=config)

    state = MainState(
        request=MainRequest(
            target="Calculate (15 + 7) * 3 - 10",
            model="gpt-4o-mini",
            chat_api_url=api_url,
            api_key=api_key,
        )
    )

    # Register tools as post-tools for this role
    for t in agent.get_tools(state):
        tool_manager.register_post_tool(t, role=agent.role_name)

    print("Executing CalculatorAgent with ReAct pattern...")
    result_state = await agent.execute(state)
    result = result_state.agent_results.get(agent.role_name.lower(), {})
    print(f"\nResult: {result}")
    return result


if __name__ == "__main__":
    asyncio.run(main())
