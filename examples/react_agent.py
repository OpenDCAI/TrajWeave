"""
ReAct Agent Example - Demonstrates agent with tool use

This example shows how to create an agent that uses tools with the ReAct pattern.
The agent can perform calculations using provided tools.
"""

from flowagent import BaseAgent, register
from flowagent.core import ReactConfig
from flowagent.state import MainState, MainRequest


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
        """Define tools for this agent"""
        
        def add(a: float, b: float) -> float:
            """Add two numbers together"""
            return a + b
        
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers together"""
            return a * b
        
        def subtract(a: float, b: float) -> float:
            """Subtract b from a"""
            return a - b
        
        def divide(a: float, b: float) -> float:
            """Divide a by b"""
            if b == 0:
                return "Error: Division by zero"
            return a / b
        
        return [add, multiply, subtract, divide]


if __name__ == "__main__":
    # Create agent configuration with ReAct mode
    config = ReactConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        max_iterations=5
    )

    # Initialize agent
    agent = CalculatorAgent(config=config)

    # Create state with request
    state = MainState(
        request=MainRequest(
            target="Calculate (15 + 7) * 3 - 10",
            model="gpt-4o-mini"
        )
    )

    # Execute agent
    print("Executing CalculatorAgent with ReAct pattern...")
    result = agent.execute(state)
    print(f"\nResult: {result}")
