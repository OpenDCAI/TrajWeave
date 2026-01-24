"""
Simple Agent Example - Demonstrates basic agent creation

This example shows how to create a minimal agent using FlowAgent framework.
The agent uses simple execution mode with system and task prompts.
"""

import asyncio
from flowagent import BaseAgent, register, SimpleConfig
from flowagent.state import MainState, MainRequest


@register("greeter")
class GreeterAgent(BaseAgent):
    """A simple agent that greets users"""

    @property
    def role_name(self) -> str:
        return "Greeter"

    @property
    def system_prompt_template_name(self) -> str:
        return "greeter_system"

    @property
    def task_prompt_template_name(self) -> str:
        return "greeter_task"


async def main():
    """Main async function to run the example"""
    # Create agent configuration
    config = SimpleConfig(
        model="gpt-4o-mini",
        temperature=0.7
    )

    # Initialize agent
    agent = GreeterAgent(config=config)

    # Create state with request
    state = MainState(
        request=MainRequest(
            target="Say hello to Alice in a friendly way",
            model="gpt-4o-mini"
        )
    )

    # Execute agent
    print("Executing GreeterAgent...")
    result_state = await agent.execute(state)

    # Extract result from state
    result = result_state.agent_results.get("Greeter", {})
    print(f"\nResult: {result}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
