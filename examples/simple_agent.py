"""
Simple Agent Example - Demonstrates basic agent creation

This example shows how to create a minimal agent using FlowAgent framework.
The agent uses simple execution mode with system and task prompts.
"""

import asyncio
import os
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
    # Resolve API settings (OpenAI-compatible)
    # - DF_API_URL should be a base URL like: https://api.openai.com/v1
    # - DF_API_KEY is your API key
    api_url = os.getenv("DF_API_URL")
    api_key = os.getenv("DF_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_url or api_url == "test":
        raise RuntimeError(
            "Missing DF_API_URL. Set it to an OpenAI-compatible base URL, e.g. "
            "https://api.openai.com/v1 (or your proxy/local gateway base)."
        )

    if not api_key or api_key == "test":
        raise RuntimeError(
            "Missing DF_API_KEY (or OPENAI_API_KEY). Please export your API key in the environment."
        )

    # Create agent configuration
    config = SimpleConfig(
        model_name="gpt-4o-mini",
        chat_api_url=api_url,
        temperature=0.7
    )

    # Initialize agent
    agent = GreeterAgent(execution_config=config)

    # Create state with request
    state = MainState(
        request=MainRequest(
            target="Say hello to Alice in a friendly way",
            model="gpt-4o-mini",
            chat_api_url=api_url,
            api_key=api_key,
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
