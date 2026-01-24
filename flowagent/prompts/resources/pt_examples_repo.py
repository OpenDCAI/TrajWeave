"""
Prompt templates for example agents
"""

from flowagent.prompts.repository import PromptsRepository


class ExamplesPromptsRepo(PromptsRepository):
    """Prompt templates for example agents"""

    def __init__(self):
        super().__init__()
        self._init_prompts()

    def _init_prompts(self):
        """Initialize all example prompt templates"""

        # ===== Greeter Agent Prompts =====
        self.greeter_system = """You are a friendly greeter assistant. Your role is to greet people in a warm and welcoming manner."""

        self.greeter_task = """Please greet the following person: {{ target }}

Be friendly, warm, and personable in your greeting."""

        # ===== Calculator Agent Prompts =====
        self.calculator_system = """You are a helpful calculator assistant. You can perform mathematical operations using the tools available to you.

When the user asks you to perform calculations, use the appropriate tools to compute the results."""

        self.calculator_task = """Please help with the following calculation task: {{ target }}

Use the available tools to perform the calculations and provide the result."""


# Register the repository
examples_repo = ExamplesPromptsRepo()
