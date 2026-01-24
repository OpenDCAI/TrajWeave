<div align="center">

# FlowAgent

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-flowagent-blue?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/flowagent/)

**A flexible LLM agent framework for building intelligent applications**

FlowAgent provides a clean, extensible framework for building LLM-powered agents with support for multiple execution modes, tool integration, and graph-based workflows.

[Quick Start](#-quick-start) • [Features](#-features) • [Examples](#-examples) • [Documentation](#-documentation)

</div>

---

## ✨ Features

- **🎯 Multiple Execution Modes**: Simple, ReAct, Graph, Parallel, VLM, and Planning modes
- **🔧 Tool Integration**: Easy-to-use tool management system with pre/post tool execution
- **📊 Graph-Based Workflows**: Build complex multi-agent workflows using LangGraph
- **🎨 Flexible Prompt System**: Template-based prompt management with multi-language support
- **📦 State Management**: Clean state management with dataclasses
- **🔍 Trajectory Tracking**: Built-in execution history tracking and export
- **🚀 Minimal Dependencies**: Lightweight core with optional extensions

---

## 🚀 Quick Start

### Installation

```bash
pip install flowagent
```

### Basic Usage

```python
from flowagent import BaseAgent, register, SimpleConfig
from flowagent.state import MainState, MainRequest

@register("my_agent")
class MyAgent(BaseAgent):
    @property
    def role_name(self) -> str:
        return "MyAgent"

    @property
    def system_prompt_template_name(self) -> str:
        return "my_agent_system"

# Create and execute agent
config = SimpleConfig(model="gpt-4o-mini")
agent = MyAgent(config=config)

state = MainState(
    request=MainRequest(
        target="Your task here",
        model="gpt-4o-mini"
    )
)

result = agent.execute(state)
```

---

## 📚 Core Concepts

### Execution Modes

FlowAgent supports multiple execution modes for different use cases:

- **Simple**: Single LLM call for straightforward tasks
- **ReAct**: Reasoning and acting loop with tool use
- **Graph**: Complex multi-agent workflows with LangGraph
- **Parallel**: Concurrent LLM calls for parallel processing
- **VLM**: Vision-language model support
- **Planning**: Plan generation and execution modes

### Agent Registration

Agents are automatically discovered and registered using the `@register` decorator:

```python
from flowagent import BaseAgent, register, AgentRegistry

@register("calculator")
class CalculatorAgent(BaseAgent):
    # Agent implementation
    pass

# Get registered agent
agent_class = AgentRegistry.get("calculator")
```

### Tool Integration

Define tools for your agents easily:

```python
def get_tools(self, state: MainState):
    def add(a: float, b: float) -> float:
        """Add two numbers"""
        return a + b

    return [add]
```

---

## 📖 Examples

Check out the [examples/](examples/) directory for complete examples:

- **[simple_agent.py](examples/simple_agent.py)**: Basic agent with system and task prompts
- **[react_agent.py](examples/react_agent.py)**: ReAct pattern with tool use

---

## 📂 Project Structure

```
flowagent/
├── core/              # Core agent system (BaseAgent, Registry, Strategies)
├── llm/               # LLM infrastructure (Text, Image callers)
├── parsers/           # Output parsers (JSON, XML, Text)
├── graph/             # Graph building (LangGraph integration)
├── state/             # State management (MainState, MainRequest)
├── prompts/           # Prompt template system
├── tools/             # Tool management
├── workflow/          # Workflow orchestration
├── storage/           # Storage service
├── trajectory/        # Trajectory tracking
├── utils.py           # Core utilities
└── logger.py          # Logging
```

---

## 🔧 Advanced Usage

### Custom Execution Strategies

```python
from flowagent.core import ReactConfig

config = ReactConfig(
    model="gpt-4o-mini",
    temperature=0.0,
    max_iterations=5
)
```

### Graph-Based Workflows

```python
from flowagent.graph import GraphBuilder

# Build complex multi-agent workflows
builder = GraphBuilder()
# Add nodes, edges, and conditional logic
```

### Prompt Templates

```python
from flowagent.prompts import PromptsTemplateGenerator

generator = PromptsTemplateGenerator()
prompt = generator.generate("template_name", context={"key": "value"})
```

---

## 📄 Documentation

For detailed documentation, please visit:

- **Getting Started**: [docs/getting_started.md](docs/getting_started.md)
- **Core Concepts**: [docs/core_concepts.md](docs/core_concepts.md)
- **API Reference**: [docs/api_reference.md](docs/api_reference.md)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🌟 Acknowledgments

FlowAgent is built on top of:
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Graph-based agent orchestration

---

<div align="center">

**Built with ❤️ by the FlowAgent Team**

[![GitHub stars](https://img.shields.io/github/stars/OpenDCAI/FlowAgent?style=social)](https://github.com/OpenDCAI/FlowAgent/stargazers)

</div>
