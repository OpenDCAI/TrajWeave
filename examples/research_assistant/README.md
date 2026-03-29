# FlowAgent Multi-Scenario Demo

Multi-agent demo with 6 scenarios, powered by FlowAgent + Chainlit.

## Scenarios

| Scenario | Mode | Description |
|---|---|---|
| Deep Research | Multi-SubAgent | Orchestrator → Researcher → Analyzer → Writer |
| Code Review | Simple Agent + Skill | Professional code review report |
| Smart Chat | Simple + Memory | Multi-turn conversation with cross-session memory |
| Image Analysis | VLM | Vision Language Model image understanding |
| Task Planner | Plan-Execute | Break down complex tasks into steps |
| Data Analysis | Parallel SubAgents | Trend/Impact/Risk analysts + Synthesizer |

## Features Demonstrated

| FlowAgent Feature | Usage |
|---|---|
| Sub-Agent System | Dynamic sub-agent spawning (spawn / spawn_many) |
| Long-term Memory | SQLite-backed cross-session memory |
| Context Budget | Token usage monitoring and compression |
| Observability | Agent execution tracing (LangSmith compatible) |
| Model Router | Task-based model selection |
| Markdown Skills | Extensible skill system |

## Quick Start

### 1. Install Dependencies

```bash
pip install chainlit aiosqlite tiktoken
```

### 2. Configure

```bash
export DF_API_URL="https://api.openai.com/v1"  # or any OpenAI-compatible endpoint
export DF_API_KEY="sk-your-key-here"
export DF_MODEL="gpt-4o"  # optional, defaults to gpt-4o
```

### 3. Run

```bash
cd examples/research_assistant
chainlit run app.py
```

### 4. Open Browser

Navigate to http://localhost:8000

## Architecture

```
User Input
  │
  ▼
Orchestrator (task decomposition)
  │
  ├── Researcher (web search, sequential)
  │
  ├── Analyzer  ┐
  │             ├── parallel execution
  └── Writer    ┘
        │
        ▼
  Streaming Report Output
```

## Memory

User memory is stored in `~/.flowagent/research_assistant_memory.db` (SQLite).
It persists across sessions and includes:
- User preferences
- Research history
- Session summaries
