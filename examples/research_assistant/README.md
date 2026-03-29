# FlowAgent Research Assistant Demo

Multi-agent research assistant with memory, powered by FlowAgent + Chainlit.

## Features Demonstrated

| FlowAgent Feature | Usage |
|---|---|
| Sub-Agent System | 3 specialized agents (Researcher/Analyzer/Writer) |
| Long-term Memory | Remembers user preferences across sessions |
| Context Budget | Token usage monitoring |
| Observability | Agent execution tracing |
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
