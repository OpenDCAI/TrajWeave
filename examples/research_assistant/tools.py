"""Research Assistant 工具定义"""
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """搜索互联网获取相关信息。

    Args:
        query: 搜索查询关键词
    """
    # Mock search results based on keywords
    mock_results = {
        "transformer": (
            "## Search Results for: {query}\n\n"
            "1. **Attention Is All You Need (2017)** - The original Transformer paper by Vaswani et al. "
            "introduced the self-attention mechanism that revolutionized NLP.\n\n"
            "2. **GPT-4 and Beyond (2024-2025)** - Large language models based on Transformer architecture "
            "have achieved remarkable capabilities in reasoning, coding, and multimodal understanding.\n\n"
            "3. **Efficient Transformers (2025)** - Recent work focuses on reducing the quadratic complexity "
            "of attention: Flash Attention 3, Ring Attention, and Mixture of Experts (MoE) architectures.\n\n"
            "4. **Vision Transformers (ViT)** - Transformers have been successfully applied to computer vision, "
            "with models like DINOv2 and SAM2 achieving state-of-the-art results.\n\n"
            "5. **Multimodal Transformers** - Models like GPT-4V, Claude 3, and Gemini combine text, image, "
            "and video understanding in unified Transformer architectures."
        ),
        "agent": (
            "## Search Results for: {query}\n\n"
            "1. **AI Agents in 2025-2026** - Autonomous AI agents are the hottest trend, with frameworks "
            "like LangGraph, CrewAI, and DeerFlow enabling complex multi-agent workflows.\n\n"
            "2. **DeerFlow 2.0 by ByteDance** - A super agent harness supporting sub-agent spawning, "
            "sandboxed execution, and long-term memory. 50k+ GitHub stars.\n\n"
            "3. **Agent Design Patterns** - ReAct, Plan-and-Execute, and Reflection are the dominant "
            "patterns for building reliable AI agents.\n\n"
            "4. **Tool Use and Function Calling** - Modern LLMs support native function calling, "
            "enabling agents to interact with external systems reliably."
        ),
        "default": (
            "## Search Results for: {query}\n\n"
            "1. Found several relevant articles and papers on this topic.\n\n"
            "2. Key developments include recent advances in AI/ML, new open-source tools, "
            "and growing industry adoption.\n\n"
            "3. Multiple research groups are actively working on improvements and applications.\n\n"
            "4. Community discussions highlight both opportunities and challenges in this area."
        ),
    }

    query_lower = query.lower()
    for keyword, result_template in mock_results.items():
        if keyword != "default" and keyword in query_lower:
            return result_template.format(query=query)
    return mock_results["default"].format(query=query)


@tool
def analyze_data(data: str, focus: str = "general") -> str:
    """分析和综合研究数据，提取关键主题和洞察。

    Args:
        data: 需要分析的研究数据
        focus: 分析焦点方向
    """
    # Mock analysis
    data_preview = data[:200] if len(data) > 200 else data
    return (
        f"## Analysis Results (Focus: {focus})\n\n"
        f"**Data Overview**: Analyzed {len(data)} characters of research data.\n\n"
        f"**Key Themes Identified**:\n"
        f"- Theme 1: Core technical concepts and their evolution\n"
        f"- Theme 2: Practical applications and industry impact\n"
        f"- Theme 3: Future directions and open challenges\n\n"
        f"**Confidence**: High (based on multiple corroborating sources)\n\n"
        f"**Data Preview**: {data_preview}..."
    )
