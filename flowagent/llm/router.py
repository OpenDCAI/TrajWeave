"""
模型路由器 - DeerFlow 2.0 对齐

DeerFlow 2.0 的关键设计：不同任务使用不同模型
- 复杂推理 → 强模型（Claude Sonnet, GPT-4o）
- 对话摘要 → 便宜模型（GPT-4o-mini, DeepSeek）
- 代码生成 → 代码专用模型
- 嵌入向量 → 嵌入专用模型

通过配置文件声明路由规则，避免硬编码。

使用示例：
router = ModelRouter({
    "reasoning_model": "gpt-4o",
    "summary_model": "gpt-4o-mini",
    "code_model": "claude-sonnet",
    "default_model": "gpt-4o",
})

# 按任务类型获取 LLM
llm = router.get_llm("summarization", state)  # → gpt-4o-mini
llm = router.get_llm("reasoning", state)# → gpt-4o
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from flowagent.logger import get_logger

if TYPE_CHECKING:
    from flowagent.state.base import MainState

log = get_logger(__name__)

# 默认路由配置
_DEFAULT_ROUTING = {
    "reasoning": None,    # 使用 state 中的默认模型
    "summarization": None,   # 使用 state 中的默认模型
    "code_generation": None,
    "embedding": None,
    "default": None,
}


class ModelRouter:
    """模型路由器 - 按任务类型选择最优模型

    Attributes:
    routing_config: 任务类型 → 模型名称 的映射
    """

    def __init__(self, config: Optional[Dict[str, str]] = None):
        self._routing = dict(_DEFAULT_ROUTING)
        if config:
            self._routing.update(config)
        log.info(f"ModelRouter 初始化: {self._routing}")

    def get_model_name(self, task_type: str = "default") -> Optional[str]:
        """获取指定任务类型对应的模型名称

        Args:
        task_type: 任务类型，如 "reasoning", "summarization" 等

        Returns:
        模型名称，如果该任务类型未配置则返回 None（使用默认模型）
        """
        model = self._routing.get(task_type) or self._routing.get("default")
        return model

    def get_llm(
        self,
        task_type: str,
        state: "MainState",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        """获取指定任务类型的 LLM 实例

        如果路由配置中指定了模型，使用指定模型；
        否则使用 state.request 中的默认模型和配置。

        Args:
        task_type: 任务类型
        state: MainState（包含 LLM 配置）
        temperature: 温度参数
        max_tokens: 最大 Token 数

        Returns:
        ChatOpenAI 实例
        """
        from langchain_openai import ChatOpenAI

        model_name = self.get_model_name(task_type)

        llm = ChatOpenAI(
            openai_api_base=state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=model_name or state.request.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        log.debug(
            f"[ModelRouter] 为任务 '{task_type}' 提供模型: "
            f"{model_name or state.request.model}"
        )
        return llm

    def update_routing(self, task_type: str, model_name: str) -> None:
        """动态更新路由规则

        Args:
        task_type: 任务类型
        model_name: 模型名称
        """
        self._routing[task_type] = model_name
        log.info(f"[ModelRouter] 更新路由: {task_type} → {model_name}")

    def get_routing_table(self) -> Dict[str, Optional[str]]:
        """获取当前路由表"""
        return dict(self._routing)


# 全局单例
_router_instance: Optional[ModelRouter] = None


def get_model_router(config: Optional[Dict[str, str]] = None) -> ModelRouter:
    """获取全局模型路由器实例"""
    global _router_instance
    if _router_instance is None:
        _router_instance = ModelRouter(config)
    return _router_instance
