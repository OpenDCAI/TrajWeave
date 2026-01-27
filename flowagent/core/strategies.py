from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, List, Optional
from flowagent.logger import get_logger
import asyncio  # 添加导入
import warnings  # 添加废弃警告支持
import difflib
from pathlib import Path


def _get_tool_expected_keys(tool: Any) -> set:
    """Best-effort extraction of a LangChain tool's expected argument keys."""
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return set()
    fields = getattr(schema, "model_fields", None)
    if fields is None:
        fields = getattr(schema, "__fields__", None)
    if not fields:
        return set()
    return set(fields.keys())


def _normalize_tool_args(tool: Any, tool_args: Any) -> Any:
    """Normalize common alias keys (e.g. path/query) to a tool's expected keys."""
    if not isinstance(tool_args, dict):
        return tool_args

    expected = _get_tool_expected_keys(tool)
    normalized = dict(tool_args)

    def _alias(src: str, dst: str) -> None:
        if src in normalized and dst not in normalized and (not expected or dst in expected):
            normalized[dst] = normalized[src]

    # Common aliases seen in planning JSON
    _alias("path", "directory")
    _alias("query", "pattern")

    return normalized


def _pick_allowed_filename(description: str, requested_basename: str, allowed: List[str]) -> Optional[str]:
    """Pick a filename from `allowed` when planner invented/mistyped a name.

    Strategy: close-match basename -> keyword heuristics -> first .py file -> first allowed.
    """
    if not allowed:
        return None

    requested_basename = (requested_basename or "").strip()
    description_l = (description or "").lower()

    # 1) Direct close match on basename
    if requested_basename:
        guess = difflib.get_close_matches(requested_basename, allowed, n=1, cutoff=0.6)
        if guess:
            return guess[0]

    # 2) Keyword-based picks for common core files
    preferred = [
        ("base_agent", "base_agent.py"),
        ("execution_config", "execution_config.py"),
        ("factory", "factory.py"),
        ("strategy", "strategies.py"),
        ("config", "configs.py"),
        ("registry", "registry.py"),
        ("plan", "plan_utils.py"),
        ("__init__", "__init__.py"),
        ("readme", "README.md"),
        ("documentation", "documentation.md"),
    ]
    for kw, fname in preferred:
        if kw in description_l and fname in allowed:
            return fname

    # 3) Fall back to a reasonable python file
    for fname in allowed:
        if fname.endswith(".py") and fname != "__init__.py":
            return fname

    return allowed[0]

if TYPE_CHECKING:
    from flowagent.core.base_agent import BaseAgent
    from flowagent.state import MainState

log = get_logger(__name__)


class ExecutionStrategy(ABC):
    """执行策略基类"""
    
    def __init__(self, agent: "BaseAgent", config: Any):
        self.agent = agent
        self.config = config
    
    @abstractmethod
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        """执行策略的核心方法"""
        pass


class SimpleStrategy(ExecutionStrategy):
    """简单模式策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[SimpleStrategy] 执行 {self.agent.role_name}")
        pre_tool_results = await self.agent.execute_pre_tools(state)
        result = await self.agent.process_simple_mode(state, pre_tool_results)
        return result


class ValidationRetryStrategy(ExecutionStrategy):
    """验证重试模式策略（原ReactStrategy）- 带验证的循环调用，验证失败则重试"""

    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[ValidationRetryStrategy] 执行 {self.agent.role_name}，最大重试: {self.config.max_retries}")
        
        # 注入自定义验证器
        if self.config.validators:
            original_validators = self.agent.get_react_validators
            def custom_validators():
                return original_validators() + self.config.validators
            self.agent.get_react_validators = custom_validators
        
        pre_tool_results = await self.agent.execute_pre_tools(state)
        
        # 临时覆盖 react_max_retries
        original_retries = self.agent.react_max_retries
        self.agent.react_max_retries = self.config.max_retries
        
        result = await self.agent.process_react_mode(state, pre_tool_results)
        
        self.agent.react_max_retries = original_retries
        return result


class ReactStrategy(ExecutionStrategy):
    """ReAct模式策略（原GraphStrategy）- 使用LangGraph的工具调用循环，真正的推理+行动"""

    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[ReactStrategy] 执行 {self.agent.role_name} ReAct模式")
        pre_tool_results = await self.agent.execute_pre_tools(state)
        
        post_tools = self.agent.get_post_tools()
        if not post_tools:
            log.warning("无后置工具，回退到简单模式")
            return await self.agent.process_simple_mode(state, pre_tool_results)
        
        result = await self.agent._execute_react_graph(state, pre_tool_results)
        return result


class VLMStrategy(ExecutionStrategy):
    """视觉语言模型策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[VLMStrategy] 执行 {self.agent.role_name} VLM模式: {self.config.vlm_mode}")
        
        # 构建 VLM 配置字典
        vlm_config = {
            "mode": self.config.vlm_mode,
            "image_detail": self.config.image_detail,
            "max_image_size": self.config.max_image_size,
            **self.config.additional_params
        }
        
        # 临时注入配置
        # original_vlm_config = getattr(self.agent, 'vlm_config', {})

        self.agent.vlm_config.update(vlm_config)
        self.agent.model_name = self.config.model_name
        self.agent.temperature = self.config.temperature
        self.agent.max_tokens = self.config.max_tokens

        log.info(f"VLMStrategy 执行 {self.agent.role_name} VLM模式: {self.agent.vlm_config} + {kwargs},self.config={self.config}")
        result = await self.agent._execute_vlm(state, **kwargs)
        log.critical(f"VLMStrategy 执行 {self.agent.role_name} VLM模式结果: {result}")
        
        # self.agent.vlm_config = original_vlm_config
        return result


class ParallelStrategy(ExecutionStrategy):
    """并行模式策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[ParallelStrategy] 执行 {self.agent.role_name}，并行度限制: {self.config.concurrency_limit}")
        
        # 执行前置工具获取结果
        pre_tool_results = await self.agent.execute_pre_tools(state)
        
        # 检查是否有parallel_items，如果没有且pre_tool_results是列表，自动转换为parallel_items
        if "parallel_items" not in pre_tool_results and isinstance(pre_tool_results, list):
            pre_tool_results = {"parallel_items": pre_tool_results}
        
        # 执行并行模式
        return await self.agent.process_parallel_mode(state, pre_tool_results)


# ==================== Planning Agent 策略 ====================

class PlanSolveStrategy(ExecutionStrategy):
    """
    Plan-and-Solve 策略
    
    一次性生成完整计划，然后按顺序执行，不回头调整。
    
    流程:
    1. 调用 Planner 生成完整计划
    2. (可选) Human-in-the-Loop: 等待用户审批计划
    3. 按顺序执行每个步骤
    4. 收集结果并返回
    """
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        from flowagent.state import PlanningState
        
        log.info(f"[PlanSolveStrategy] 执行 {self.agent.role_name}")
        
        # 确保状态类型正确
        if not isinstance(state, PlanningState):
            log.warning("状态不是 PlanningState 类型，可能会缺少某些功能")
        
        # Step 1: 生成计划
        log.info("[PlanSolveStrategy] Step 1: 生成计划")
        plan = await self._generate_plan(state)
        
        if not plan:
            return {"error": "无法生成计划", "status": "failed"}
        
        log.info(f"[PlanSolveStrategy] 生成了 {len(plan)} 步计划")
        
        # 更新状态
        if hasattr(state, 'plan'):
            state.plan = plan

        # 自动追加“落盘汇总”收尾步骤：把各步 tool_output 汇总写入 documentation.md（通过 write_file 工具）
        if self.config.executor_tools and hasattr(state, "plan_steps") and state.plan_steps:
            tool_names = {t.name for t in self.config.executor_tools}
            if "write_file" in tool_names:
                state.plan_steps.append(
                    {
                        "description": "将上述步骤的工具输出汇总写入 flowagent/core/documentation.md",
                        "tool": "write_file",
                        "tool_input": {
                            "path": "flowagent/core/documentation.md",
                            "content": "__AUTO_GENERATE_DOCUMENTATION__",
                        },
                    }
                )
                plan.append("将上述步骤的工具输出汇总写入 flowagent/core/documentation.md")
        
        # Step 2: Human-in-the-Loop - 计划审批
        if self.config.require_plan_approval:
            log.info("[PlanSolveStrategy] Step 2: 等待用户审批计划")
            # 使用 LangGraph 的 interrupt 机制
            from langgraph.types import interrupt
            
            approval_response = interrupt({
                "type": "plan_approval",
                "message": "请审批以下计划:",
                "plan": plan,
                "options": ["approve", "reject", "modify"]
            })
            
            if approval_response.get("action") == "reject":
                return {
                    "status": "rejected",
                    "message": "计划被用户拒绝",
                    "plan": plan
                }
            elif approval_response.get("action") == "modify":
                plan = approval_response.get("modified_plan", plan)
                if hasattr(state, 'plan'):
                    state.plan = plan
        
        # 标记计划已审批
        if hasattr(state, 'plan_approved'):
            state.plan_approved = True
        
        # Step 3: 按顺序执行计划
        log.info("[PlanSolveStrategy] Step 3: 执行计划")
        results = []

        # 用于执行期硬约束：read_file 只能读取 list_dir 输出中真实存在的文件
        dir_listings: Dict[str, List[str]] = {}

        # 优先使用结构化计划（每步包含 tool + tool_input），避免执行阶段“凭空编造”
        plan_steps = None
        if hasattr(state, "plan_steps") and state.plan_steps:
            plan_steps = state.plan_steps
            if len(plan_steps) != len(plan):
                log.warning("state.plan_steps 与 plan 长度不一致，将回退到 plan 字符串执行")
                plan_steps = None
        
        for i, step in enumerate(plan):
            step_meta = plan_steps[i] if plan_steps else None
            step_desc = step_meta.get("description") if isinstance(step_meta, dict) else step
            log.info(f"[PlanSolveStrategy] 执行步骤 {i+1}/{len(plan)}: {str(step_desc)[:50]}...")
            
            try:
                if step_meta and isinstance(step_meta, dict) and step_meta.get("tool"):
                    step_result = await self._execute_structured_step(
                        state,
                        step_meta,
                        i,
                        results_so_far=results,
                        dir_listings=dir_listings,
                    )
                else:
                    step_result = await self._execute_step(state, step, i)
                results.append({
                    "step_index": i,
                    "step": step_desc,
                    "result": step_result,
                    "status": "completed"
                })

                # 如果本步是 list_dir，则把输出文件名缓存起来，供后续 read_file 做“只能读已列出文件”的硬约束
                if isinstance(step_result, dict) and step_result.get("tool") == "list_dir":
                    listed_path = str((step_result.get("tool_input") or {}).get("path") or ".")
                    normalized_dir = listed_path.replace("\\", "/").rstrip("/")
                    raw_output = str(step_result.get("tool_output") or "")
                    entries: List[str] = []
                    for line in raw_output.splitlines():
                        name = line.strip()
                        if not name or name.endswith("/"):
                            continue
                        entries.append(name)
                    if entries:
                        dir_listings[normalized_dir] = entries
                
                # 更新状态
                if hasattr(state, 'mark_step_complete'):
                    state.mark_step_complete(str(step_result))
                    
            except Exception as e:
                log.error(f"[PlanSolveStrategy] 步骤 {i+1} 执行失败: {e}")
                results.append({
                    "step_index": i,
                    "step": step,
                    "error": str(e),
                    "status": "failed"
                })
                
                if not self.config.continue_on_error:
                    break

        # 如果自动收尾写盘步骤没有被执行到（例如中途 break），这里尽量补执行一次。
        if plan_steps and self.config.executor_tools:
            already_wrote = any(
                isinstance(r.get("result"), dict)
                and (r.get("result") or {}).get("tool") == "write_file"
                and ((r.get("result") or {}).get("tool_input") or {}).get("path") == "flowagent/core/documentation.md"
                for r in results
            )

            last_step = plan_steps[-1] if plan_steps else None
            if (
                not already_wrote
                and isinstance(last_step, dict)
                and last_step.get("tool") == "write_file"
                and ((last_step.get("tool_input") or {}).get("content") == "__AUTO_GENERATE_DOCUMENTATION__")
            ):
                try:
                    step_result = await self._execute_structured_step(
                        state,
                        last_step,
                        len(results),
                        results_so_far=results,
                        dir_listings=dir_listings,
                    )
                    results.append(
                        {
                            "step_index": len(results),
                            "step": last_step.get("description") or "write documentation",
                            "result": step_result,
                            "status": "completed",
                        }
                    )
                except Exception as e:
                    log.error(f"[PlanSolveStrategy] 收尾写 documentation.md 失败: {e}")
                    results.append(
                        {
                            "step_index": len(results),
                            "step": last_step.get("description") or "write documentation",
                            "error": str(e),
                            "status": "failed",
                        }
                    )
        
        # Step 4: 返回结果
        failed_steps = sum(1 for r in results if r.get("status") != "completed")
        overall_status = "completed" if failed_steps == 0 else "completed_with_errors"
        return {
            "status": overall_status,
            "plan": plan,
            "results": results,
            "total_steps": len(plan),
            "completed_steps": sum(1 for r in results if r.get("status") == "completed")
        }
    
    async def _generate_plan(self, state: "MainState") -> List[str]:
        """
        使用 LLM 生成计划
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field
        try:
            from pydantic import ConfigDict
        except Exception:  # pragma: no cover
            ConfigDict = None
        
        # 获取任务描述
        task = state.request.target if hasattr(state, 'request') else ""
        if hasattr(state, 'original_task') and state.original_task:
            task = state.original_task

        # 尽量从任务里识别要分析的目录，用于约束第 1 步必须 list_dir
        import re
        dir_hint = None
        m = re.search(r"(flowagent[/\\]core)\b", task)
        if m:
            dir_hint = m.group(1).replace("\\", "/")
        
        # 获取可用工具信息
        tools_info = ""
        available_tools: List[str] = []
        if hasattr(state, 'executor_tools') and state.executor_tools:
            available_tools = list(state.executor_tools)
            tools_info = f"\n可用工具: {', '.join(available_tools)}"
        
        # 构建 Planner 的提示词
        system_prompt = """你是一个任务规划专家。你的职责是分析用户的任务，并生成一个清晰、可执行的分步计划。

规则:
1. 每个步骤应该是独立的、可执行的操作
2. 步骤之间应该有逻辑顺序
3. 步骤描述应该清晰具体
4. 步骤数量不宜过多，通常 3-7 步为宜
5. 不要包含无关的步骤
6. 如果任务涉及分析某个目录：第 1 步必须是 list_dir 列出目录下文件；后续 read_file 只能读取第 1 步 list_dir 输出里真实存在的文件名（禁止编造文件名）。"""

        tool_param_help = ""
        if available_tools:
            known_signatures = {
                "list_dir": "list_dir(path: str = '.')",
                "read_file": "read_file(path: str)",
                "write_file": "write_file(path: str, content: str)",
                "search_code": "search_code(pattern: str, directory: str = '.')",
            }
            lines = []
            for t in available_tools:
                sig = known_signatures.get(t)
                if sig:
                    lines.append(f"- {sig}")
            if lines:
                tool_param_help = "\n\n工具参数(用于生成 tool_input 的 key):\n" + "\n".join(lines)

        task_prompt = f"""请为以下任务生成执行计划:

任务: {task}
{tools_info}

输出要求:
1. 必须输出 JSON，且仅输出 JSON（不要 Markdown/不要解释）
2. steps 中每一步必须包含: description, tool, tool_input
3. tool 必须来自可用工具列表；tool_input 必须是 JSON 对象
4. 每一步都要能通过一次工具调用落地执行（不要写泛泛的“阅读/分析/总结”，而要写成可执行的工具调用）
5. tool_input 的 key 必须与所选工具的参数名一致（例如 search_code 需要 pattern/directory，而不是 path/query）
6. 不要输出不存在的文件名（例如 flow.py、config.py 之类如果目录里没有就不要用）；如果不确定，优先选择目录中真实存在的 .py 文件（如 base_agent.py、configs.py）。
{tool_param_help}

返回格式示例:
{{
    "steps": [
        {{"description": "列出 flowagent/core 下的文件", "tool": "list_dir", "tool_input": {{"path": "flowagent/core"}}}},
        {{"description": "阅读 base_agent.py 头部", "tool": "read_file", "tool_input": {{"path": "flowagent/core/base_agent.py"}}}}
    ]
}}"""

        if dir_hint:
            task_prompt += (
                f"\n\n硬性约束: 第 1 步必须是 list_dir(tool_input={{\"path\": \"{dir_hint}\"}})。"
            )

        # 创建 LLM
        planner_model = self.config.planner_model or self.config.model_name or state.request.model
        llm = ChatOpenAI(
            openai_api_base=self.config.chat_api_url or state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=planner_model,
            temperature=self.config.planner_temperature,
        )
        
        # 使用结构化输出：每步必须指定 tool + tool_input，执行阶段直接调用工具，避免编造
        class PlanStepOutput(BaseModel):
            description: str = Field(description="步骤描述")
            tool: str = Field(description="本步骤必须使用的工具名称")
            tool_input: Dict[str, Any] = Field(default_factory=dict, description="工具入参(JSON对象)")

            if ConfigDict:
                model_config = ConfigDict(extra="forbid")

        class PlanOutput(BaseModel):
            steps: List[PlanStepOutput] = Field(description="结构化计划步骤列表")

            if ConfigDict:
                model_config = ConfigDict(extra="forbid")
        
        try:
            system_prompt_with_tools = system_prompt
            if available_tools:
                system_prompt_with_tools += "\n\n你只能从以下工具中选择(每步必须选一个):\n" + "\n".join(
                    f"- {t}" for t in available_tools
                )
                system_prompt_with_tools += "\n\n重要: 输出必须为结构化 JSON，不要输出额外文本。"

            # OpenAI Structured Outputs (response_format) 对 JSON Schema 要求非常严格。
            # 这里改用 function_calling 方式，避免因 schema 细节（如 additionalProperties）触发 400。
            structured_llm = llm.with_structured_output(PlanOutput, method="function_calling")
            response = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt_with_tools),
                HumanMessage(content=task_prompt)
            ])
            
            # 限制最大步骤数
            structured_steps = response.steps[:self.config.max_plan_steps]
            # 保存结构化计划到 state（如支持），并返回字符串 plan 供日志/兼容使用
            if hasattr(state, "plan_steps"):
                state.plan_steps = [
                    {
                        "description": s.description,
                        "tool": s.tool,
                        "tool_input": s.tool_input,
                    }
                    for s in structured_steps
                ]
            return [s.description for s in structured_steps]
            
        except Exception as e:
            log.error(f"[PlanSolveStrategy] 生成计划失败: {e}")
            # 回退到非结构化输出
            try:
                response = await llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=task_prompt)
                ])
                # 尝试解析 JSON
                import json
                import re
                content = response.content
                # 提取 JSON
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                    raw_steps = data.get("steps", [])[: self.config.max_plan_steps]

                    # 如果已经是结构化 steps，直接使用，保持 plan 与 state.plan_steps 一致。
                    structured: List[Dict[str, Any]] = []
                    for s in raw_steps:
                        if isinstance(s, dict) and {"description", "tool", "tool_input"}.issubset(set(s.keys())):
                            structured.append(
                                {
                                    "description": str(s.get("description") or ""),
                                    "tool": str(s.get("tool") or ""),
                                    "tool_input": s.get("tool_input") or {},
                                }
                            )

                    if structured and hasattr(state, "plan_steps"):
                        state.plan_steps = structured
                        return [s["description"] for s in structured]

                    # 否则退化为字符串步骤（兼容旧格式）
                    steps_as_text = [str(s) for s in raw_steps]
                    return steps_as_text
            except Exception as e2:
                log.error(f"[PlanSolveStrategy] 回退解析也失败: {e2}")
            return []

    async def _execute_structured_step(
        self,
        state: "MainState",
        step_meta: Dict[str, Any],
        step_index: int,
        *,
        results_so_far: Optional[List[Dict[str, Any]]] = None,
        dir_listings: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """直接执行结构化步骤：按 tool/tool_input 调用工具，并返回可验证结果。"""

        tool_name = step_meta.get("tool")
        tool_input = step_meta.get("tool_input") or {}
        description = step_meta.get("description") or ""

        tool_map = {t.name: t for t in (self.config.executor_tools or [])}
        if tool_name not in tool_map:
            raise RuntimeError(
                f"计划要求使用工具 '{tool_name}'，但该工具未在 executor_tools 中注册。"
            )

        tool_obj = tool_map[tool_name]
        tool_input = _normalize_tool_args(tool_obj, tool_input)

        results_so_far = results_so_far or []
        dir_listings = dir_listings or {}

        # 执行期硬约束：read_file 只能读取真实存在的文件；如果已有 list_dir 输出，则必须来自其列出的文件集合。
        if tool_name == "read_file" and isinstance(tool_input, dict):
            raw_path = str(tool_input.get("path") or "")
            if raw_path:
                normalized_path = raw_path.replace("\\", "/")
                parent_dir = str(Path(normalized_path).parent).replace("\\", "/").rstrip("/")
                basename = Path(normalized_path).name

                allowed = dir_listings.get(parent_dir)
                if allowed is not None and basename not in allowed:
                    picked = _pick_allowed_filename(description, basename, allowed)
                    if picked:
                        fixed = f"{parent_dir}/{picked}" if parent_dir else picked
                        log.warning(
                            f"[PlanSolveStrategy] 修正 read_file 路径(不在 list_dir 输出): {raw_path} -> {fixed}"
                        )
                        tool_input["path"] = fixed
                    else:
                        raise RuntimeError(f"read_file 无可用文件可读: list_dir({parent_dir}) 为空")

                final_path = str(tool_input.get("path") or "")
                if final_path and not Path(final_path).exists():
                    raise RuntimeError(f"读取失败: 文件不存在: {final_path}")

        # 收尾落盘：自动生成 documentation.md 内容
        if tool_name == "write_file" and isinstance(tool_input, dict):
            if tool_input.get("content") == "__AUTO_GENERATE_DOCUMENTATION__":
                tool_input["content"] = self._build_documentation_markdown(results_so_far)
        try:
            if hasattr(tool_obj, "ainvoke"):
                tool_result = await tool_obj.ainvoke(tool_input)
            else:
                tool_result = tool_obj.invoke(tool_input)
        except Exception as e:
            raise RuntimeError(f"工具 {tool_name} 执行失败: {e}") from e

        tool_result_text = str(tool_result)

        # 约定式错误：示例工具会以中文前缀返回失败信息；这里将其视为失败而不是成功结果。
        failure_prefixes = (
            "读取失败:",
            "写入失败:",
            "列目录失败:",
            "搜索失败:",
            "目录不存在:",
            "不是目录:",
        )
        if tool_result_text.startswith(failure_prefixes):
            raise RuntimeError(tool_result_text)

        # 统一返回结构，避免把模型总结当成“事实来源”
        return {
            "description": description,
            "tool": tool_name,
            "tool_input": tool_input,
            "tool_output": tool_result_text,
        }

    def _build_documentation_markdown(self, results_so_far: List[Dict[str, Any]]) -> str:
        lines: List[str] = [
            "# FlowAgent Core Documentation",
            "",
            "## Steps & Tool Outputs",
            "",
        ]

        for item in results_so_far:
            idx = item.get("step_index")
            title = item.get("step")
            status = item.get("status")
            step_no = (idx + 1) if isinstance(idx, int) else idx
            lines.append(f"### Step {step_no}: {title}")
            lines.append(f"- Status: {status}")

            if status == "completed":
                result = item.get("result")
                if isinstance(result, dict):
                    lines.append(f"- Tool: {result.get('tool')}")
                    lines.append(f"- Tool input: {result.get('tool_input')}")
                    lines.append("")
                    lines.append("```text")
                    lines.append(str(result.get("tool_output") or "")[:5000])
                    lines.append("```")
                else:
                    lines.append("")
                    lines.append("```text")
                    lines.append(str(result)[:5000])
                    lines.append("```")
            else:
                lines.append(f"- Error: {item.get('error')}")

            lines.append("")

        return "\n".join(lines) + "\n"

    async def _execute_step(self, state: "MainState", step: str, step_index: int) -> str:
        """
        执行单个计划步骤（支持工具调用）
        """
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
        from langchain_openai import ChatOpenAI

        # 获取上下文
        context = ""
        if hasattr(state, 'past_steps') and state.past_steps:
            context = "\n已完成的步骤:\n"
            for prev_step, prev_result in state.past_steps:
                context += f"- {prev_step}: {prev_result[:100]}...\n"

        # 构建系统提示词
        system_prompt = """你是一个任务执行专家。你需要执行给定的步骤。

    规则:
    1. 如果有可用工具，必须至少调用一个工具完成任务
    2. 专注于当前步骤，不要做额外的事情
    3. 执行完成后，总结执行结果"""

        task_prompt = f"""请执行以下步骤:

步骤 {step_index + 1}: {step}
{context}

请执行这个步骤。"""

        # 创建 LLM
        executor_model = self.config.executor_model or self.config.model_name or state.request.model
        llm = ChatOpenAI(
            openai_api_base=self.config.chat_api_url or state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=executor_model,
            temperature=self.config.executor_temperature,
        )

        # 如果有工具，绑定工具并进入工具调用循环
        if self.config.executor_tools:
            return await self._execute_with_tools(
                llm, self.config.executor_tools,
                system_prompt, task_prompt
            )

        # 无工具，直接调用LLM
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt)
        ])
        return response.content

    async def _execute_with_tools(
        self,
        llm,
        tools: List,
        system_prompt: str,
        task_prompt: str,
        max_iterations: int = 5
    ) -> str:
        """带工具调用的执行循环"""
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

        # 绑定工具（支持 tool_choice）
        tool_choice = getattr(self.config, "tool_mode", None)
        if tool_choice:
            llm_with_tools = llm.bind_tools(tools, tool_choice=tool_choice)
        else:
            llm_with_tools = llm.bind_tools(tools)

        # 构建工具名称到工具的映射
        tool_map = {t.name: t for t in tools}

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt)
        ]

        saw_any_tool_call = False

        for _ in range(max_iterations):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            # 检查是否有工具调用
            if not response.tool_calls:
                if tool_choice == "required":
                    # 某些网关/兼容实现可能忽略 tool_choice，此处强制失败避免“凭空编造”
                    raise RuntimeError(
                        "tool_mode='required' 但模型未产生 tool_calls；为避免幻觉输出，已终止该步骤。"
                    )
                return response.content

            saw_any_tool_call = True

            # 执行工具调用
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in tool_map:
                    try:
                        tool = tool_map[tool_name]
                        tool_args = _normalize_tool_args(tool, tool_args)
                        # 异步或同步调用
                        if hasattr(tool, 'ainvoke'):
                            result = await tool.ainvoke(tool_args)
                        else:
                            result = tool.invoke(tool_args)
                        tool_result = str(result)
                    except Exception as e:
                        tool_result = f"工具执行错误: {e}"
                else:
                    tool_result = f"未知工具: {tool_name}"

                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                ))

        # 达到最大迭代，返回最后的内容
        if tool_choice == "required" and not saw_any_tool_call:
            raise RuntimeError(
                "tool_mode='required' 但未发生任何工具调用；为避免幻觉输出，已终止该步骤。"
            )
        return messages[-1].content if messages else "执行超时"


class PlanExecuteStrategy(ExecutionStrategy):
    """
    Plan-and-Execute (Replanning) 策略
    
    动态生成和调整计划。执行一步后评估结果，决定是继续执行、
    调整计划还是完成任务。
    
    流程:
    1. 调用 Planner 生成初始计划
    2. (可选) Human-in-the-Loop: 等待用户审批计划
    3. 执行当前步骤
    4. (可选) Human-in-the-Loop: 用户确认/干预
    5. 调用 Replanner 评估并决定下一步
    6. 循环 3-5 直到完成
    """
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        from flowagent.state import PlanningState
        
        log.info(f"[PlanExecuteStrategy] 执行 {self.agent.role_name}")
        
        # 确保状态类型正确
        if not isinstance(state, PlanningState):
            log.warning("状态不是 PlanningState 类型，可能会缺少某些功能")
        
        replanning_count = 0
        max_rounds = self.config.max_replanning_rounds
        
        # Step 1: 生成初始计划
        log.info("[PlanExecuteStrategy] Step 1: 生成初始计划")
        plan = await self._generate_plan(state)
        
        if not plan:
            return {"error": "无法生成计划", "status": "failed"}
        
        log.info(f"[PlanExecuteStrategy] 生成了 {len(plan)} 步计划")
        
        # 更新状态
        if hasattr(state, 'plan'):
            state.plan = plan
        
        # Step 2: Human-in-the-Loop - 计划审批
        if self.config.require_plan_approval:
            log.info("[PlanExecuteStrategy] Step 2: 等待用户审批计划")
            from langgraph.types import interrupt
            
            approval_response = interrupt({
                "type": "plan_approval",
                "message": "请审批以下计划:",
                "plan": plan,
                "options": ["approve", "reject", "modify"]
            })
            
            if approval_response.get("action") == "reject":
                return {
                    "status": "rejected",
                    "message": "计划被用户拒绝",
                    "plan": plan
                }
            elif approval_response.get("action") == "modify":
                plan = approval_response.get("modified_plan", plan)
                if hasattr(state, 'plan'):
                    state.plan = plan
        
        if hasattr(state, 'plan_approved'):
            state.plan_approved = True
        
        # Step 3-5: 执行循环
        current_step_index = 0
        results = []
        
        while current_step_index < len(plan) and replanning_count <= max_rounds:
            current_step = plan[current_step_index]
            log.info(f"[PlanExecuteStrategy] 执行步骤 {current_step_index + 1}/{len(plan)}: {current_step[:50]}...")
            
            # Human-in-the-Loop: 执行前中断
            if self.config.interrupt_before_step:
                from langgraph.types import interrupt
                
                step_approval = interrupt({
                    "type": "step_approval",
                    "message": f"即将执行步骤 {current_step_index + 1}:",
                    "step": current_step,
                    "past_steps": results,
                    "remaining_steps": plan[current_step_index:],
                    "options": ["continue", "skip", "abort", "modify"]
                })
                
                if step_approval.get("action") == "abort":
                    return {
                        "status": "aborted",
                        "message": "用户中止执行",
                        "results": results
                    }
                elif step_approval.get("action") == "skip":
                    current_step_index += 1
                    continue
                elif step_approval.get("action") == "modify":
                    current_step = step_approval.get("modified_step", current_step)
            
            # 执行步骤
            try:
                step_result = await self._execute_step(state, current_step, current_step_index)
                results.append({
                    "step_index": current_step_index,
                    "step": current_step,
                    "result": step_result,
                    "status": "completed"
                })
                
                # 更新状态
                if hasattr(state, 'past_steps'):
                    state.past_steps.append((current_step, step_result))
                    
            except Exception as e:
                log.error(f"[PlanExecuteStrategy] 步骤执行失败: {e}")
                results.append({
                    "step_index": current_step_index,
                    "step": current_step,
                    "error": str(e),
                    "status": "failed"
                })
                
                if self.config.auto_replan_on_error:
                    if hasattr(state, 'is_replanning_needed'):
                        state.is_replanning_needed = True
                elif not self.config.continue_on_error:
                    return {
                        "status": "failed",
                        "error": str(e),
                        "results": results
                    }
            
            # Human-in-the-Loop: 执行后中断
            if self.config.interrupt_after_step:
                from langgraph.types import interrupt
                
                post_step = interrupt({
                    "type": "step_completed",
                    "message": f"步骤 {current_step_index + 1} 执行完成",
                    "step": current_step,
                    "result": results[-1] if results else None,
                    "options": ["continue", "replan", "abort"]
                })
                
                if post_step.get("action") == "abort":
                    return {
                        "status": "aborted",
                        "message": "用户中止执行",
                        "results": results
                    }
                elif post_step.get("action") == "replan":
                    if hasattr(state, 'is_replanning_needed'):
                        state.is_replanning_needed = True
            
            # Step 5: Replanner 决策
            replan_decision = await self._replan_decision(state, plan, current_step_index, results)
            
            if replan_decision["action"] == "finish":
                log.info("[PlanExecuteStrategy] Replanner 判断任务已完成")
                return {
                    "status": "completed",
                    "response": replan_decision.get("response", ""),
                    "plan": plan,
                    "results": results
                }
            elif replan_decision["action"] == "replan":
                log.info("[PlanExecuteStrategy] 触发重规划")
                replanning_count += 1
                
                if hasattr(state, 'replanning_count'):
                    state.replanning_count = replanning_count
                
                # Human-in-the-Loop: 重规划确认
                if self.config.interrupt_on_replan:
                    from langgraph.types import interrupt
                    
                    replan_approval = interrupt({
                        "type": "replan_approval",
                        "message": "Replanner 建议调整计划",
                        "reason": replan_decision.get("reason", ""),
                        "new_plan": replan_decision.get("new_plan", []),
                        "options": ["approve", "reject", "modify"]
                    })
                    
                    if replan_approval.get("action") == "reject":
                        # 继续原计划
                        current_step_index += 1
                        continue
                    elif replan_approval.get("action") == "modify":
                        replan_decision["new_plan"] = replan_approval.get("modified_plan", 
                                                                          replan_decision.get("new_plan", []))
                
                # 应用新计划
                new_plan = replan_decision.get("new_plan", [])
                if new_plan:
                    plan = new_plan
                    if hasattr(state, 'plan'):
                        state.plan = plan
                    current_step_index = 0  # 重置到新计划的开始
                else:
                    current_step_index += 1
            else:
                # 继续下一步
                current_step_index += 1
        
        # 检查是否因为达到最大轮数而退出
        if replanning_count > max_rounds:
            return {
                "status": "max_rounds_reached",
                "message": f"达到最大重规划轮数 ({max_rounds})",
                "results": results
            }
        
        return {
            "status": "completed",
            "plan": plan,
            "results": results
        }
    
    async def _generate_plan(self, state: "MainState") -> List[str]:
        """
        使用 LLM 生成计划 (与 PlanSolveStrategy 类似)
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field
        
        task = state.request.target if hasattr(state, 'request') else ""
        if hasattr(state, 'original_task') and state.original_task:
            task = state.original_task
        
        tools_info = ""
        if hasattr(state, 'executor_tools') and state.executor_tools:
            tools_info = f"\n可用工具: {', '.join(state.executor_tools)}"
        
        # 如果有历史执行记录，包含在上下文中
        history_info = ""
        if hasattr(state, 'past_steps') and state.past_steps:
            history_info = "\n\n已执行的步骤:\n"
            for step, result in state.past_steps:
                history_info += f"- {step}: {result[:100]}...\n"
        
        system_prompt = """你是一个任务规划专家。你的职责是分析用户的任务，并生成一个清晰、可执行的分步计划。

规则:
1. 每个步骤应该是独立的、可执行的操作
2. 步骤之间应该有逻辑顺序
3. 步骤描述应该清晰具体
4. 如果有已执行的步骤，基于其结果规划后续步骤
5. 步骤数量不宜过多，通常 3-7 步为宜"""

        task_prompt = f"""请为以下任务生成执行计划:

任务: {task}
{tools_info}
{history_info}

请以 JSON 格式返回计划，格式如下:
{{"steps": ["步骤1描述", "步骤2描述", ...]}}"""

        planner_model = self.config.planner_model or self.config.model_name or state.request.model
        llm = ChatOpenAI(
            openai_api_base=self.config.chat_api_url or state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=planner_model,
            temperature=self.config.planner_temperature,
        )
        
        class PlanOutput(BaseModel):
            steps: List[str] = Field(description="计划步骤列表")
        
        try:
            structured_llm = llm.with_structured_output(PlanOutput, method="function_calling")
            response = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=task_prompt)
            ])
            return response.steps[:self.config.max_plan_steps]
        except Exception as e:
            log.error(f"[PlanExecuteStrategy] 生成计划失败: {e}")
            return []
    
    async def _execute_step(self, state: "MainState", step: str, step_index: int) -> str:
        """
        执行单个计划步骤（支持工具调用）
        """
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
        from langchain_openai import ChatOpenAI

        context = ""
        if hasattr(state, 'past_steps') and state.past_steps:
            context = "\n已完成的步骤:\n"
            for prev_step, prev_result in state.past_steps:
                context += f"- {prev_step}: {prev_result[:100]}...\n"

        system_prompt = """你是一个任务执行专家。你需要执行给定的步骤。
    如果有可用工具，必须至少调用一个工具完成任务。"""

        task_prompt = f"""请执行以下步骤:

步骤 {step_index + 1}: {step}
{context}

请执行这个步骤。"""

        executor_model = self.config.executor_model or self.config.model_name or state.request.model
        llm = ChatOpenAI(
            openai_api_base=self.config.chat_api_url or state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=executor_model,
            temperature=self.config.executor_temperature,
        )

        # 如果有工具，进入工具调用循环
        if self.config.executor_tools:
            return await self._execute_with_tools(
                llm, self.config.executor_tools,
                system_prompt, task_prompt
            )

        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt)
        ])
        return response.content

    async def _execute_with_tools(
        self, llm, tools: List,
        system_prompt: str, task_prompt: str,
        max_iterations: int = 5
    ) -> str:
        """带工具调用的执行循环"""
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

        tool_choice = getattr(self.config, "tool_mode", None)
        if tool_choice:
            llm_with_tools = llm.bind_tools(tools, tool_choice=tool_choice)
        else:
            llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt)
        ]

        saw_any_tool_call = False

        for _ in range(max_iterations):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                if tool_choice == "required":
                    raise RuntimeError(
                        "tool_mode='required' 但模型未产生 tool_calls；为避免幻觉输出，已终止该步骤。"
                    )
                return response.content

            saw_any_tool_call = True

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in tool_map:
                    try:
                        tool = tool_map[tool_name]
                        tool_args = _normalize_tool_args(tool, tool_args)
                        if hasattr(tool, 'ainvoke'):
                            result = await tool.ainvoke(tool_args)
                        else:
                            result = tool.invoke(tool_args)
                        tool_result = str(result)
                    except Exception as e:
                        tool_result = f"工具执行错误: {e}"
                else:
                    tool_result = f"未知工具: {tool_name}"

                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                ))

        if tool_choice == "required" and not saw_any_tool_call:
            raise RuntimeError(
                "tool_mode='required' 但未发生任何工具调用；为避免幻觉输出，已终止该步骤。"
            )
        return messages[-1].content if messages else "执行超时"
    
    async def _replan_decision(
        self, 
        state: "MainState", 
        current_plan: List[str], 
        current_index: int,
        results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Replanner: 决定是继续执行、重规划还是完成
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field
        from typing import Union, Literal
        
        task = state.request.target if hasattr(state, 'request') else ""
        if hasattr(state, 'original_task') and state.original_task:
            task = state.original_task
        
        # 构建执行历史
        history = ""
        for r in results:
            status = r.get("status", "unknown")
            if status == "completed":
                history += f"✓ {r['step']}: {r['result'][:100]}...\n"
            else:
                history += f"✗ {r['step']}: 失败 - {r.get('error', 'unknown')}\n"
        
        # 剩余计划
        remaining = current_plan[current_index + 1:]
        remaining_str = "\n".join(f"- {s}" for s in remaining) if remaining else "无"
        
        system_prompt = """你是一个任务评估专家。根据任务目标和已执行的步骤，你需要决定:
1. finish - 任务已完成，可以返回最终结果
2. continue - 继续执行剩余计划
3. replan - 需要调整计划

只有当任务目标已经达成时才选择 finish。
如果执行结果显示需要调整后续步骤，选择 replan 并提供新计划。
否则选择 continue。"""

        task_prompt = f"""原始任务: {task}

已执行步骤和结果:
{history}

剩余计划:
{remaining_str}

请评估并决定下一步行动。如果选择 replan，请提供新的计划步骤。

以 JSON 格式返回:
{{"action": "finish|continue|replan", "reason": "决策原因", "response": "最终回答(仅finish时)", "new_plan": ["新步骤"](仅replan时)}}"""

        replanner_model = self.config.replanner_model or self.config.model_name or state.request.model
        llm = ChatOpenAI(
            openai_api_base=self.config.chat_api_url or state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=replanner_model,
            temperature=self.config.replanner_temperature,
        )
        
        class ReplanDecision(BaseModel):
            action: Literal["finish", "continue", "replan"] = Field(description="决策动作")
            reason: str = Field(description="决策原因")
            response: Optional[str] = Field(default=None, description="最终回答，仅当 action=finish 时")
            new_plan: Optional[List[str]] = Field(default=None, description="新计划，仅当 action=replan 时")
        
        try:
            structured_llm = llm.with_structured_output(ReplanDecision, method="function_calling")
            decision = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=task_prompt)
            ])
            
            return {
                "action": decision.action,
                "reason": decision.reason,
                "response": decision.response,
                "new_plan": decision.new_plan
            }
        except Exception as e:
            log.error(f"[PlanExecuteStrategy] Replanner 决策失败: {e}")
            # 默认继续执行
            return {"action": "continue", "reason": f"决策失败: {e}"}


class StrategyFactory:
    """策略工厂"""

    _strategies = {
        "simple": SimpleStrategy,
        "validation_retry": ValidationRetryStrategy,  # 原react，验证重试模式
        "react": ReactStrategy,                       # 原graph，真正的ReAct模式
        "vlm": VLMStrategy,
        "parallel": ParallelStrategy,
        "plan_solve": PlanSolveStrategy,
        "plan_execute": PlanExecuteStrategy,
    }

    @classmethod
    def create(cls, mode: str, agent: "BaseAgent", config: Any) -> ExecutionStrategy:
        # 处理废弃的模式名称
        if mode.lower() == "graph":
            warnings.warn(
                "mode='graph' 已废弃，请使用 mode='react'。将在未来版本中移除。",
                DeprecationWarning,
                stacklevel=2
            )
            mode = "react"

        strategy_cls = cls._strategies.get(mode.lower())
        if not strategy_cls:
            raise ValueError(f"不支持的执行模式: {mode}，可选: {list(cls._strategies.keys())}")
        return strategy_cls(agent, config)
    
    @classmethod
    def register(cls, mode: str, strategy_cls: type):
        """注册自定义策略"""
        cls._strategies[mode.lower()] = strategy_cls
