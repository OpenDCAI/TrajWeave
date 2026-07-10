from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.local import extract_final_int
from trajweave.backends.policy import PolicyRequest, PolicyResponse
from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.math import MathTask, SolverVerifierMathEnvironment
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.agentflow import AgentFlowPlannerToolOrchestra
from trajweave.orchestration.maporl_debate import MAPoRLDebateOrchestra
from trajweave.orchestration.search_answer import SearchAnswerOrchestra
from trajweave.orchestration.solver_verifier import SolverVerifierOrchestra
from trajweave.recipes.agentflow.planner_tool import default_agentflow_team
from trajweave.recipes.doctor_mas.math_smoke import default_team
from trajweave.recipes.doctor_mas.search_smoke import default_search_team
from trajweave.recipes.maporl.debate_math import default_debate_team
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


@dataclass
class HFLocalWorkerPolicyBackend:
    worker: Any
    session_id: int
    validate: bool = False
    call_index: int = 0

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        prompt_ids = self.worker._encode_prompt_text(request.prompt)
        seed = self.session_id * 100_003 + self.call_index
        self.call_index += 1
        response_ids = self.worker._generate_local_response_ids(
            prompt_ids,
            policy_group=request.agent.policy_group,
            sample_seed=seed,
            validate=self.validate,
            prompt_text=request.prompt,
        )
        model_path = (
            self.worker._maporl_worker_group_model_path(request.agent.policy_group)
            or self.worker.model_config.local_path
        )
        return PolicyResponse(
            text=self.worker._decode_response_ids(response_ids),
            token_ids=response_ids,
            metadata={
                "prompt_ids": prompt_ids,
                "rollout_source": "hf_local_tq",
                "policy_version": self.worker._local_policy_version(),
                "worker_group_model_path": str(model_path),
            },
        )


def build_hf_workflow_outputs(
    worker: Any,
    *,
    recipe: str,
    prompt: dict[str, Any],
    session_id: int,
    validate: bool = False,
) -> list[AgentLoopOutput]:
    task_id = str(to_python(prompt.get("uid", prompt.get("index", "task"))))
    raw_prompt = to_python(prompt.get("raw_prompt", []))
    question = _question_from_prompt(raw_prompt)
    reward_model = to_python(prompt.get("reward_model", {})) or {}
    ground_truth = str(reward_model.get("ground_truth", ""))
    backend = HFLocalWorkerPolicyBackend(worker=worker, session_id=session_id, validate=validate)

    if recipe == "doctor_mas_math":
        task = MathTask(task_id=task_id, question=question, answer=_integer_ground_truth(ground_truth))
        max_turns = _drmas_math_max_turns(worker.config)
        team = default_team(max_turns=max_turns)
        trajectory = _run_protocol(
            task=task,
            team=team,
            protocol=SolverVerifierOrchestra(),
            environment=SolverVerifierMathEnvironment(),
            backend=backend,
            session_id=session_id,
        )
    elif recipe == "doctor_mas_search":
        extra_info = to_python(prompt.get("extra_info", {})) or {}
        task = SearchTask(
            task_id=task_id,
            question=question,
            answer=ground_truth,
            search_query=str(extra_info.get("search_query") or question),
            documents=_search_documents(extra_info),
        )
        team = default_search_team(max_turns=_drmas_search_max_turns(worker.config))
        trajectory = _run_protocol(
            task=task,
            team=team,
            protocol=SearchAnswerOrchestra(),
            environment=SearchAnswerEnvironment(),
            backend=backend,
            session_id=session_id,
        )
    elif recipe == "maporl_debate_math":
        agent_ids = worker._maporl_agent_ids()
        model_ids = worker._maporl_model_ids(default_agent_ids=agent_ids)
        max_rounds = worker._maporl_max_rounds()
        task = MathTask(task_id=task_id, question=question, answer=_integer_ground_truth(ground_truth))
        team = default_debate_team(
            agent_count=len(agent_ids),
            max_turns=max_rounds,
            agent_ids=tuple(agent_ids),
            model_ids=tuple(model_ids),
            policy_separation=worker._maporl_policy_separation(),
            collaboration_separation=worker._maporl_collaboration_separation(),
            task_training=worker._maporl_task_training(),
        )
        trajectory = _run_protocol(
            task=task,
            team=team,
            protocol=MAPoRLDebateOrchestra(
                consensus_threshold=worker._maporl_consensus_threshold(default=len(agent_ids)),
                early_stop=worker._maporl_early_stop(),
                reward_feedback=worker._maporl_reward_feedback(),
                criteria_for_consensus_percentage=worker._maporl_consensus_percentage(),
                criteria_for_consensus_reward_threshold=worker._maporl_consensus_reward_threshold(),
            ),
            environment=SolverVerifierMathEnvironment(),
            backend=backend,
            session_id=session_id,
        )
    elif recipe == "agentflow_planner_tool":
        max_steps = worker._agentflow_max_steps()
        task = MathTask(task_id=task_id, question=question, answer=_integer_ground_truth(ground_truth))
        team = default_agentflow_team(max_steps=max_steps)
        trajectory = _run_protocol(
            task=task,
            team=team,
            protocol=AgentFlowPlannerToolOrchestra(tool_name=worker._agentflow_enabled_tools()[0]),
            environment=SolverVerifierMathEnvironment(),
            backend=backend,
            session_id=session_id,
        )
    else:
        raise ValueError(f"Unsupported HF workflow recipe: {recipe}")

    return _trajectory_to_outputs(worker, trajectory=trajectory, team=team)


def _run_protocol(
    *,
    task: Any,
    team: TeamSpec,
    protocol: Any,
    environment: Any,
    backend: HFLocalWorkerPolicyBackend,
    session_id: int,
) -> MultiAgentTrajectory:
    observation = environment.initial_observation(task)
    trajectory = protocol.run(
        episode_id=f"{task.task_id}_{session_id}",
        rollout_group=task.task_id,
        task=task,
        team=team,
        observation=observation,
        policy_backend=backend,
        environment=environment,
    )
    reward, success = environment.evaluate(task, trajectory.final_answer)
    trajectory.global_reward = float(reward)
    trajectory.success = bool(success)
    return trajectory


def _trajectory_to_outputs(worker: Any, *, trajectory: MultiAgentTrajectory, team: TeamSpec) -> list[AgentLoopOutput]:
    trainable_agents = {agent.name for agent in team.agents if agent.trainable}
    trainable_turns = [turn for turn in trajectory.turns if turn.agent_name in trainable_agents]
    if not trainable_turns:
        raise RuntimeError(f"Workflow {trajectory.team_name} produced no trainable turns.")

    metrics = AgentLoopMetrics(
        generate_sequences=float(len(trainable_turns)),
        tool_calls=float(sum(turn.role == "tool" for turn in trajectory.turns)),
        compute_score=0.0,
        num_preempted=-1,
    )
    outputs = []
    for output_index, turn in enumerate(trainable_turns):
        metadata = dict(turn.metadata)
        prompt_ids = [int(item) for item in metadata.pop("prompt_ids", worker._encode_prompt_text(turn.prompt))]
        metadata.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "trajweave_agent_name": turn.agent_name,
                "trajweave_role": turn.role,
                "agent_id": turn.agent_name,
                "policy_group": turn.policy_group,
                "worker_group": turn.policy_group,
                "traj_uid": trajectory.episode_id,
                "workflow_success": bool(trajectory.success),
                "final_answer": trajectory.final_answer,
                "prompt_text": turn.prompt,
                "response_text": turn.action_text,
                "observation_text": turn.observation,
                "policy_version": worker._local_policy_version(),
            }
        )
        if trajectory.team_name == "agentflow_planner_tool":
            step_id = metadata.get("step_id")
            metadata["agentflow_trace"] = [
                _trace_event(item)
                for item in trajectory.turns
                if item.agent_name not in trainable_agents and item.metadata.get("step_id") == step_id
            ]
        outputs.append(
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=[int(item) for item in turn.action_token_ids],
                response_mask=[1] * len(turn.action_token_ids),
                reward_score=float(trajectory.global_reward or 0.0),
                num_turns=output_index + 1,
                metrics=metrics,
                extra_fields=metadata,
            )
        )
    return outputs


def _trace_event(turn: AgentTurn) -> dict[str, Any]:
    return {
        "agent_name": turn.agent_name,
        "role": turn.role,
        "policy_group": turn.policy_group,
        "agent_id": turn.agent_name,
        "action_text": turn.action_text,
        **turn.metadata,
    }


def _question_from_prompt(raw_prompt: Any) -> str:
    if isinstance(raw_prompt, list):
        for message in reversed(raw_prompt):
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message["content"])
    if isinstance(raw_prompt, dict) and raw_prompt.get("content") is not None:
        return str(raw_prompt["content"])
    return str(raw_prompt)


def _integer_ground_truth(value: str) -> int:
    parsed = extract_final_int(value)
    if parsed is None:
        raise ValueError(f"Math workflow requires an integer ground_truth, got {value!r}.")
    return parsed


def _search_documents(extra_info: dict[str, Any]) -> tuple[SearchDocument, ...]:
    raw_documents = extra_info.get("documents") or extra_info.get("search_documents") or []
    documents = []
    for item in raw_documents:
        item = to_python(item)
        if isinstance(item, dict) and item.get("text"):
            documents.append(SearchDocument(title=str(item.get("title", "Document")), text=str(item["text"])))
    if not documents:
        raise ValueError(
            "Search workflow requires extra_info.documents; ground_truth is never injected as search evidence."
        )
    return tuple(documents)


def _drmas_math_max_turns(config: Any) -> int:
    agent_cfg = config_get(config, "agent", {}) or {}
    orchestra_cfg = config_get(agent_cfg, "orchestra", {}) or {}
    math_cfg = config_get(orchestra_cfg, "math", {}) or {}
    return int(config_get(math_cfg, "max_loop_num", 3))


def _drmas_search_max_turns(config: Any) -> int:
    agent_cfg = config_get(config, "agent", {}) or {}
    orchestra_cfg = config_get(agent_cfg, "orchestra", {}) or {}
    search_cfg = config_get(orchestra_cfg, "search", {}) or {}
    return int(config_get(search_cfg, "max_loop_num", 2))
