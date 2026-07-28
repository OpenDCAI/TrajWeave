from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.local import extract_final_int
from trajweave.backends.policy import PolicyRequest, PolicyResponse
from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.credit.atgrpo import apply_mixed_reward
from trajweave.credit.comas import CoMASInteractionCreditAssigner
from trajweave.envs.base import evaluate_trajectory
from trajweave.envs.comas import CoMASMathEnvironment
from trajweave.envs.math import MathTask, SolverVerifierMathEnvironment
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.agentflow import AgentFlowPlannerToolOrchestra
from trajweave.orchestration.comas import CoMASPeerReviewOrchestra
from trajweave.orchestration.gigpo import GiGPOSolverVerifierOrchestra
from trajweave.orchestration.maporl_debate import MAPoRLDebateOrchestra
from trajweave.orchestration.matpo import PlannerWorkerOrchestra
from trajweave.orchestration.search_answer import SearchAnswerOrchestra
from trajweave.orchestration.solver_verifier import SolverVerifierOrchestra
from trajweave.recipes.agentflow.planner_tool import default_agentflow_team
from trajweave.recipes.atgrpo.solver_verifier_math import default_atgrpo_team
from trajweave.recipes.comas.peer_review_math import CoMASRulePolicyBackend, default_comas_team
from trajweave.recipes.doctor_mas.math_smoke import default_team
from trajweave.recipes.doctor_mas.search_smoke import default_search_team
from trajweave.recipes.gigpo.solver_verifier_math import default_gigpo_team
from trajweave.recipes.maporl.debate_math import default_debate_team
from trajweave.recipes.matpo.smoke import default_team as default_matpo_team
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
        resolver = getattr(self.worker, "_worker_group_model_path", None)
        if resolver is None:
            resolver = self.worker._maporl_worker_group_model_path
        model_path = resolver(request.agent.policy_group) or self.worker.model_config.local_path
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
    elif recipe == "matpo_browse":
        extra_info = to_python(prompt.get("extra_info", {})) or {}
        task = SearchTask(
            task_id=task_id,
            question=question,
            answer=ground_truth,
            search_query=str(extra_info.get("search_query") or question),
            documents=_search_documents(extra_info),
        )
        planner_agent = _matpo_planner_agent(worker.config)
        worker_agent = _matpo_worker_agent(worker.config)
        tool_name = _matpo_tool_name(worker.config)
        team = default_matpo_team(
            max_turns=_matpo_max_turns(worker.config),
            planner_agent=planner_agent,
            worker_agent=worker_agent,
            tool_name=tool_name,
        )
        trajectory = _run_protocol(
            task=task,
            team=team,
            protocol=PlannerWorkerOrchestra(
                planner_name=planner_agent,
                worker_name=worker_agent,
                tool_name=tool_name,
            ),
            environment=SearchAnswerEnvironment(),
            backend=backend,
            session_id=session_id,
        )
    elif recipe == "gigpo_solver_verifier_math":
        task = MathTask(task_id=task_id, question=question, answer=_integer_ground_truth(ground_truth))
        team = default_gigpo_team(max_steps=_gigpo_max_steps(worker.config))
        trajectory = _run_protocol(
            task=task,
            team=team,
            protocol=GiGPOSolverVerifierOrchestra(),
            environment=SolverVerifierMathEnvironment(),
            backend=backend,
            session_id=session_id,
        )
        _annotate_sparse_step_rewards(trajectory, team=team)
    elif recipe == "atgrpo_solver_verifier_math":
        task = MathTask(task_id=task_id, question=question, answer=_integer_ground_truth(ground_truth))
        team = default_atgrpo_team(max_turns=_atgrpo_max_turns(worker.config))
        trajectory = _run_protocol(
            task=task,
            team=team,
            protocol=SolverVerifierOrchestra(),
            environment=SolverVerifierMathEnvironment(),
            backend=backend,
            session_id=session_id,
        )
        mixed_reward = _atgrpo_mixed_reward_settings(worker.config)
        if mixed_reward["enabled"]:
            apply_mixed_reward(
                trajectory,
                team,
                alpha=mixed_reward["alpha"],
                verifier_local_reward_scale=mixed_reward["verifier_local_reward"],
            )
    elif recipe == "comas_peer_review_math":
        task = MathTask(task_id=task_id, question=question, answer=_integer_ground_truth(ground_truth))
        team, protocol, environment = _comas_runtime_components(worker)
        trajectory = _run_protocol(
            task=task,
            team=team,
            protocol=protocol,
            environment=environment,
            backend=backend,
            session_id=session_id,
        )
        CoMASInteractionCreditAssigner().allocate_trajectory(trajectory)
    else:
        raise ValueError(f"Unsupported HF workflow recipe: {recipe}")

    return _trajectory_to_outputs(worker, trajectory=trajectory, team=team)


def _run_protocol(
    *,
    task: Any,
    team: TeamSpec,
    protocol: Any,
    environment: Any,
    backend: Any,
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
    reward, success = evaluate_trajectory(environment, task, trajectory)
    trajectory.global_reward = float(reward)
    trajectory.success = bool(success)
    return trajectory


def _trajectory_to_outputs(worker: Any, *, trajectory: MultiAgentTrajectory, team: TeamSpec) -> list[AgentLoopOutput]:
    trainable_agents = {agent.name for agent in team.agents if agent.trainable}
    trainable_turns = [turn for turn in trajectory.turns if turn.agent_name in trainable_agents]
    if trajectory.team_name == "matpo_planner_worker_browse":
        # Keep the worker's tool-call turns, the planner's converged final-answer turn,
        # and every planner delegation turn (there can be more than one across rounds
        # when max_turns > 1) so its delegation text -- which now actually drives the
        # worker's search query -- receives gradient signal.
        trainable_turns = [
            turn
            for turn in trainable_turns
            if bool(turn.metadata.get("is_from_subagent_tool"))
            or bool(turn.done)
            or turn.metadata.get("matpo_turn_role") == "delegate"
        ]
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
                "workflow_evaluation_reward": float(trajectory.global_reward or 0.0),
                "final_answer": trajectory.final_answer,
                "prompt_text": turn.prompt,
                "response_text": turn.action_text,
                "observation_text": turn.observation,
                "policy_version": worker._local_policy_version(),
            }
        )
        if turn.anchor_observation is not None:
            metadata["anchor_obs"] = _canonical_transition_value(turn.anchor_observation)
            metadata["next_obs"] = _canonical_transition_value(turn.next_observation)
            metadata["step_reward"] = float(turn.step_reward or 0.0)
            metadata["active_mask"] = 1.0
        if trajectory.team_name == "atgrpo_solver_verifier_math":
            # AT-GRPO's ATGRPOHooks groups advantages by (rollout_group, turn_id, agent_id),
            # so turn_id must be the orchestra's absolute turn index (SolverVerifierOrchestra's
            # own counter), not agent_loop.py's default of "position within trainable turns".
            # Both agents are currently trainable so the two indices happen to coincide, but
            # this keeps the field correct if a non-trainable agent is ever added to the team.
            metadata["turn_id"] = turn.turn_id
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
                reward_score=_turn_training_reward(turn, trajectory=trajectory),
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


def _matpo_orchestra_config(config: Any) -> Any:
    agent_cfg = config_get(config, "agent", {}) or {}
    orchestra_cfg = config_get(agent_cfg, "orchestra", {}) or {}
    return config_get(orchestra_cfg, "matpo", {}) or {}


def _matpo_max_turns(config: Any) -> int:
    matpo_cfg = _matpo_orchestra_config(config)
    return int(config_get(matpo_cfg, "max_turns", 3))


def _matpo_planner_agent(config: Any) -> str:
    matpo_cfg = _matpo_orchestra_config(config)
    return str(config_get(matpo_cfg, "planner_agent", "planner"))


def _matpo_worker_agent(config: Any) -> str:
    matpo_cfg = _matpo_orchestra_config(config)
    return str(config_get(matpo_cfg, "worker_agent", "browsing_agent"))


def _matpo_tool_name(config: Any) -> str:
    matpo_cfg = _matpo_orchestra_config(config)
    return str(config_get(matpo_cfg, "tool_name", "search_and_browse"))


def _gigpo_max_steps(config: Any) -> int:
    agent_cfg = config_get(config, "agent", {}) or {}
    orchestra_cfg = config_get(agent_cfg, "orchestra", {}) or {}
    gigpo_cfg = config_get(orchestra_cfg, "gigpo", {}) or {}
    return int(config_get(gigpo_cfg, "max_steps", 2))


def _atgrpo_max_turns(config: Any) -> int:
    agent_cfg = config_get(config, "agent", {}) or {}
    orchestra_cfg = config_get(agent_cfg, "orchestra", {}) or {}
    atgrpo_cfg = config_get(orchestra_cfg, "atgrpo", {}) or {}
    return int(config_get(atgrpo_cfg, "max_turns", 3))


def _atgrpo_mixed_reward_settings(config: Any) -> dict[str, Any]:
    agent_cfg = config_get(config, "agent", {}) or {}
    orchestra_cfg = config_get(agent_cfg, "orchestra", {}) or {}
    atgrpo_cfg = config_get(orchestra_cfg, "atgrpo", {}) or {}
    mixed_reward_cfg = config_get(atgrpo_cfg, "mixed_reward", {}) or {}
    return {
        "enabled": bool(config_get(mixed_reward_cfg, "enabled", False)),
        "alpha": float(config_get(mixed_reward_cfg, "alpha", 1.0)),
        "verifier_local_reward": float(config_get(mixed_reward_cfg, "verifier_local_reward", 1.0)),
    }


def build_rule_comas_workflow_outputs(
    worker: Any,
    *,
    prompt: dict[str, Any],
    session_id: int,
) -> list[AgentLoopOutput]:
    task_id = str(to_python(prompt.get("uid", prompt.get("index", "task"))))
    raw_prompt = to_python(prompt.get("raw_prompt", []))
    question = _question_from_prompt(raw_prompt)
    reward_model = to_python(prompt.get("reward_model", {})) or {}
    task = MathTask(
        task_id=task_id,
        question=question,
        answer=_integer_ground_truth(str(reward_model.get("ground_truth", ""))),
    )
    team, protocol, environment = _comas_runtime_components(worker)
    backend = _WorkerEncodedPolicyBackend(worker=worker, delegate=CoMASRulePolicyBackend())
    trajectory = _run_protocol(
        task=task,
        team=team,
        protocol=protocol,
        environment=environment,
        backend=backend,
        session_id=session_id,
    )
    CoMASInteractionCreditAssigner().allocate_trajectory(trajectory)
    return _trajectory_to_outputs(worker, trajectory=trajectory, team=team)


@dataclass
class _WorkerEncodedPolicyBackend:
    worker: Any
    delegate: Any

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        response = self.delegate.generate(request)
        response.token_ids = self.worker._encode_text(response.text)
        response.logprobs = [0.0] * len(response.token_ids)
        response.metadata["rollout_source"] = "synthetic_tq"
        return response


def _comas_runtime_components(worker: Any) -> tuple[TeamSpec, CoMASPeerReviewOrchestra, CoMASMathEnvironment]:
    agent_ids = worker._comas_agent_ids()
    model_ids = worker._comas_model_ids(default_agent_ids=agent_ids)
    num_rounds = worker._comas_num_rounds()
    team = default_comas_team(
        agent_ids=tuple(agent_ids),
        model_ids=tuple(model_ids),
        num_rounds=num_rounds,
    )
    protocol = CoMASPeerReviewOrchestra(
        num_rounds=num_rounds,
        num_references=worker._comas_num_references(),
        task_name=worker._comas_task_name(),
        assignment_seed=worker._comas_assignment_seed(),
    )
    return team, protocol, CoMASMathEnvironment()


def _turn_training_reward(turn: AgentTurn, *, trajectory: MultiAgentTrajectory) -> float:
    if turn.reward is not None:
        return float(turn.reward)
    return float(trajectory.global_reward or 0.0)


def _annotate_sparse_step_rewards(trajectory: MultiAgentTrajectory, *, team: TeamSpec) -> None:
    trainable_agents = {agent.name for agent in team.trainable_agents()}
    trainable_turns = trajectory.trainable_turns(trainable_agents)
    for turn in trainable_turns:
        turn.step_reward = 0.0
    if trainable_turns:
        trainable_turns[-1].step_reward = float(trajectory.global_reward or 0.0)


def _canonical_transition_value(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
