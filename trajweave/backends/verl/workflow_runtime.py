from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from trajweave.backends.local import RuleBasedMathPolicyBackend, extract_final_int
from trajweave.backends.policy import PolicyRequest, PolicyResponse
from trajweave.backends.verl.emitters.comlrl import (
    _ACTOR_CRITIC_ALGORITHMS,
    _PREFERENCE_ALGORITHMS,
    _REINFORCE_ALGORITHMS,
    CoMLRLEmitterMixin,
)
from trajweave.backends.verl.multi_actor.critic_config import resolve_actor_critic_settings
from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import required_ground_truth, resolve_comlrl_extra_fields, to_python
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory, TrainingSample
from trajweave.credit.atgrpo import apply_mixed_reward
from trajweave.credit.comas import CoMASInteractionCreditAssigner
from trajweave.credit.comlrl import (
    CoMLRLReinforceCreditAssigner,
    build_joint_preference_pairs,
    preference_pairs_to_training_samples,
)
from trajweave.credit.comlrl.iterative import compare_policy_candidates_by_index, select_policy_comparisons
from trajweave.credit.marft import apply_marft_trajectory_credit
from trajweave.credit.matpo.parent_broadcast import apply_matpo_trajectory_reward
from trajweave.credit.mrlx import apply_mrlx_trajectory_rewards
from trajweave.envs.base import evaluate_trajectory
from trajweave.envs.comas import CoMASMathEnvironment
from trajweave.envs.comlrl import JointMathEnvironment
from trajweave.envs.math import MathTask, SolverVerifierMathEnvironment
from trajweave.envs.math.c3 import C3MathEnvironment, C3MathTask
from trajweave.envs.search import SearchAnswerEnvironment, SearchDocument, SearchTask
from trajweave.orchestration.agentflow import AgentFlowPlannerToolOrchestra
from trajweave.orchestration.atgrpo import SelectedSpineSolverVerifierOrchestra
from trajweave.orchestration.c3 import C3PrefixTreeOrchestra
from trajweave.orchestration.comas import CoMASPeerReviewOrchestra
from trajweave.orchestration.comlrl import FullJointTreeBuilder
from trajweave.orchestration.comlrl.comparator import (
    FrozenPolicySnapshot,
    PolicyComparator,
    RemoteComparator,
    canonical_comparator_policy,
)
from trajweave.orchestration.gigpo import GiGPOSolverVerifierOrchestra
from trajweave.orchestration.maporl_debate import MAPoRLDebateOrchestra
from trajweave.orchestration.marft import MARFTWorkflowGraph, MARFTWorkflowOrchestra
from trajweave.orchestration.matpo import PlannerWorkerOrchestra
from trajweave.orchestration.mrlx import MrlXResearchOrchestra
from trajweave.orchestration.search_answer import SearchAnswerOrchestra
from trajweave.orchestration.solver_verifier import SolverVerifierOrchestra
from trajweave.recipes.agentflow.planner_tool import default_agentflow_team
from trajweave.recipes.atgrpo.solver_verifier_math import default_atgrpo_team
from trajweave.recipes.c3.math_prefix import default_c3_team
from trajweave.recipes.comas.peer_review_math import CoMASRulePolicyBackend, default_comas_team
from trajweave.recipes.doctor_mas.math_smoke import default_team
from trajweave.recipes.doctor_mas.search_smoke import default_search_team
from trajweave.recipes.gigpo.solver_verifier_math import default_gigpo_team
from trajweave.recipes.maporl.debate_math import default_debate_team
from trajweave.recipes.marft.config import load_reward_callable
from trajweave.recipes.marft.math_workflow import DEFAULT_ROLE_PROMPTS, default_marft_team
from trajweave.recipes.matpo.smoke import default_team as default_matpo_team
from trajweave.recipes.mrlx.research_qa import default_mrlx_team
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
            generation_config=request.agent.generation_config,
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
    ground_truth = required_ground_truth(prompt)
    backend = HFLocalWorkerPolicyBackend(worker=worker, session_id=session_id, validate=validate)

    if recipe == "comlrl_joint_math":
        task = MathTask(task_id=task_id, question=question, answer=_integer_ground_truth(ground_truth))
        return _build_comlrl_outputs(
            worker,
            task=task,
            prompt=prompt,
            session_id=session_id,
            policy_backend=backend,
            validate=validate,
        )
    if recipe == "c3_reasoner_actor_math":
        task = C3MathTask(task_id=task_id, question=question, answer=ground_truth)
        return _build_c3_outputs(
            worker,
            task=task,
            session_id=session_id,
            policy_backend=backend,
            validate=validate,
        )
    if recipe == "marft_math_workflow":
        task = MathTask(task_id=task_id, question=question, answer=ground_truth)
        team, protocol, environment = _marft_runtime_components(worker)
        trajectory = _run_protocol(
            task=task,
            team=team,
            protocol=protocol,
            environment=environment,
            backend=backend,
            session_id=session_id,
        )
        _apply_marft_runtime_credit(worker, trajectory=trajectory, environment=environment, prompt=prompt)
    elif recipe == "doctor_mas_math":
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
        _apply_matpo_training_reward(trajectory, config=worker.config)
    elif recipe == "mrlx_research_qa":
        extra_info = to_python(prompt.get("extra_info", {})) or {}
        task = SearchTask(
            task_id=task_id,
            question=question,
            answer=ground_truth,
            search_query=str(extra_info.get("search_query") or question),
            documents=_search_documents(extra_info),
        )
        explorer_agent, adapter_agent = worker._mrlx_agent_ids()
        explorer_group, adapter_group = worker._mrlx_model_ids()
        orchestra_config = worker._mrlx_orchestra_config()
        research_rounds = int(config_get(orchestra_config, "research_rounds", 1))
        team = default_mrlx_team(
            explorer_agent=explorer_agent,
            adapter_agent=adapter_agent,
            explorer_model_id=explorer_group,
            adapter_model_id=adapter_group,
            tool_name=worker._mrlx_tool_name(),
            research_rounds=research_rounds,
        )
        trajectory = _run_protocol(
            task=task,
            team=team,
            protocol=MrlXResearchOrchestra(
                explorer_name=explorer_agent,
                adapter_name=adapter_agent,
                tool_name=worker._mrlx_tool_name(),
                research_rounds=research_rounds,
            ),
            environment=SearchAnswerEnvironment(),
            backend=backend,
            session_id=session_id,
        )
        apply_mrlx_trajectory_rewards(
            trajectory,
            explorer_agent=explorer_agent,
            adapter_agent=adapter_agent,
            explorer_format_bonus=worker._mrlx_explorer_format_bonus(),
            adapter_format_bonus=worker._mrlx_adapter_format_bonus(),
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
        branch_factor = int(to_python(prompt.get("__atgrpo_branch_factor__", 1)))
        trajectory = _run_atgrpo_selected_spine(
            task=task,
            team=team,
            environment=SolverVerifierMathEnvironment(),
            backend=backend,
            session_id=session_id,
            branch_factor=branch_factor,
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


def _run_atgrpo_selected_spine(
    *,
    task: MathTask,
    team: TeamSpec,
    environment: SolverVerifierMathEnvironment,
    backend: Any,
    session_id: int,
    branch_factor: int,
) -> MultiAgentTrajectory:
    observation = environment.initial_observation(task)
    trajectory = SelectedSpineSolverVerifierOrchestra().run_tree(
        episode_id=f"{task.task_id}_{session_id}",
        rollout_group=task.task_id,
        task=task,
        team=team,
        observation=observation,
        policy_backend=backend,
        environment=environment,
        branch_factor=branch_factor,
    )
    reward, success = evaluate_trajectory(environment, task, trajectory)
    trajectory.global_reward = float(reward)
    trajectory.success = bool(success)
    return trajectory


def build_synthetic_c3_workflow_outputs(
    worker: Any,
    *,
    prompt: dict[str, Any],
    session_id: int,
    validate: bool = False,
) -> list[AgentLoopOutput]:
    task_id = str(to_python(prompt.get("uid", prompt.get("index", "task"))))
    raw_prompt = to_python(prompt.get("raw_prompt", []))
    task = C3MathTask(
        task_id=task_id,
        question=_question_from_prompt(raw_prompt),
        answer=required_ground_truth(prompt),
    )
    backend = _WorkerEncodedPolicyBackend(worker=worker, delegate=RuleBasedMathPolicyBackend())
    return _build_c3_outputs(
        worker,
        task=task,
        session_id=session_id,
        policy_backend=backend,
        validate=validate,
    )


def _build_c3_outputs(
    worker: Any,
    *,
    task: C3MathTask,
    session_id: int,
    policy_backend: Any,
    validate: bool,
) -> list[AgentLoopOutput]:
    team = default_c3_team(model_ids=worker._c3_model_ids())
    configured_fanout = worker._c3_fanout()
    # The upstream C3 evaluation default is one sample per prompt. Expanding the
    # training tree here and selecting its highest-reward leaf would leak the
    # ground truth into validation and report oracle best-of-tree accuracy.
    fanout = tuple(1 for _ in configured_fanout) if validate else configured_fanout
    episode_id = f"{task.task_id}_{session_id}:c3"
    trajectory = C3PrefixTreeOrchestra(fanout=fanout, allow_singleton_fanout=validate).run_tree(
        episode_id=episode_id,
        rollout_group=task.task_id,
        task=task,
        team=team,
        observation=task.question,
        policy_backend=policy_backend,
        environment=C3MathEnvironment(),
    )
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
    _annotate_actor_critic_metadata(worker, trajectory=trajectory, team=team)
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
    comlrl_horizon = _comlrl_trajectory_horizon(trajectory) if trajectory.team_name == "comlrl_joint_math" else None
    for output_index, turn in enumerate(trainable_turns):
        metadata = _metadata_from_turn_or_sample(turn)
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
        if turn.root_id is not None:
            metadata.update(
                {
                    "root_id": turn.root_id,
                    "node_id": turn.node_id,
                    "parent_node_id": turn.parent_node_id or "",
                    "observation_group_id": turn.observation_group_id,
                    "branch_index": int(turn.branch_index or 0),
                    "selected_for_expansion": bool(turn.selected_for_expansion),
                    "local_score": float(turn.local_score or 0.0),
                }
            )
        if trajectory.team_name in {
            "atgrpo_solver_verifier_math",
            "c3_reasoner_actor_math",
            "marft_math_workflow",
        }:
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
        metadata.update(
            resolve_comlrl_extra_fields(
                metadata,
                row_id=f"{trajectory.episode_id}:{turn.turn_id}:{turn.agent_name}",
            )
        )
        reward_score = _turn_training_reward(turn, trajectory=trajectory)
        outputs.append(
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=[int(item) for item in turn.action_token_ids],
                response_mask=[1] * len(turn.action_token_ids),
                reward_score=reward_score,
                num_turns=comlrl_horizon or (output_index + 1),
                metrics=metrics,
                extra_fields=metadata,
            )
        )
    return outputs


def _annotate_actor_critic_metadata(
    worker: Any,
    *,
    trajectory: MultiAgentTrajectory,
    team: TeamSpec,
) -> None:
    if trajectory.team_name == "c3_reasoner_actor_math":
        return
    if _actor_critic_runtime_settings(getattr(worker, "config", None)) is None:
        return
    trainable_names = {agent.name for agent in team.agents if agent.trainable}
    trainable_turns = [turn for turn in trajectory.turns if turn.agent_name in trainable_names]
    transition_ids = {id(turn): _single_actor_critic_transition_id(turn) for turn in trainable_turns}
    fallback_turns: dict[str, int] = {}
    for turn in trainable_turns:
        transition_id = transition_ids[id(turn)]
        fallback_turns[transition_id] = min(fallback_turns.get(transition_id, turn.turn_id), turn.turn_id)
    for turn in trainable_turns:
        transition_id = transition_ids[id(turn)]
        fields = _joint_transition_fields(trajectory, turn, transition_id)
        fields.setdefault("turn_id", fallback_turns[transition_id])
        turn.metadata.update(fields)


def _actor_critic_runtime_settings(config: Any) -> dict[str, Any] | None:
    return resolve_actor_critic_settings(config)


def _single_actor_critic_transition_id(turn: AgentTurn) -> str:
    if len(turn.joint_action_ids) != 1 or len(turn.joint_transition_ids) != 1:
        raise ValueError("IAC/MAAC training turns must reference exactly one joint action and one joint transition.")
    transition_id = str(turn.joint_transition_ids[0])
    if not transition_id:
        raise ValueError("IAC/MAAC joint transition IDs must be non-empty.")
    return transition_id


def _joint_transition_fields(
    trajectory: MultiAgentTrajectory,
    turn: AgentTurn,
    transition_id: str,
) -> dict[str, Any]:
    transitions = {transition.joint_transition_id: transition for transition in trajectory.joint_transitions}
    actions = {action.joint_action_id: action for action in trajectory.joint_actions}
    transition = transitions.get(transition_id)
    if transition is not None:
        action = actions.get(transition.joint_action_id)
        reward = action.shared_reward if action is not None else None
        return {
            "turn_id": int(transition.source_turn),
            "joint_reward": float(reward or 0.0),
            "joint_done": bool(transition.done),
            "joint_truncated": bool(transition.truncated),
            "joint_stop_reason": transition.stop_reason or "",
        }
    return {
        "joint_reward": float(
            turn.metadata.get(
                "joint_reward",
                turn.reward if turn.reward is not None else trajectory.global_reward or 0.0,
            )
        ),
        "joint_done": bool(turn.metadata.get("joint_done", turn.done)),
        "joint_truncated": bool(turn.metadata.get("joint_truncated", False)),
        "joint_stop_reason": str(turn.metadata.get("joint_stop_reason", "")),
    }


def _metadata_from_turn_or_sample(source: AgentTurn | TrainingSample) -> dict[str, Any]:
    metadata = dict(source.metadata)
    for field in ("completion_id", "tree_node_id", "joint_action_ids", "joint_transition_ids"):
        metadata[field] = to_python(getattr(source, field))
    return metadata


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


def _apply_matpo_training_reward(trajectory: MultiAgentTrajectory, *, config: Any) -> None:
    matpo_cfg = _matpo_orchestra_config(config)
    combined_reward = apply_matpo_trajectory_reward(
        trajectory,
        accuracy_reward_weight=float(config_get(matpo_cfg, "accuracy_reward_weight", 0.9)),
        tool_format_reward_weight=float(config_get(matpo_cfg, "tool_format_reward_weight", 0.1)),
    )
    final_planner = next(
        (
            turn
            for turn in reversed(trajectory.turns)
            if not bool(turn.metadata.get("is_from_subagent_tool", False))
            and bool(turn.done)
            and turn.metadata.get("matpo_turn_role") in {"final", "invalid_planner_call", "invalid_tool"}
        ),
        None,
    )
    if final_planner is None:
        raise RuntimeError("MATPO workflow did not produce exactly one final planner row.")
    reward_metadata = {
        "matpo_accuracy_reward": trajectory.metadata["matpo_accuracy_reward"],
        "matpo_planner_format": trajectory.metadata["matpo_planner_format"],
        "matpo_worker_formats": trajectory.metadata["matpo_worker_formats"],
        "matpo_combined_reward": combined_reward,
    }
    for turn in trajectory.turns:
        turn.reward = combined_reward
        turn.metadata.update(reward_metadata)


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


def build_synthetic_comlrl_workflow_outputs(
    worker: Any,
    *,
    prompt: dict[str, Any],
    session_id: int,
    validate: bool = False,
) -> list[AgentLoopOutput]:
    task_id = str(to_python(prompt.get("uid", prompt.get("index", "task"))))
    raw_prompt = to_python(prompt.get("raw_prompt", []))
    reward_model = to_python(prompt.get("reward_model", {})) or {}
    task = MathTask(
        task_id=task_id,
        question=_question_from_prompt(raw_prompt),
        answer=_integer_ground_truth(str(reward_model.get("ground_truth", ""))),
    )
    backend = _SyntheticCoMLRLPolicyBackend(worker=worker, answer=task.answer)
    return _build_comlrl_outputs(
        worker,
        task=task,
        prompt=prompt,
        session_id=session_id,
        policy_backend=backend,
        validate=validate,
    )


@dataclass
class _SyntheticCoMLRLPolicyBackend:
    worker: Any
    answer: int

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        candidate_index = int(request.metadata["candidate_index"])
        answer = self.answer if candidate_index == 0 else self.answer + candidate_index
        text = f"Final answer: {answer}"
        token_ids = _encode_worker_text(self.worker, text)
        return PolicyResponse(
            text=text,
            token_ids=token_ids,
            logprobs=[0.0] * len(token_ids),
            metadata={
                "prompt_ids": _encode_worker_prompt(self.worker, request.prompt),
                "rollout_source": "synthetic_tq",
                "policy_version": _worker_policy_version(self.worker),
            },
        )


def _build_comlrl_outputs(
    worker: Any,
    *,
    task: MathTask,
    prompt: dict[str, Any],
    session_id: int,
    policy_backend: Any,
    validate: bool = False,
) -> list[AgentLoopOutput]:
    algorithm = _comlrl_call(worker, "_comlrl_algorithm")
    # CoMLRL v1.4.1 evaluation samples exactly one completion per agent and
    # follows that single joint path. Training keeps the configured K/tree.
    num_candidates = 1 if validate else _comlrl_call(worker, "_comlrl_num_candidates", prompt)
    joint_mode = _comlrl_call(worker, "_comlrl_joint_mode")
    agent_ids = _comlrl_call(worker, "_comlrl_agent_ids")
    model_ids = _comlrl_call(worker, "_comlrl_model_ids", default_agent_ids=agent_ids)
    max_turns = _comlrl_call(worker, "_comlrl_max_turns")
    team = _comlrl_team(worker, agent_ids=agent_ids, model_ids=model_ids, max_turns=max_turns, algorithm=algorithm)
    iterative_context = _comlrl_call(worker, "_comlrl_iterative_context")
    preference_phase = iterative_context.get("phase") == "preference" and not validate
    score_preferences_with_rm = (
        preference_phase and iterative_context.get("preference_scoring_reward") == "reward_model"
    )
    scorer = (
        None
        if validate or (preference_phase and not score_preferences_with_rm)
        else _comlrl_call(worker, "_comlrl_marlhf_reward_scorer", team)
    )
    environment = JointMathEnvironment(evaluate_task_reward=scorer is None)
    episode_id = f"{task.task_id}_{session_id}"
    tree = FullJointTreeBuilder(
        num_candidates=num_candidates,
        joint_mode=joint_mode,
        max_turns=max_turns,
        max_joint_actions=_comlrl_call(worker, "_comlrl_optional_positive_int", "max_joint_actions"),
        max_tree_nodes=_comlrl_call(worker, "_comlrl_optional_positive_int", "max_tree_nodes"),
        early_stop_threshold=None if validate else _comlrl_call(worker, "_comlrl_early_stop_threshold"),
    ).build(
        root_id=f"{episode_id}:root",
        task=task,
        team=team,
        observations_by_agent=environment.initial_observations(task, agent_ids),
        policy_backend=policy_backend,
        environment=environment,
    )
    if scorer is not None:
        _apply_marlhf_joint_rewards(tree, team=team, scorer=scorer)

    trajectory = MultiAgentTrajectory(
        episode_id=episode_id,
        task_id=task.task_id,
        rollout_group=task.task_id,
        team_name=team.name,
        metadata={"comlrl_algorithm": algorithm, "validation_agent_id": agent_ids[0]},
    )
    tree.attach_to_trajectory(trajectory)
    if validate:
        final_action = tree.joint_actions[-1]
        trajectory.global_reward = float(final_action.shared_reward or 0.0)
        trajectory.success = bool(final_action.done)
        trajectory.final_answer = _comlrl_action_answer(tree, final_action, agent_ids[0])
    else:
        rewards = [float(action.shared_reward or 0.0) for action in tree.joint_actions]
        trajectory.global_reward = max(rewards, default=0.0)
        trajectory.success = any(action.done for action in tree.joint_actions)
        trajectory.final_answer = _comlrl_final_answer(tree, agent_ids[0])

    if preference_phase:
        return _build_iterative_comparator_outputs(
            worker,
            task=task,
            team=team,
            trajectory=trajectory,
            policy_backend=policy_backend,
            environment=environment,
            scorer=scorer,
            context=iterative_context,
        )

    if algorithm in _REINFORCE_ALGORITHMS:
        advantage_mode = {
            "magrpo": "mean",
            "mareinforce": "raw",
            "marloo": "rloo",
            "maremax": "max",
        }[algorithm]
        samples = CoMLRLReinforceCreditAssigner(
            advantage_mode=advantage_mode,
            normalize=_comlrl_call(worker, "_comlrl_normalize_advantages"),
            sequence_kl_coefficient=_comlrl_call(worker, "_comlrl_sequence_kl_coefficient"),
        ).assign([trajectory], team)
        outputs = _comlrl_training_samples_to_outputs(worker, trajectory=trajectory, samples=samples)
        return _finalize_comlrl_outputs(worker, trajectory=trajectory, outputs=outputs, validate=validate)
    if algorithm in _ACTOR_CRITIC_ALGORITHMS:
        _attach_linear_comlrl_turns(trajectory, team=team)
        outputs = _trajectory_to_outputs(worker, trajectory=trajectory, team=team)
        return _finalize_comlrl_outputs(worker, trajectory=trajectory, outputs=outputs, validate=validate)
    if algorithm in _PREFERENCE_ALGORITHMS:
        selection, limit, random_seed = _comlrl_call(worker, "_comlrl_preference_pair_settings")
        pairs = build_joint_preference_pairs(
            trajectory,
            pair_selection=selection,
            pairs_per_sample=limit,
            random_seed=random_seed,
        )
        samples = preference_pairs_to_training_samples(trajectory, pairs, team)
        if samples:
            outputs = _comlrl_training_samples_to_outputs(worker, trajectory=trajectory, samples=samples)
        else:
            outputs = _standard_empty_preference_outputs(worker, trajectory=trajectory, team=team)
        return _finalize_comlrl_outputs(worker, trajectory=trajectory, outputs=outputs, validate=validate)
    raise AssertionError(f"Unhandled CoMLRL algorithm: {algorithm}")


@dataclass
class _CurrentComparatorProvider:
    backend: Any
    team: TeamSpec
    task_id: str
    candidate_offset: int

    def generate(self, prompt, *, agent_index, num_candidates, context=None):
        del context
        agent = self.team.agents[agent_index]
        output = []
        for candidate_index in range(num_candidates):
            response = self.backend.generate(
                PolicyRequest(
                    agent=agent,
                    task_id=self.task_id,
                    observation=prompt,
                    prompt=prompt,
                    metadata={
                        "candidate_index": self.candidate_offset + candidate_index,
                        "comparator": True,
                    },
                )
            )
            output.append(response.text)
        return output


@dataclass
class _SnapshotComparatorProvider:
    worker: Any
    team: TeamSpec
    model_paths: tuple[str, ...]
    iteration: int

    def generate(self, prompt, *, agent_index, num_candidates, context=None):
        del context
        if len(self.model_paths) != len(self.team.agents):
            raise ValueError("iterative comparator model_paths must cover every agent")
        group_id = f"__comparator__:{self.iteration}:{agent_index}"
        overrides = getattr(self.worker, "_trajweave_model_path_overrides", {})
        overrides = dict(overrides)
        overrides[group_id] = self.model_paths[agent_index]
        self.worker._trajweave_model_path_overrides = overrides
        prompt_ids = _encode_worker_prompt(self.worker, prompt)
        output = []
        for candidate_index in range(num_candidates):
            response_ids = self.worker._generate_local_response_ids(
                prompt_ids,
                policy_group=group_id,
                sample_seed=self.iteration * 100_003 + agent_index * 10_003 + candidate_index,
                validate=False,
                prompt_text=prompt,
            )
            output.append(self.worker._decode_response_ids(response_ids))
        return output


def _build_iterative_comparator_outputs(
    worker: Any,
    *,
    task: MathTask,
    team: TeamSpec,
    trajectory: MultiAgentTrajectory,
    policy_backend: Any,
    environment: JointMathEnvironment,
    scorer: Any,
    context: dict[str, Any],
) -> list[AgentLoopOutput]:
    if int(team.max_turns) != 1:
        raise ValueError("CoMLRL iterative preference generation requires max_turns=1")
    completions = {completion.completion_id: completion for completion in trajectory.joint_completions}
    actions = sorted(
        trajectory.joint_actions,
        key=lambda action: next(iter(action.candidate_indices.values())),
    )
    if not actions:
        raise ValueError("iterative current policy generated no candidates")
    agent_names = tuple(agent.name for agent in team.agents)
    prompts = tuple(
        str(completions[actions[0].completion_ids[name]].metadata.get("prompt", "")) for name in agent_names
    )
    current_candidates = tuple(
        tuple(completions[action.completion_ids[name]].text for action in actions) for name in agent_names
    )
    current_rewards = [float(action.shared_reward or 0.0) for action in actions]
    comparator_config = context.get("comparator", {}) or {}
    if not isinstance(comparator_config, dict):
        raise TypeError("iterative comparator context must be a mapping")
    comparator_policy = canonical_comparator_policy(comparator_config.get("policy", "current"))
    generation_mode = str(comparator_config.get("generation_mode", "decentralized"))
    iteration = int(context.get("iteration", 0))
    comparator_count = int(comparator_config.get("num_candidates", len(actions)))
    current_provider = _CurrentComparatorProvider(policy_backend, team, task.task_id, len(actions))
    if comparator_policy == "api":
        comparator = RemoteComparator(
            url=str(comparator_config.get("url", "")),
            num_agents=len(agent_names),
            api_format=str(comparator_config.get("api_format", "generic")),
            generation_mode=generation_mode,
            model=comparator_config.get("model"),
            timeout=float(comparator_config.get("timeout", 120.0)),
            headers=comparator_config.get("headers"),
            api_key=comparator_config.get("api_key"),
            api_key_env=comparator_config.get("api_key_env"),
            response_field=str(comparator_config.get("response_field", "completions")),
            extra_body=comparator_config.get("extra_body"),
            max_candidates_per_request=comparator_config.get("max_candidates_per_request"),
        )
    elif comparator_policy == "current":
        comparator = PolicyComparator(
            policy="current",
            current_policy=current_provider,
            num_agents=len(agent_names),
            generation_mode=generation_mode,
            centralized_agent_index=int(comparator_config.get("centralized_agent_index", 0)),
        )
    else:
        model_paths = comparator_config.get("model_paths")
        if not model_paths:
            raise ValueError(f"iterative comparator policy {comparator_policy!r} requires frozen model_paths")
        if isinstance(model_paths, dict):
            ordered_paths = tuple(str(model_paths[agent.policy_group]) for agent in team.agents)
        else:
            ordered_paths = tuple(str(path) for path in model_paths)
        snapshot_provider = _SnapshotComparatorProvider(worker, team, ordered_paths, iteration)
        snapshot = FrozenPolicySnapshot(
            snapshot_provider,
            snapshot_id=str(comparator_config.get("snapshot_id", f"{comparator_policy}-{iteration:04d}")),
            metadata={"model_paths": ordered_paths},
        )
        snapshot_kwargs: dict[str, Any] = {}
        if comparator_policy == "current_copy":
            snapshot_kwargs["current_copy"] = snapshot
        elif comparator_policy == "history":
            snapshot_kwargs["history_resolver"] = lambda _iteration: snapshot
        elif comparator_policy == "model":
            snapshot_kwargs["model"] = snapshot
        comparator = PolicyComparator(
            policy=comparator_policy,
            current_policy=current_provider,
            num_agents=len(agent_names),
            generation_mode=generation_mode,
            centralized_agent_index=int(comparator_config.get("centralized_agent_index", 0)),
            **snapshot_kwargs,
        )
    compared = comparator.generate(
        prompts,
        num_candidates=comparator_count,
        iteration=iteration,
        context={"task_id": task.task_id},
    )
    observations = environment.initial_observations(task, agent_names)
    comparator_responses = [
        {
            name: compared.candidates_by_agent[agent_index][candidate_index]
            for agent_index, name in enumerate(agent_names)
        }
        for candidate_index in range(compared.num_candidates)
    ]
    if scorer is not None:
        comparator_prompts = [dict(zip(agent_names, prompts, strict=True))] * len(comparator_responses)
        comparator_rewards = scorer.score_many(comparator_prompts, comparator_responses)
    else:
        comparator_rewards = [
            float(
                environment.step_joint(
                    task,
                    observations,
                    responses,
                    dict.fromkeys(agent_names, ()),
                    0,
                ).team_reward
            )
            for responses in comparator_responses
        ]
    comparisons = compare_policy_candidates_by_index(current_rewards, comparator_rewards)
    selection = str(context.get("pair_selection", "comparator_reward"))
    pair_limit = context.get("pairs_per_sample", 4)
    selected = select_policy_comparisons(
        comparisons,
        mode=selection,
        limit=None if selection == "all" else int(pair_limit),
        seed=int(context.get("random_seed", 0)) + iteration,
    )
    if not selected:
        return _iterative_empty_preference_outputs(
            worker,
            trajectory=trajectory,
            team=team,
            prompts=prompts,
            current_candidates=current_candidates,
            iteration=iteration,
        )
    metrics = AgentLoopMetrics(
        generate_sequences=float(len(selected) * len(agent_names) * 2),
        tool_calls=0.0,
        compute_score=0.0,
        num_preempted=-1,
    )
    candidate_mean = sum((*current_rewards, *comparator_rewards)) / (len(current_rewards) + len(comparator_rewards))
    outputs = []
    for pair_index, comparison in enumerate(selected):
        candidate_index = comparison.candidate_index
        pair_id = f"{trajectory.episode_id}:iter:{iteration}:pair:{candidate_index}:{pair_index}"
        chosen_reward = max(comparison.current_reward, comparison.comparator_reward)
        rejected_reward = min(comparison.current_reward, comparison.comparator_reward)
        for agent_index, agent in enumerate(team.agents):
            current_text = current_candidates[agent_index][candidate_index]
            comparator_text = compared.candidates_by_agent[agent_index][candidate_index]
            for side in ("chosen", "rejected"):
                source = comparison.winner_source if side == "chosen" else comparison.loser_source
                text = current_text if source == "current" else comparator_text
                reward = chosen_reward if side == "chosen" else rejected_reward
                response_ids = _encode_worker_text(worker, text)
                row_id = f"{pair_id}:{agent.name}:{side}"
                metadata = {
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajweave_agent_name": agent.name,
                    "trajweave_role": agent.role,
                    "agent_id": agent.name,
                    "policy_group": agent.policy_group,
                    "worker_group": agent.policy_group,
                    "worker_group_model_path": _worker_group_model_path(worker, agent.policy_group) or "",
                    "traj_uid": trajectory.episode_id,
                    "turn_id": 0,
                    "workflow_success": bool(trajectory.success),
                    "workflow_evaluation_reward": float(trajectory.global_reward or 0.0),
                    "final_answer": trajectory.final_answer,
                    "prompt_text": prompts[agent_index],
                    "response_text": text,
                    "observation_text": prompts[agent_index],
                    "completion_id": row_id,
                    "tree_node_id": f"{trajectory.episode_id}:root",
                    "joint_action_ids": [f"{pair_id}:{side}"],
                    "joint_transition_ids": [f"{pair_id}:{side}:transition"],
                    "joint_return_components": [reward],
                    "projected_joint_return": reward,
                    "effective_projected_joint_return": reward,
                    "joint_reward": reward,
                    "joint_done": True,
                    "joint_truncated": False,
                    "joint_stop_reason": "",
                    "joint_sampling_mode": "aligned",
                    "preference_pair_id": pair_id,
                    "preference_side": side,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "candidate_mean": candidate_mean,
                    "preference_loss_mask": 1.0,
                    "raw_policy_reward": current_rewards[candidate_index],
                    "raw_comparator_reward": comparator_rewards[candidate_index],
                    "raw_candidate_rewards": [*current_rewards, *comparator_rewards],
                    "policy_provenance": {"policy": "current"},
                    "comparator_provenance": {
                        "policy": compared.policy,
                        "generation_mode": compared.generation_mode,
                        **dict(compared.provenance),
                    },
                    "winner_source": comparison.winner_source,
                    "loser_source": comparison.loser_source,
                    "policy_version": _worker_policy_version(worker),
                }
                metadata.update(resolve_comlrl_extra_fields(metadata, row_id=row_id))
                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=_encode_worker_prompt(worker, prompts[agent_index]),
                        response_ids=response_ids,
                        response_mask=[1] * len(response_ids),
                        reward_score=reward,
                        num_turns=1,
                        metrics=metrics,
                        extra_fields=metadata,
                    )
                )
    return outputs


def _iterative_empty_preference_outputs(
    worker: Any,
    *,
    trajectory: MultiAgentTrajectory,
    team: TeamSpec,
    prompts: tuple[str, ...],
    current_candidates: tuple[tuple[str, ...], ...],
    iteration: int,
) -> list[AgentLoopOutput]:
    agent = team.agents[0]
    prompt = prompts[0]
    text = current_candidates[0][0]
    response_ids = _encode_worker_text(worker, text)
    row_id = f"{trajectory.episode_id}:iter:{iteration}:empty"
    metadata = {
        "turn_scores": [],
        "tool_rewards": [],
        "trajweave_agent_name": agent.name,
        "trajweave_role": agent.role,
        "agent_id": agent.name,
        "policy_group": agent.policy_group,
        "worker_group": agent.policy_group,
        "worker_group_model_path": _worker_group_model_path(worker, agent.policy_group) or "",
        "traj_uid": trajectory.episode_id,
        "turn_id": 0,
        "workflow_success": bool(trajectory.success),
        "workflow_evaluation_reward": float(trajectory.global_reward or 0.0),
        "final_answer": trajectory.final_answer,
        "prompt_text": prompt,
        "response_text": text,
        "observation_text": prompt,
        "completion_id": row_id,
        "tree_node_id": f"{trajectory.episode_id}:root",
        "joint_action_ids": [f"{row_id}:action"],
        "joint_transition_ids": [f"{row_id}:transition"],
        "joint_return_components": [0.0],
        "projected_joint_return": 0.0,
        "effective_projected_joint_return": 0.0,
        "joint_reward": 0.0,
        "joint_done": True,
        "joint_truncated": False,
        "joint_stop_reason": "all_tied",
        "joint_sampling_mode": "aligned",
        "preference_pair_id": row_id,
        "preference_side": "__empty__",
        "chosen_reward": 0.0,
        "rejected_reward": 0.0,
        "candidate_mean": 0.0,
        "preference_loss_mask": 0.0,
        "policy_version": _worker_policy_version(worker),
    }
    metadata.update(resolve_comlrl_extra_fields(metadata, row_id=row_id))
    return [
        AgentLoopOutput(
            prompt_ids=_encode_worker_prompt(worker, prompt),
            response_ids=response_ids,
            response_mask=[1] * len(response_ids),
            reward_score=0.0,
            num_turns=1,
            metrics=AgentLoopMetrics(
                generate_sequences=1.0,
                tool_calls=0.0,
                compute_score=0.0,
                num_preempted=-1,
            ),
            extra_fields=metadata,
        )
    ]


def _apply_marlhf_joint_rewards(tree: Any, *, team: TeamSpec, scorer: Any) -> None:
    completions = {completion.completion_id: completion for completion in tree.completions}
    prompts: list[dict[str, str]] = []
    responses: list[dict[str, str]] = []
    for action in tree.joint_actions:
        action_prompts: dict[str, str] = {}
        action_responses: dict[str, str] = {}
        for agent in team.agents:
            completion = completions[action.completion_ids[agent.name]]
            action_prompts[agent.name] = str(completion.metadata.get("prompt", ""))
            action_responses[agent.name] = completion.text
        prompts.append(action_prompts)
        responses.append(action_responses)
    learned_rewards = scorer.score_many(prompts, responses)
    if len(learned_rewards) != len(tree.joint_actions):
        raise RuntimeError("MARLHF reward scorer must return one reward per joint action")
    for action, learned_reward in zip(tree.joint_actions, learned_rewards, strict=True):
        if action.shared_reward is not None:
            action.metadata["task_reward"] = float(action.shared_reward)
        action.shared_reward = float(learned_reward)
        action.metadata["reward_source"] = "marlhf_joint_reward_model"


def _comlrl_team(
    worker: Any,
    *,
    agent_ids: list[str],
    model_ids: list[str],
    max_turns: int,
    algorithm: str,
) -> TeamSpec:
    agents = []
    groups = []
    for agent_id, model_id in zip(agent_ids, model_ids, strict=True):
        model_path = _worker_group_model_path(worker, model_id)
        agents.append(
            AgentSpec(
                name=agent_id,
                role="solver",
                policy_group=model_id,
                trainable=True,
                model_path=model_path,
            )
        )
        groups.append(
            PolicyGroupSpec(
                name=model_id,
                model_path=model_path,
                trainable=True,
                backend="local",
            )
        )
    return TeamSpec(
        name="comlrl_joint_math",
        agents=tuple(agents),
        policy_groups=tuple(groups),
        orchestra="comlrl_full_joint_tree",
        reward="joint_math",
        credit=algorithm,
        max_turns=max_turns,
        metadata={"algorithm": algorithm},
    )


def _comlrl_training_samples_to_outputs(
    worker: Any,
    *,
    trajectory: MultiAgentTrajectory,
    samples: list[TrainingSample],
) -> list[AgentLoopOutput]:
    if not samples:
        raise RuntimeError("CoMLRL rollout produced no trainable completion samples.")
    metrics = AgentLoopMetrics(
        generate_sequences=float(len(samples)),
        tool_calls=0.0,
        compute_score=0.0,
        num_preempted=-1,
    )
    outputs = []
    completions = {completion.completion_id: completion for completion in trajectory.joint_completions}
    horizon = _comlrl_trajectory_horizon(trajectory)
    for sample in samples:
        effective_return = float(
            sample.metadata.get(
                "effective_projected_joint_return",
                sample.reward if sample.reward is not None else 0.0,
            )
        )
        joint_advantage = float(sample.advantage if sample.advantage is not None else 0.0)
        metadata = _metadata_from_turn_or_sample(sample)
        completion = completions[str(sample.completion_id)]
        metadata.update(completion.metadata)
        metadata.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "trajweave_agent_name": sample.agent_name,
                "trajweave_role": sample.role,
                "agent_id": sample.agent_name,
                "policy_group": sample.policy_group,
                "worker_group": sample.policy_group,
                "worker_group_model_path": _worker_group_model_path(worker, sample.policy_group) or "",
                "traj_uid": trajectory.episode_id,
                "turn_id": sample.turn_id,
                "workflow_success": bool(trajectory.success),
                "workflow_evaluation_reward": float(trajectory.global_reward or 0.0),
                "final_answer": trajectory.final_answer,
                "prompt_text": sample.prompt,
                "response_text": sample.response,
                "observation_text": str(sample.metadata.get("observation", "")),
                "joint_advantage": joint_advantage,
                "effective_projected_joint_return": effective_return,
                "policy_version": _worker_policy_version(worker),
            }
        )
        metadata.update(resolve_comlrl_extra_fields(metadata, row_id=sample.sample_id))
        prompt_ids = _encode_worker_prompt(worker, sample.prompt)
        response_ids = list(sample.response_token_ids) or _encode_worker_text(worker, sample.response)
        outputs.append(
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                reward_score=effective_return,
                num_turns=horizon,
                metrics=metrics,
                extra_fields=metadata,
            )
        )
    return outputs


def _standard_empty_preference_outputs(
    worker: Any,
    *,
    trajectory: MultiAgentTrajectory,
    team: TeamSpec,
) -> list[AgentLoopOutput]:
    completions_by_agent = {
        agent.name: sorted(
            (completion for completion in trajectory.joint_completions if completion.agent_name == agent.name),
            key=lambda completion: (completion.candidate_index, completion.completion_id),
        )
        for agent in team.agents
    }
    if any(not completions for completions in completions_by_agent.values()):
        raise RuntimeError("Tied CoMLRL preference rollout is missing an agent completion.")
    prompts = tuple(str(completions_by_agent[agent.name][0].metadata.get("prompt", "")) for agent in team.agents)
    candidates = tuple(
        tuple(completion.text for completion in completions_by_agent[agent.name]) for agent in team.agents
    )
    return _iterative_empty_preference_outputs(
        worker,
        trajectory=trajectory,
        team=team,
        prompts=prompts,
        current_candidates=candidates,
        iteration=0,
    )


def _finalize_comlrl_outputs(
    worker: Any,
    *,
    trajectory: MultiAgentTrajectory,
    outputs: list[AgentLoopOutput],
    validate: bool,
) -> list[AgentLoopOutput]:
    if not outputs:
        raise RuntimeError("CoMLRL rollout produced no TransferQueue outputs.")
    horizon = _comlrl_trajectory_horizon(trajectory)
    for output in outputs:
        output.num_turns = horizon
    if not validate:
        return outputs

    summary_agent = str(trajectory.metadata.get("validation_agent_id", ""))
    summary_index = next(
        (
            index
            for index in range(len(outputs) - 1, -1, -1)
            if str(outputs[index].extra_fields.get("agent_id", "")) == summary_agent
        ),
        len(outputs) - 1,
    )
    if summary_index != len(outputs) - 1:
        outputs.append(outputs.pop(summary_index))
    final_output = outputs[-1]
    evaluation_reward = float(trajectory.global_reward or 0.0)
    final_answer = str(trajectory.final_answer or final_output.extra_fields.get("response_text", ""))
    final_ids = _encode_worker_text(worker, final_answer)
    final_output.response_ids = final_ids
    final_output.response_mask = [1] * len(final_ids)
    final_output.reward_score = evaluation_reward
    final_output.extra_fields["response_text"] = final_answer
    final_output.extra_fields["workflow_evaluation_reward"] = evaluation_reward
    final_output.extra_fields["comlrl_validation_summary"] = True
    return outputs


def _comlrl_trajectory_horizon(trajectory: MultiAgentTrajectory) -> int:
    source_turns = [int(transition.source_turn) for transition in trajectory.joint_transitions]
    source_turns.extend(int(turn.turn_id) for turn in trajectory.turns)
    return max(source_turns, default=0) + 1


def _attach_linear_comlrl_turns(trajectory: MultiAgentTrajectory, *, team: TeamSpec) -> None:
    actions_by_node = {action.tree_node_id: action for action in trajectory.joint_actions}
    transitions = {transition.joint_transition_id: transition for transition in trajectory.joint_transitions}
    nodes = {node.tree_node_id: node for node in trajectory.joint_nodes}
    for completion in trajectory.joint_completions:
        action = actions_by_node[completion.tree_node_id]
        transition = transitions[action.joint_transition_id]
        node = nodes[completion.tree_node_id]
        agent = team.agent(completion.agent_name)
        next_observations = transition.metadata.get("next_observations_by_agent", {})
        trajectory.add_turn(
            AgentTurn(
                episode_id=trajectory.episode_id,
                task_id=trajectory.task_id,
                turn_id=int(transition.source_turn),
                agent_name=completion.agent_name,
                role=agent.role,
                policy_group=agent.policy_group,
                observation=str(completion.metadata.get("observation", "")),
                prompt=str(completion.metadata.get("prompt", "")),
                action_text=completion.text,
                action_token_ids=list(completion.token_ids),
                action_logprobs=list(completion.logprobs),
                next_observation=next_observations.get(completion.agent_name),
                reward=float(action.shared_reward or 0.0),
                done=bool(action.done),
                root_id=node.root_id,
                node_id=node.tree_node_id,
                completion_id=completion.completion_id,
                tree_node_id=completion.tree_node_id,
                joint_action_ids=[action.joint_action_id],
                joint_transition_ids=[transition.joint_transition_id],
                metadata={
                    **dict(completion.metadata),
                    "turn_id": int(transition.source_turn),
                    "joint_reward": float(action.shared_reward or 0.0),
                    "joint_done": bool(action.done),
                    "joint_truncated": bool(action.truncated),
                    "joint_stop_reason": action.stop_reason or "",
                    "joint_sampling_mode": "aligned",
                },
            )
        )


def _comlrl_final_answer(tree: Any, first_agent: str) -> str:
    for action in tree.joint_actions:
        if action.done:
            return _comlrl_action_answer(tree, action, first_agent)
    return ""


def _comlrl_action_answer(tree: Any, action: Any, agent_name: str) -> str:
    completions = {completion.completion_id: completion for completion in tree.completions}
    return completions[action.completion_ids[agent_name]].text


def _comlrl_call(worker: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(worker, method_name, None)
    if method is not None:
        return method(*args, **kwargs)
    return getattr(CoMLRLEmitterMixin, method_name)(worker, *args, **kwargs)


def _worker_group_model_path(worker: Any, group_id: str) -> str | None:
    resolver = getattr(worker, "_worker_group_model_path", None)
    if resolver is None:
        resolver = getattr(worker, "_maporl_worker_group_model_path", None)
    if resolver is None:
        return None
    value = resolver(group_id)
    return str(value) if value else None


def _worker_policy_version(worker: Any) -> int:
    resolver = getattr(worker, "_local_policy_version", None)
    return int(resolver()) if resolver is not None else 0


def _encode_worker_prompt(worker: Any, text: str) -> list[int]:
    encoder = getattr(worker, "_encode_prompt_text", None)
    if encoder is not None:
        return [int(value) for value in encoder(text)]
    encoder = getattr(worker, "_encode_prompt", None)
    if encoder is not None:
        return [int(value) for value in encoder(text)]
    return _encode_worker_text(worker, text)


def _encode_worker_text(worker: Any, text: str) -> list[int]:
    encoder = getattr(worker, "_encode_text", None)
    if encoder is not None:
        return [int(value) for value in encoder(text)]
    tokenizer = getattr(worker, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        raise AttributeError("CoMLRL worker must provide _encode_text or tokenizer.encode.")
    return [int(value) for value in tokenizer.encode(text)]


def build_rule_comas_workflow_outputs(
    worker: Any,
    *,
    prompt: dict[str, Any],
    session_id: int,
) -> list[AgentLoopOutput]:
    task_id = str(to_python(prompt.get("uid", prompt.get("index", "task"))))
    raw_prompt = to_python(prompt.get("raw_prompt", []))
    question = _question_from_prompt(raw_prompt)
    task = MathTask(
        task_id=task_id,
        question=question,
        answer=_integer_ground_truth(required_ground_truth(prompt)),
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


def build_rule_marft_workflow_outputs(
    worker: Any,
    *,
    prompt: dict[str, Any],
    session_id: int,
) -> list[AgentLoopOutput]:
    task_id = str(to_python(prompt.get("uid", prompt.get("index", "task"))))
    task = MathTask(
        task_id=task_id,
        question=_question_from_prompt(to_python(prompt.get("raw_prompt", []))),
        answer=required_ground_truth(prompt),
    )
    team, protocol, environment = _marft_runtime_components(worker)
    backend = _WorkerEncodedPolicyBackend(worker=worker, delegate=RuleBasedMathPolicyBackend())
    trajectory = _run_protocol(
        task=task,
        team=team,
        protocol=protocol,
        environment=environment,
        backend=backend,
        session_id=session_id,
    )
    _apply_marft_runtime_credit(worker, trajectory=trajectory, environment=environment, prompt=prompt)
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


def _marft_runtime_components(
    worker: Any,
) -> tuple[TeamSpec, MARFTWorkflowOrchestra, SolverVerifierMathEnvironment]:
    role_names = worker._marft_role_names()
    model_ids = worker._marft_model_ids(role_names=role_names)
    configured_roles = worker._marft_role_configs()
    role_configs = {
        role: {
            **configured_roles.get(role, {}),
            "system_prompt": configured_roles.get(role, {}).get(
                "system_prompt",
                DEFAULT_ROLE_PROMPTS.get(role, f"Act as the {role} role."),
            ),
        }
        for role in role_names
    }
    graph_config = worker._marft_graph_config()
    graph = MARFTWorkflowGraph.from_config(graph_config) if graph_config else MARFTWorkflowGraph.sequential(role_names)
    unknown_roles = sorted({node.role_name for node in graph.nodes} - set(role_names))
    if unknown_roles:
        raise ValueError(f"MARFT runtime graph references unknown roles: {unknown_roles}.")
    team = default_marft_team(
        role_names=role_names,
        model_ids=model_ids,
        role_configs=role_configs,
    )
    prompts = {role: str(role_configs[role]["system_prompt"]) for role in role_names}
    return team, MARFTWorkflowOrchestra(graph=graph, role_prompts=prompts), SolverVerifierMathEnvironment()


def _apply_marft_runtime_credit(
    worker: Any,
    *,
    trajectory: MultiAgentTrajectory,
    environment: SolverVerifierMathEnvironment,
    prompt: dict[str, Any],
) -> None:
    settings = worker._marft_credit_settings()
    step_reward_path = settings["step_reward_fn"]
    per_agent_paths = settings["per_agent_reward_fns"]
    apply_marft_trajectory_credit(
        trajectory,
        strategy=settings["strategy"],
        discount=settings["discount"],
        gamma=settings["gamma"],
        step_reward_fn=load_reward_callable(step_reward_path) if step_reward_path else None,
        per_agent_reward_fns={role: load_reward_callable(path) for role, path in per_agent_paths.items()},
        environment=environment,
        data=to_python(prompt),
    )


def _turn_training_reward(turn: AgentTurn, *, trajectory: MultiAgentTrajectory) -> float:
    return float(turn.reward) if turn.reward is not None else float(trajectory.global_reward or 0.0)


def _annotate_sparse_step_rewards(trajectory: MultiAgentTrajectory, *, team: TeamSpec) -> None:
    trainable_agents = {agent.name for agent in team.trainable_agents()}
    trainable_turns = trajectory.trainable_turns(trainable_agents)
    for turn in trainable_turns:
        turn.step_reward = 0.0
    if trainable_turns:
        trainable_turns[-1].step_reward = float(trajectory.global_reward or 0.0)


def _canonical_transition_value(value: Any) -> str:

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
