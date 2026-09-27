from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core.joint_trajectory import (
    JointAction,
    JointCompletion,
    JointTransition,
    JointTreeNode,
    validate_joint_trajectory,
)
from trajweave.core.specs import AgentSpec, TeamSpec
from trajweave.envs.comlrl import JointEnvironment
from trajweave.orchestration.comlrl.joint_rollout import (
    canonical_joint_mode,
    compose_joint_action_indices,
)

if TYPE_CHECKING:
    from trajweave.core.trajectory import MultiAgentTrajectory


@dataclass
class JointTreeBuildResult:
    root_id: str
    agent_names: tuple[str, ...]
    nodes: list[JointTreeNode] = field(default_factory=list)
    completions: list[JointCompletion] = field(default_factory=list)
    joint_actions: list[JointAction] = field(default_factory=list)
    transitions: list[JointTransition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        validate_joint_trajectory(
            self.nodes,
            self.completions,
            self.joint_actions,
            self.transitions,
            set(self.agent_names),
        )

    def attach_to_trajectory(self, trajectory: MultiAgentTrajectory) -> MultiAgentTrajectory:
        self.validate()
        trajectory.joint_nodes = list(self.nodes)
        trajectory.joint_completions = list(self.completions)
        trajectory.joint_actions = list(self.joint_actions)
        trajectory.joint_transitions = list(self.transitions)
        trajectory.metadata["joint_tree"] = {
            "root_id": self.root_id,
            "agent_names": list(self.agent_names),
            **self.metadata,
        }
        trajectory.metadata["joint_sampling_mode"] = self.metadata["joint_mode"]
        return trajectory


@dataclass(frozen=True)
class FullJointTreeBuilder:
    num_candidates: int
    joint_mode: str = "aligned"
    max_turns: int | None = None
    max_joint_actions: int | None = None
    max_tree_nodes: int | None = None
    early_stop_threshold: float | None = None

    def build(
        self,
        *,
        root_id: str,
        task: Any,
        team: TeamSpec,
        observations_by_agent: Mapping[str, str],
        policy_backend: PolicyBackend,
        environment: JointEnvironment,
        histories_by_agent: Mapping[str, Sequence[str]] | None = None,
    ) -> JointTreeBuildResult:
        settings = self._validated_settings(team)
        agent_names = tuple(agent.name for agent in team.agents)
        observations = _copy_observations(observations_by_agent, agent_names)
        histories = _copy_histories(histories_by_agent, agent_names)
        if not isinstance(root_id, str) or not root_id:
            raise ValueError("root_id must be a non-empty string")
        task_id = getattr(task, "task_id", None)
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task must expose a non-empty task_id")

        result = JointTreeBuildResult(
            root_id=root_id,
            agent_names=agent_names,
            metadata={
                "joint_mode": settings.joint_mode,
                "num_candidates": self.num_candidates,
                "max_turns": settings.max_turns,
                "max_joint_actions": self.max_joint_actions,
                "max_tree_nodes": self.max_tree_nodes,
                "early_stop_threshold": self.early_stop_threshold,
            },
        )
        root = JointTreeNode(
            tree_node_id=root_id,
            root_id=root_id,
            depth=0,
            metadata={
                "observations_by_agent": observations,
                "histories_by_agent": histories,
            },
        )
        result.nodes.append(root)
        pending = [root]

        while pending:
            node = pending.pop(0)
            combination_count = (
                self.num_candidates if settings.joint_mode == "aligned" else self.num_candidates ** len(agent_names)
            )
            remaining_actions = settings.max_joint_actions - len(result.joint_actions)
            if combination_count > remaining_actions:
                _truncate_unexpanded_node(result, node, stop_reason="max_joint_actions")
                result.metadata["max_joint_actions_reached"] = True
                continue
            node_observations = _copy_observations(node.metadata["observations_by_agent"], agent_names)
            node_histories = _copy_histories(node.metadata["histories_by_agent"], agent_names)
            completions_by_agent = self._generate_candidates(
                node=node,
                task_id=task_id,
                team=team,
                observations_by_agent=node_observations,
                histories_by_agent=node_histories,
                policy_backend=policy_backend,
            )
            for agent_name in agent_names:
                result.completions.extend(completions_by_agent[agent_name])

            action_records: list[tuple[JointAction, JointTransition, dict[str, str], dict[str, str], bool]] = []
            combinations = compose_joint_action_indices(agent_names, self.num_candidates, settings.joint_mode)

            for joint_index, candidate_indices in enumerate(combinations):
                selected = {
                    agent_name: completions_by_agent[agent_name][candidate_indices[agent_offset]]
                    for agent_offset, agent_name in enumerate(agent_names)
                }
                joint_action_id = f"{node.tree_node_id}:joint:{joint_index}"
                transition_id = f"{joint_action_id}:transition"
                actions_by_agent = {name: selected[name].text for name in agent_names}
                step = environment.step_joint(
                    task,
                    dict(node_observations),
                    actions_by_agent,
                    {name: list(node_histories[name]) for name in agent_names},
                    node.depth,
                )
                next_observations = _copy_observations(step.next_observations_by_agent, agent_names)
                next_histories = {name: [*node_histories[name], actions_by_agent[name]] for name in agent_names}
                action = JointAction(
                    joint_action_id=joint_action_id,
                    tree_node_id=node.tree_node_id,
                    completion_ids={name: selected[name].completion_id for name in agent_names},
                    candidate_indices={name: candidate_indices[offset] for offset, name in enumerate(agent_names)},
                    shared_reward=float(step.team_reward),
                    done=bool(step.done),
                    joint_transition_id=transition_id,
                    metadata={"turn_index": node.depth},
                )
                transition = JointTransition(
                    joint_transition_id=transition_id,
                    joint_action_id=joint_action_id,
                    source_tree_node_id=node.tree_node_id,
                    source_turn=node.depth,
                    done=bool(step.done),
                    metadata={
                        **dict(step.metadata),
                        "team_reward": float(step.team_reward),
                        "next_observations_by_agent": next_observations,
                    },
                )
                result.joint_actions.append(action)
                result.transitions.append(transition)
                action_records.append((action, transition, next_observations, next_histories, bool(step.done)))

            rewards = [record[0].shared_reward for record in action_records]
            stopped_early = (
                self.early_stop_threshold is not None
                and bool(rewards)
                and fmean(float(reward) for reward in rewards if reward is not None) > self.early_stop_threshold
            )
            node.metadata["mean_joint_reward"] = (
                fmean(float(reward) for reward in rewards if reward is not None) if rewards else None
            )
            node.metadata["stopped_early"] = stopped_early
            stop_reason = None
            if stopped_early:
                stop_reason = "early_stop"
            elif node.depth + 1 >= settings.max_turns:
                stop_reason = "max_turns"
            if stop_reason is not None:
                node.truncated = True
                node.stop_reason = node.stop_reason or stop_reason
                for action, transition, _next_observations, _next_histories, done in action_records:
                    if not done:
                        action.truncated = True
                        action.stop_reason = stop_reason
                        transition.truncated = True
                        transition.stop_reason = stop_reason
                continue

            for action, transition, next_observations, next_histories, done in action_records:
                if done:
                    continue
                if len(result.nodes) >= settings.max_tree_nodes:
                    action.truncated = True
                    action.stop_reason = "max_tree_nodes"
                    transition.truncated = True
                    transition.stop_reason = "max_tree_nodes"
                    action.metadata["expansion_stopped"] = action.stop_reason
                    transition.metadata["expansion_stopped"] = transition.stop_reason
                    result.metadata["max_tree_nodes_reached"] = True
                    continue
                child_id = f"{action.joint_action_id}:child"
                child = JointTreeNode(
                    tree_node_id=child_id,
                    root_id=root_id,
                    depth=node.depth + 1,
                    parent_joint_action_id=action.joint_action_id,
                    metadata={
                        "observations_by_agent": next_observations,
                        "histories_by_agent": next_histories,
                    },
                )
                action.child_tree_node_id = child_id
                transition.target_tree_node_id = child_id
                transition.target_turn = child.depth
                result.nodes.append(child)
                pending.append(child)

        result.validate()
        return result

    def _generate_candidates(
        self,
        *,
        node: JointTreeNode,
        task_id: str,
        team: TeamSpec,
        observations_by_agent: Mapping[str, str],
        histories_by_agent: Mapping[str, Sequence[str]],
        policy_backend: PolicyBackend,
    ) -> dict[str, list[JointCompletion]]:
        team_context = _render_team_context(histories_by_agent)
        generated: dict[str, list[JointCompletion]] = {}
        for agent_offset, agent in enumerate(team.agents):
            prompt = _build_prompt(
                agent, observations_by_agent[agent.name], histories_by_agent[agent.name], team_context
            )
            candidates: list[JointCompletion] = []
            for candidate_index in range(self.num_candidates):
                response = policy_backend.generate(
                    PolicyRequest(
                        agent=agent,
                        task_id=task_id,
                        observation=observations_by_agent[agent.name],
                        prompt=prompt,
                        team_context=team_context,
                        metadata={
                            "tree_node_id": node.tree_node_id,
                            "turn_index": node.depth,
                            "candidate_index": candidate_index,
                        },
                    )
                )
                completion = JointCompletion(
                    completion_id=(f"{node.tree_node_id}:completion:{agent_offset}:{candidate_index}"),
                    tree_node_id=node.tree_node_id,
                    agent_name=agent.name,
                    candidate_index=candidate_index,
                    text=response.text,
                    token_ids=list(response.token_ids),
                    logprobs=list(response.logprobs),
                    metadata={
                        **dict(response.metadata),
                        "observation": observations_by_agent[agent.name],
                        "prompt": prompt,
                        "turn_index": node.depth,
                    },
                )
                completion.validate()
                candidates.append(completion)
            generated[agent.name] = candidates
        return generated

    def _validated_settings(self, team: TeamSpec) -> _ValidatedSettings:
        if isinstance(self.num_candidates, bool) or not isinstance(self.num_candidates, int):
            raise TypeError("num_candidates must be an integer")
        if self.num_candidates < 1:
            raise ValueError("num_candidates must be positive")
        if not team.agents:
            raise ValueError("team must contain at least one agent")
        agent_names = [agent.name for agent in team.agents]
        if any(not name for name in agent_names) or len(agent_names) != len(set(agent_names)):
            raise ValueError("team agent names must be non-empty and unique")
        joint_mode = canonical_joint_mode(self.joint_mode)
        max_turns = team.max_turns if self.max_turns is None else self.max_turns
        max_joint_actions = self.max_joint_actions
        max_tree_nodes = self.max_tree_nodes
        _validate_positive_limit("max_turns", max_turns)
        if max_joint_actions is None:
            max_joint_actions = _unbounded_tree_limit()
        else:
            _validate_positive_limit("max_joint_actions", max_joint_actions)
            root_action_count = (
                self.num_candidates if joint_mode == "aligned" else self.num_candidates ** len(agent_names)
            )
            if max_joint_actions < root_action_count:
                raise ValueError(
                    "max_joint_actions must fit one complete root joint-action group: "
                    f"required={root_action_count}, configured={max_joint_actions}"
                )
        if max_tree_nodes is None:
            max_tree_nodes = _unbounded_tree_limit()
        else:
            _validate_positive_limit("max_tree_nodes", max_tree_nodes)
        if self.early_stop_threshold is not None and isinstance(self.early_stop_threshold, bool):
            raise TypeError("early_stop_threshold must be numeric or None")
        return _ValidatedSettings(
            joint_mode=joint_mode,
            max_turns=max_turns,
            max_joint_actions=max_joint_actions,
            max_tree_nodes=max_tree_nodes,
        )


@dataclass(frozen=True)
class _ValidatedSettings:
    joint_mode: str
    max_turns: int
    max_joint_actions: int
    max_tree_nodes: int


def _truncate_unexpanded_node(
    result: JointTreeBuildResult,
    node: JointTreeNode,
    *,
    stop_reason: str,
) -> None:
    """Turn an unexpandable pending node back into a terminal incoming edge."""

    parent_action_id = node.parent_joint_action_id
    if parent_action_id is None:
        node.truncated = True
        node.stop_reason = stop_reason
        node.metadata["expansion_stopped"] = stop_reason
        return

    try:
        action = next(item for item in result.joint_actions if item.joint_action_id == parent_action_id)
        transition = next(item for item in result.transitions if item.joint_transition_id == action.joint_transition_id)
    except StopIteration as exc:  # pragma: no cover - internal tree-construction invariant
        raise RuntimeError(f"Pending joint-tree node {node.tree_node_id!r} has no incoming edge.") from exc

    action.child_tree_node_id = None
    action.truncated = True
    action.stop_reason = stop_reason
    action.metadata["expansion_stopped"] = stop_reason
    transition.target_tree_node_id = None
    transition.target_turn = None
    transition.truncated = True
    transition.stop_reason = stop_reason
    transition.metadata["expansion_stopped"] = stop_reason
    result.nodes.remove(node)


def build_full_joint_tree(
    *,
    root_id: str,
    task: Any,
    team: TeamSpec,
    observations_by_agent: Mapping[str, str],
    policy_backend: PolicyBackend,
    environment: JointEnvironment,
    num_candidates: int,
    joint_mode: str = "aligned",
    max_turns: int | None = None,
    max_joint_actions: int | None = None,
    max_tree_nodes: int | None = None,
    early_stop_threshold: float | None = None,
    histories_by_agent: Mapping[str, Sequence[str]] | None = None,
) -> JointTreeBuildResult:
    return FullJointTreeBuilder(
        num_candidates=num_candidates,
        joint_mode=joint_mode,
        max_turns=max_turns,
        max_joint_actions=max_joint_actions,
        max_tree_nodes=max_tree_nodes,
        early_stop_threshold=early_stop_threshold,
    ).build(
        root_id=root_id,
        task=task,
        team=team,
        observations_by_agent=observations_by_agent,
        policy_backend=policy_backend,
        environment=environment,
        histories_by_agent=histories_by_agent,
    )


def _copy_observations(values: Mapping[str, str], agent_names: Sequence[str]) -> dict[str, str]:
    if set(values) != set(agent_names):
        raise ValueError("observations must contain exactly the team's agents")
    copied = {name: values[name] for name in agent_names}
    if any(not isinstance(value, str) for value in copied.values()):
        raise TypeError("each observation must be a string")
    return copied


def _copy_histories(
    values: Mapping[str, Sequence[str]] | None,
    agent_names: Sequence[str],
) -> dict[str, list[str]]:
    if values is None:
        return {name: [] for name in agent_names}
    if set(values) != set(agent_names):
        raise ValueError("histories must contain exactly the team's agents")
    copied: dict[str, list[str]] = {}
    for name in agent_names:
        history = values[name]
        if isinstance(history, (str, bytes)) or any(not isinstance(item, str) for item in history):
            raise TypeError("each history must be a sequence of strings")
        copied[name] = list(history)
    return copied


def _build_prompt(agent: AgentSpec, observation: str, history: Sequence[str], team_context: str) -> str:
    own_history = "\n".join(history)
    if agent.prompt_template:
        return agent.prompt_template.format(
            agent_name=agent.name,
            role=agent.role,
            observation=observation,
            history=own_history,
            team_context=team_context,
        )
    history_block = own_history or "(none)"
    return (
        f"Role: {agent.role}\n\nObservation:\n{observation}\n\n"
        f"Your prior actions:\n{history_block}\n\nRespond with your next action."
    )


def _render_team_context(histories_by_agent: Mapping[str, Sequence[str]]) -> str:
    lines: list[str] = []
    for agent_name, history in histories_by_agent.items():
        for turn_index, action in enumerate(history):
            lines.append(f"{agent_name}[{turn_index}]: {action}")
    return "\n".join(lines)


def _validate_positive_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _unbounded_tree_limit() -> int:
    return 2**63 - 1
