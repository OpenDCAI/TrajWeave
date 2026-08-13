from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product as cartesian_product
from math import isfinite, prod, sqrt
from statistics import fmean

from trajweave.core.joint_trajectory import (
    JointAction,
    JointCompletion,
    JointTreeNode,
    validate_joint_trajectory,
)
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample


@dataclass(frozen=True)
class CompletionReturnProjection:
    completion_id: str
    joint_action_ids: list[str]
    joint_transition_ids: list[str]
    joint_return_components: list[float]
    projected_joint_return: float

    @property
    def component_action_ids(self) -> list[str]:
        return self.joint_action_ids

    @property
    def component_transition_ids(self) -> list[str]:
        return self.joint_transition_ids

    @property
    def component_returns(self) -> list[float]:
        return self.joint_return_components

    @property
    def projected_return(self) -> float:
        return self.projected_joint_return


def compute_joint_tree_returns(
    nodes: Sequence[JointTreeNode],
    actions: Sequence[JointAction],
    discount: float = 1.0,
) -> dict[str, float]:
    """Compute CoMLRL v1.4.1 action returns without mutating the rollout tree."""

    discount_value = _finite_float(discount, "discount")
    if discount_value != 1.0:
        raise ValueError("discount must be exactly 1.0 for CoMLRL v1.4.1 semantics")

    nodes_by_id = _index_unique(nodes, "tree_node_id")
    actions_by_id = _index_unique(actions, "joint_action_id")
    actions_by_node: dict[str, list[JointAction]] = defaultdict(list)

    for action in actions:
        if action.tree_node_id not in nodes_by_id:
            raise ValueError(f"joint action {action.joint_action_id} references unknown source node")
        _finite_float(action.shared_reward, f"shared_reward for {action.joint_action_id}")
        if action.done and action.truncated:
            raise ValueError(f"joint action {action.joint_action_id} cannot be both done and truncated")
        if action.done or action.truncated:
            if action.child_tree_node_id is not None:
                raise ValueError(f"terminal joint action {action.joint_action_id} cannot reference a child node")
        elif action.child_tree_node_id is None:
            raise ValueError(f"non-terminal joint action {action.joint_action_id} has no child node")
        if action.child_tree_node_id is not None:
            child = nodes_by_id.get(action.child_tree_node_id)
            if child is None:
                raise ValueError(f"joint action {action.joint_action_id} references unknown child node")
            if child.parent_joint_action_id != action.joint_action_id:
                raise ValueError(
                    f"child node {child.tree_node_id} does not point back to joint action {action.joint_action_id}"
                )
        actions_by_node[action.tree_node_id].append(action)

    for node in nodes:
        if node.parent_joint_action_id is None:
            continue
        parent = actions_by_id.get(node.parent_joint_action_id)
        if parent is None:
            raise ValueError(f"node {node.tree_node_id} references unknown parent joint action")
        if parent.child_tree_node_id != node.tree_node_id:
            raise ValueError(f"parent joint action {parent.joint_action_id} does not point to node {node.tree_node_id}")

    _reject_cycles(nodes_by_id, actions_by_node)
    returns: dict[str, float] = {}

    def compute_action_return(action: JointAction) -> float:
        existing = returns.get(action.joint_action_id)
        if existing is not None:
            return existing
        immediate = _finite_float(action.shared_reward, f"shared_reward for {action.joint_action_id}")
        child_actions = (
            actions_by_node.get(action.child_tree_node_id, []) if action.child_tree_node_id is not None else []
        )
        child_mean = fmean(compute_action_return(child) for child in child_actions) if child_actions else 0.0
        result = immediate + child_mean
        if not isfinite(result):
            raise ValueError(f"shared return for {action.joint_action_id} must be finite")
        returns[action.joint_action_id] = result
        return result

    for action in actions:
        compute_action_return(action)
    return {action.joint_action_id: returns[action.joint_action_id] for action in actions}


def project_joint_returns_to_completions(
    completions: Sequence[JointCompletion],
    actions: Sequence[JointAction],
    action_returns: Mapping[str, float],
    mode: str,
) -> dict[str, CompletionReturnProjection]:
    """Project action returns onto participating completions in stable input order."""

    sampling_mode = _canonical_joint_mode(mode)
    completions_by_id = _index_unique(completions, "completion_id")
    actions_by_id = _index_unique(actions, "joint_action_id")
    unknown_returns = set(action_returns) - set(actions_by_id)
    if unknown_returns:
        raise ValueError(f"action_returns contains unknown joint actions: {sorted(unknown_returns)}")

    components: dict[str, list[tuple[str, str, float]]] = {completion.completion_id: [] for completion in completions}
    for action in actions:
        if action.joint_action_id not in action_returns:
            raise ValueError(f"missing return for joint action {action.joint_action_id}")
        action_return = _finite_float(action_returns[action.joint_action_id], f"return for {action.joint_action_id}")
        for agent_name, completion_id in action.completion_ids.items():
            completion = completions_by_id.get(completion_id)
            if completion is None:
                raise ValueError(f"joint action {action.joint_action_id} references unknown completion {completion_id}")
            if completion.tree_node_id != action.tree_node_id or completion.agent_name != agent_name:
                raise ValueError(
                    f"completion {completion_id} does not belong to agent {agent_name} at node {action.tree_node_id}"
                )
            components[completion_id].append((action.joint_action_id, action.joint_transition_id, action_return))

    if sampling_mode == "cross":
        candidates_by_node: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        actions_by_node: dict[str, list[JointAction]] = defaultdict(list)
        for completion in completions:
            candidates_by_node[completion.tree_node_id][completion.agent_name].append(completion.completion_id)
        for action in actions:
            actions_by_node[action.tree_node_id].append(action)
        for node_id, per_agent in candidates_by_node.items():
            agent_names = tuple(sorted(per_agent))
            expected_count = prod(len(per_agent[agent_name]) for agent_name in agent_names)
            expected_combinations = set(cartesian_product(*(per_agent[agent_name] for agent_name in agent_names)))
            node_actions = actions_by_node[node_id]
            actual_combinations = [
                tuple(action.completion_ids.get(agent_name, "") for agent_name in agent_names)
                for action in node_actions
            ]
            if len(node_actions) != expected_count or set(actual_combinations) != expected_combinations:
                raise ValueError(
                    f"cross node {node_id} must contain a complete Cartesian grid: "
                    f"expected {expected_count} unique joint actions, got {len(set(actual_combinations))}"
                )

    projections: dict[str, CompletionReturnProjection] = {}
    for completion in completions:
        values = components[completion.completion_id]
        if not values:
            raise ValueError(f"completion {completion.completion_id} does not belong to any joint action")
        if sampling_mode == "aligned" and len(values) > 1:
            raise ValueError(f"aligned completion {completion.completion_id} belongs to more than one joint action")
        component_returns = [value[2] for value in values]
        projections[completion.completion_id] = CompletionReturnProjection(
            completion_id=completion.completion_id,
            joint_action_ids=[value[0] for value in values],
            joint_transition_ids=[value[1] for value in values],
            joint_return_components=component_returns,
            projected_joint_return=sum(component_returns),
        )
    return projections


def compute_group_advantages(
    projected_returns: Sequence[float],
    mode: str = "mean",
    normalize: bool = True,
) -> list[float]:
    """Reproduce CoMLRL v1.4.1 return baselines and optional normalization."""

    values = [_finite_float(value, "projected return") for value in projected_returns]
    if not values:
        return []
    advantage_mode = str(mode).strip().lower()
    if advantage_mode not in {"mean", "raw", "rloo", "max"}:
        raise ValueError("mode must be one of: mean, raw, rloo, max")

    mean_return = fmean(values)
    if advantage_mode == "mean":
        advantages = [value - mean_return for value in values]
    elif advantage_mode == "raw":
        advantages = list(values)
    elif advantage_mode == "max":
        max_return = max(values)
        advantages = [value - max_return for value in values]
    elif len(values) <= 1:
        advantages = [value - mean_return for value in values]
    else:
        group_size = len(values)
        total = sum(values)
        advantages = [(value * group_size - total) / (group_size - 1) for value in values]

    if normalize and len(advantages) > 1:
        mean_advantage = fmean(advantages)
        variance = fmean((advantage - mean_advantage) ** 2 for advantage in advantages)
        std = max(sqrt(variance), 1e-6)
        advantages = [(advantage - mean_advantage) / std for advantage in advantages]
    if any(not isfinite(advantage) for advantage in advantages):
        raise ValueError("advantages must be finite")
    return advantages


def apply_sequence_kl(projected_return: float, sequence_kl: float, coefficient: float) -> float:
    """Apply the v1.4.1 sequence-level KL penalty after return projection."""

    value = _finite_float(projected_return, "projected_return")
    kl_value = _finite_float(sequence_kl, "sequence_kl")
    coefficient_value = _finite_float(coefficient, "coefficient")
    if kl_value < 0.0:
        raise ValueError("sequence_kl must be non-negative")
    if coefficient_value < 0.0:
        raise ValueError("coefficient must be non-negative")
    result = value - coefficient_value * kl_value
    if not isfinite(result):
        raise ValueError("KL-adjusted return must be finite")
    return result


@dataclass
class CoMLRLReinforceCreditAssigner:
    name: str = "comlrl_reinforce"
    advantage_mode: str = "mean"
    normalize: bool = True
    sequence_kl_coefficient: float = 0.0
    sequence_kl_metadata_key: str = "sequence_kl"
    discount: float = 1.0

    def assign(
        self,
        trajectories: list[MultiAgentTrajectory],
        team: TeamSpec,
    ) -> list[TrainingSample]:
        if not trajectories:
            return []
        apply_sequence_kl(0.0, 0.0, self.sequence_kl_coefficient)
        trainable_agents = {agent.name for agent in team.trainable_agents()}
        rows: list[tuple[TrainingSample, float]] = []

        for trajectory in trajectories:
            validate_joint_trajectory(
                trajectory.joint_nodes,
                trajectory.joint_completions,
                trajectory.joint_actions,
                trajectory.joint_transitions,
                {agent.name for agent in team.agents},
            )
            action_returns = compute_joint_tree_returns(
                trajectory.joint_nodes,
                trajectory.joint_actions,
                discount=self.discount,
            )
            sampling_mode = _trajectory_sampling_mode(trajectory)
            projections = project_joint_returns_to_completions(
                trajectory.joint_completions,
                trajectory.joint_actions,
                action_returns,
                sampling_mode,
            )
            nodes_by_id = _index_unique(trajectory.joint_nodes, "tree_node_id")
            actions_by_id = _index_unique(trajectory.joint_actions, "joint_action_id")
            sampled_agents: set[str] = set()

            for completion in trajectory.joint_completions:
                if completion.agent_name not in trainable_agents:
                    continue
                try:
                    agent = team.agent(completion.agent_name)
                except KeyError as error:
                    raise ValueError(f"completion references unknown team agent {completion.agent_name}") from error
                projection = projections[completion.completion_id]
                component_actions = [actions_by_id[action_id] for action_id in projection.joint_action_ids]
                sequence_kl = _completion_sequence_kl(completion, self.sequence_kl_metadata_key)
                effective_return = apply_sequence_kl(
                    projection.projected_joint_return,
                    sequence_kl,
                    self.sequence_kl_coefficient,
                )
                node = nodes_by_id[completion.tree_node_id]
                rewards = [
                    _finite_float(action.shared_reward, f"shared_reward for {action.joint_action_id}")
                    for action in component_actions
                ]
                stop_reasons = list(
                    dict.fromkeys(action.stop_reason for action in component_actions if action.stop_reason)
                )
                sample = TrainingSample(
                    sample_id=f"{trajectory.episode_id}:{completion.completion_id}",
                    episode_id=trajectory.episode_id,
                    task_id=trajectory.task_id,
                    rollout_group=trajectory.rollout_group,
                    turn_id=node.depth,
                    agent_name=completion.agent_name,
                    role=agent.role,
                    policy_group=agent.policy_group,
                    prompt=str(completion.metadata.get("prompt", "")),
                    response=completion.text,
                    response_token_ids=list(completion.token_ids),
                    response_logprobs=list(completion.logprobs),
                    reward=effective_return,
                    root_id=node.root_id,
                    node_id=node.tree_node_id,
                    completion_id=completion.completion_id,
                    tree_node_id=completion.tree_node_id,
                    joint_action_ids=list(projection.joint_action_ids),
                    joint_transition_ids=list(projection.joint_transition_ids),
                    metadata={
                        "credit": self.name,
                        "observation": completion.metadata.get("observation", ""),
                        "joint_return_components": list(projection.joint_return_components),
                        "projected_joint_return": projection.projected_joint_return,
                        "joint_reward": fmean(rewards) if rewards else 0.0,
                        "joint_done": bool(component_actions) and all(action.done for action in component_actions),
                        "joint_truncated": bool(component_actions)
                        and all(action.truncated for action in component_actions),
                        "joint_stop_reason": ",".join(stop_reasons),
                        "joint_sampling_mode": sampling_mode,
                        "sequence_kl": sequence_kl,
                        "sequence_kl_coefficient": self.sequence_kl_coefficient,
                        "effective_projected_joint_return": effective_return,
                    },
                )
                rows.append((sample, effective_return))
                sampled_agents.add(completion.agent_name)

            missing_agents = trainable_agents - sampled_agents
            if missing_agents:
                raise ValueError(
                    f"trajectory {trajectory.episode_id} has no completion samples for trainable agents: "
                    f"{sorted(missing_agents)}"
                )

        grouped_rows: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for row_index, (sample, _return_value) in enumerate(rows):
            grouped_rows[(sample.episode_id, str(sample.tree_node_id), sample.agent_name)].append(row_index)
        for group, row_indices in grouped_rows.items():
            advantages = compute_group_advantages(
                [rows[row_index][1] for row_index in row_indices],
                mode=self.advantage_mode,
                normalize=self.normalize,
            )
            for row_index, advantage in zip(row_indices, advantages, strict=True):
                sample = rows[row_index][0]
                sample.advantage = advantage
                sample.metadata["advantage_group"] = {
                    "episode_id": group[0],
                    "tree_node_id": group[1],
                    "agent_name": group[2],
                }

        return [sample for sample, _return_value in rows]


def _trajectory_sampling_mode(trajectory: MultiAgentTrajectory) -> str:
    joint_tree = trajectory.metadata.get("joint_tree")
    nested_mode = joint_tree.get("joint_mode") if isinstance(joint_tree, Mapping) else None
    mode = trajectory.metadata.get(
        "joint_sampling_mode",
        trajectory.metadata.get("joint_mode", nested_mode or "aligned"),
    )
    return _canonical_joint_mode(str(mode))


def _completion_sequence_kl(completion: JointCompletion, metadata_key: str) -> float:
    if not isinstance(metadata_key, str) or not metadata_key:
        raise ValueError("sequence_kl_metadata_key must be a non-empty string")
    return _finite_float(completion.metadata.get(metadata_key, 0.0), metadata_key)


def _canonical_joint_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"align", "aligned"}:
        return "aligned"
    if normalized in {"cross", "crossed"}:
        return "cross"
    raise ValueError(f"unsupported joint sampling mode: {mode}")


def _finite_float(value: object, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _index_unique(items: Sequence[object], attribute: str) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for item in items:
        item_id = getattr(item, attribute)
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{attribute} must be a non-empty string")
        if item_id in indexed:
            raise ValueError(f"duplicate {attribute}: {item_id}")
        indexed[item_id] = item
    return indexed


def _reject_cycles(
    nodes_by_id: Mapping[str, JointTreeNode],
    actions_by_node: Mapping[str, Sequence[JointAction]],
) -> None:
    state: dict[str, int] = {}

    def visit(node_id: str) -> None:
        status = state.get(node_id, 0)
        if status == 1:
            raise ValueError(f"joint action graph contains a cycle at node {node_id}")
        if status == 2:
            return
        state[node_id] = 1
        for action in actions_by_node.get(node_id, ()):
            if action.child_tree_node_id is not None:
                visit(action.child_tree_node_id)
        state[node_id] = 2

    for node_id in nodes_by_id:
        visit(node_id)
