from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class JointTreeNode:
    tree_node_id: str
    root_id: str
    depth: int
    parent_joint_action_id: str | None = None
    truncated: bool = False
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.tree_node_id or not self.root_id:
            raise ValueError("tree_node_id and root_id must be non-empty")
        if self.depth < 0:
            raise ValueError("depth must be non-negative")
        if self.depth == 0:
            if self.tree_node_id != self.root_id:
                raise ValueError("a root node must reference itself as root_id")
            if self.parent_joint_action_id is not None:
                raise ValueError("a root node cannot have a parent joint action")
        elif not self.parent_joint_action_id:
            raise ValueError("a non-root node must have a parent joint action")
        if self.truncated != bool(self.stop_reason):
            raise ValueError("a truncated node must have exactly one stop_reason")


@dataclass
class JointCompletion:
    completion_id: str
    tree_node_id: str
    agent_name: str
    candidate_index: int
    text: str
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.completion_id or not self.tree_node_id or not self.agent_name:
            raise ValueError("completion_id, tree_node_id, and agent_name must be non-empty")
        if self.candidate_index < 0:
            raise ValueError("candidate_index must be non-negative")
        if self.logprobs and len(self.logprobs) != len(self.token_ids):
            raise ValueError("logprobs must align with token_ids when provided")


@dataclass
class JointAction:
    joint_action_id: str
    tree_node_id: str
    completion_ids: dict[str, str]
    candidate_indices: dict[str, int]
    joint_transition_id: str
    shared_reward: float | None = None
    shared_return: float | None = None
    child_tree_node_id: str | None = None
    done: bool = False
    truncated: bool = False
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.joint_action_id or not self.tree_node_id or not self.joint_transition_id:
            raise ValueError("joint_action_id, tree_node_id, and joint_transition_id must be non-empty")
        if not self.completion_ids:
            raise ValueError("a joint action must contain at least one agent completion")
        if set(self.completion_ids) != set(self.candidate_indices):
            raise ValueError("completion_ids and candidate_indices must have identical agent keys")
        if any(not completion_id for completion_id in self.completion_ids.values()):
            raise ValueError("completion_ids must be non-empty")
        if any(candidate_index < 0 for candidate_index in self.candidate_indices.values()):
            raise ValueError("candidate_indices must be non-negative")
        if self.done and self.truncated:
            raise ValueError("a joint action cannot be both done and truncated")
        if self.truncated != bool(self.stop_reason):
            raise ValueError("a truncated joint action must have exactly one stop_reason")
        if (self.done or self.truncated) and self.child_tree_node_id is not None:
            raise ValueError("a done or truncated joint action cannot have a child node")
        if not self.done and not self.truncated and self.child_tree_node_id is None:
            raise ValueError("a non-terminal joint action must have a child node")


@dataclass
class JointTransition:
    joint_transition_id: str
    joint_action_id: str
    source_tree_node_id: str
    source_turn: int
    target_tree_node_id: str | None = None
    target_turn: int | None = None
    done: bool = False
    truncated: bool = False
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.joint_transition_id or not self.joint_action_id or not self.source_tree_node_id:
            raise ValueError("transition and source IDs must be non-empty")
        if self.source_turn < 0:
            raise ValueError("source_turn must be non-negative")
        if self.target_turn is not None and self.target_turn < self.source_turn:
            raise ValueError("target_turn cannot precede source_turn")
        if self.done and self.truncated:
            raise ValueError("a joint transition cannot be both done and truncated")
        if self.truncated != bool(self.stop_reason):
            raise ValueError("a truncated joint transition must have exactly one stop_reason")
        if (self.done or self.truncated) and self.target_tree_node_id is not None:
            raise ValueError("a done or truncated joint transition cannot have a target node")
        if not self.done and not self.truncated and self.target_tree_node_id is None:
            raise ValueError("a non-terminal joint transition must have a target node")


def _index_unique(items: Iterable[Any], attribute: str) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for item in items:
        item_id = getattr(item, attribute)
        if item_id in indexed:
            raise ValueError(f"duplicate {attribute}: {item_id}")
        indexed[item_id] = item
    return indexed


def validate_joint_trajectory(
    nodes: list[JointTreeNode],
    completions: list[JointCompletion],
    actions: list[JointAction],
    transitions: list[JointTransition],
    agent_names: set[str] | None = None,
) -> None:
    for item in [*nodes, *completions, *actions, *transitions]:
        item.validate()

    nodes_by_id = _index_unique(nodes, "tree_node_id")
    completions_by_id = _index_unique(completions, "completion_id")
    actions_by_id = _index_unique(actions, "joint_action_id")
    transitions_by_id = _index_unique(transitions, "joint_transition_id")

    for node in nodes:
        root = nodes_by_id.get(node.root_id)
        if root is None or root.depth != 0 or root.root_id != root.tree_node_id:
            raise ValueError(f"node {node.tree_node_id} references an invalid root {node.root_id}")

    candidate_keys: set[tuple[str, str, int]] = set()
    for completion in completions:
        if completion.tree_node_id not in nodes_by_id:
            raise ValueError(f"completion {completion.completion_id} references unknown node {completion.tree_node_id}")
        candidate_key = (completion.tree_node_id, completion.agent_name, completion.candidate_index)
        if candidate_key in candidate_keys:
            raise ValueError(
                "candidate_index must be unique for each tree node and agent: "
                f"{completion.tree_node_id}/{completion.agent_name}/{completion.candidate_index}"
            )
        candidate_keys.add(candidate_key)

    expected_agents = set(agent_names) if agent_names is not None else {item.agent_name for item in completions}
    if actions and not expected_agents:
        raise ValueError("agent_names cannot be empty when joint actions are present")

    for action in actions:
        source_node = nodes_by_id.get(action.tree_node_id)
        if source_node is None:
            raise ValueError(f"joint action {action.joint_action_id} references unknown source node")
        if set(action.completion_ids) != expected_agents:
            raise ValueError(
                f"joint action {action.joint_action_id} has agents {set(action.completion_ids)}, "
                f"expected {expected_agents}"
            )
        for agent_name, completion_id in action.completion_ids.items():
            completion = completions_by_id.get(completion_id)
            if completion is None:
                raise ValueError(f"joint action {action.joint_action_id} references unknown completion {completion_id}")
            if completion.tree_node_id != action.tree_node_id or completion.agent_name != agent_name:
                raise ValueError(
                    f"completion {completion_id} does not belong to agent {agent_name} at node {action.tree_node_id}"
                )
            if completion.candidate_index != action.candidate_indices[agent_name]:
                raise ValueError(f"candidate index does not match completion {completion_id}")

        if action.child_tree_node_id is not None:
            child = nodes_by_id.get(action.child_tree_node_id)
            if child is None:
                raise ValueError(f"joint action {action.joint_action_id} references unknown child node")
            if child.parent_joint_action_id != action.joint_action_id:
                raise ValueError(f"child node {child.tree_node_id} does not point back to its parent action")
            if child.depth != source_node.depth + 1 or child.root_id != source_node.root_id:
                raise ValueError(f"child node {child.tree_node_id} has inconsistent depth or root")

    for node in nodes:
        if node.parent_joint_action_id is None:
            continue
        parent_action = actions_by_id.get(node.parent_joint_action_id)
        if parent_action is None:
            raise ValueError(f"node {node.tree_node_id} references unknown parent action")
        if parent_action.child_tree_node_id != node.tree_node_id:
            raise ValueError(f"parent action {parent_action.joint_action_id} does not point to child node")

    transitions_by_action: dict[str, JointTransition] = {}
    for transition in transitions:
        action = actions_by_id.get(transition.joint_action_id)
        if action is None:
            raise ValueError(f"transition {transition.joint_transition_id} references unknown joint action")
        if transition.joint_action_id in transitions_by_action:
            raise ValueError(f"joint action {transition.joint_action_id} has multiple transitions")
        transitions_by_action[transition.joint_action_id] = transition
        if action.joint_transition_id != transition.joint_transition_id:
            raise ValueError(f"joint action {action.joint_action_id} does not point back to its transition")
        source_node = nodes_by_id[action.tree_node_id]
        if transition.source_tree_node_id != action.tree_node_id or transition.source_turn != source_node.depth:
            raise ValueError(f"transition {transition.joint_transition_id} has an inconsistent source node or turn")
        if transition.target_tree_node_id != action.child_tree_node_id:
            raise ValueError(f"transition {transition.joint_transition_id} has an inconsistent target node")
        if transition.target_tree_node_id is not None:
            target_node = nodes_by_id.get(transition.target_tree_node_id)
            if target_node is None:
                raise ValueError(f"transition {transition.joint_transition_id} references unknown target node")
            if transition.target_turn != target_node.depth:
                raise ValueError(f"transition {transition.joint_transition_id} has an inconsistent target turn")
        elif transition.target_turn is not None:
            raise ValueError(f"transition {transition.joint_transition_id} has a target turn without a target node")
        if transition.done != action.done or transition.truncated != action.truncated:
            raise ValueError(f"transition {transition.joint_transition_id} has inconsistent terminal flags")
        if transition.stop_reason != action.stop_reason:
            raise ValueError(f"transition {transition.joint_transition_id} has an inconsistent stop reason")

    if set(transitions_by_action) != set(actions_by_id):
        missing = sorted(set(actions_by_id) - set(transitions_by_action))
        raise ValueError(f"every joint action must have exactly one transition; missing={missing}")
    for action in actions:
        if action.joint_transition_id not in transitions_by_id:
            raise ValueError(f"joint action {action.joint_action_id} references unknown transition")
