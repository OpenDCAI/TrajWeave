from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchNode:
    tree_id: str
    prompt_id: str
    node_id: int
    agent_name: str
    role: str
    policy_group: str
    prompt: str
    action_text: str
    parent_idx: int | None = None
    turn_id: int | None = None
    action_token_ids: list[int] = field(default_factory=list)
    rollout_logprobs: list[float] = field(default_factory=list)
    policy_logprobs: list[float] = field(default_factory=list)
    reward: float | None = None
    shaped_reward: float | None = None
    advantage: float | None = None
    path: tuple[int, ...] = ()
    is_terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tree_id:
            raise ValueError("SearchNode.tree_id must be non-empty.")
        if not self.prompt_id:
            raise ValueError("SearchNode.prompt_id must be non-empty.")
        if self.node_id < 0:
            raise ValueError("SearchNode.node_id must be non-negative.")
        if self.parent_idx is not None and self.parent_idx < 0:
            raise ValueError("SearchNode.parent_idx must be non-negative when present.")

    @property
    def effective_reward(self) -> float:
        if self.shaped_reward is not None:
            return float(self.shaped_reward)
        if self.reward is not None:
            return float(self.reward)
        return 0.0


@dataclass
class TreeTrajectory:
    tree_id: str
    prompt_id: str
    task_id: str
    rollout_group: str
    nodes: list[SearchNode] = field(default_factory=list)
    final_answer: str = ""
    global_reward: float | None = None
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tree_id:
            raise ValueError("TreeTrajectory.tree_id must be non-empty.")
        if not self.prompt_id:
            raise ValueError("TreeTrajectory.prompt_id must be non-empty.")
        for node in self.nodes:
            self._validate_node_identity(node)
        self.validate_links()

    def add_node(self, node: SearchNode) -> None:
        self._validate_node_identity(node)
        if any(existing.node_id == node.node_id for existing in self.nodes):
            raise ValueError(f"Duplicate SearchNode.node_id in tree {self.tree_id!r}: {node.node_id}.")
        existing_node_ids = {existing.node_id for existing in self.nodes}
        if node.parent_idx is not None and node.parent_idx not in existing_node_ids:
            raise ValueError(
                f"SearchNode {node.node_id} in tree {self.tree_id!r} references missing parent_idx "
                f"{node.parent_idx}."
            )
        if node.parent_idx == node.node_id:
            raise ValueError(f"SearchNode {node.node_id} in tree {self.tree_id!r} cannot parent itself.")
        self.nodes.append(node)
        self.validate_links()

    def trainable_nodes(self, trainable_agent_names: set[str]) -> list[SearchNode]:
        return [node for node in self.nodes if node.agent_name in trainable_agent_names]

    def validate_links(self) -> None:
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError(f"Duplicate SearchNode.node_id in tree {self.tree_id!r}.")
        for node in self.nodes:
            if node.parent_idx is None:
                continue
            if node.parent_idx not in node_ids:
                raise ValueError(
                    f"SearchNode {node.node_id} in tree {self.tree_id!r} references missing parent_idx "
                    f"{node.parent_idx}."
                )
            if node.parent_idx == node.node_id:
                raise ValueError(f"SearchNode {node.node_id} in tree {self.tree_id!r} cannot parent itself.")

    def _validate_node_identity(self, node: SearchNode) -> None:
        if node.tree_id != self.tree_id:
            raise ValueError(f"Node tree_id {node.tree_id!r} does not match trajectory tree_id {self.tree_id!r}.")
        if node.prompt_id != self.prompt_id:
            raise ValueError(
                f"Node prompt_id {node.prompt_id!r} does not match trajectory prompt_id {self.prompt_id!r}."
            )
