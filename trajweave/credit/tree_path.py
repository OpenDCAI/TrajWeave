from __future__ import annotations

from dataclasses import dataclass

from trajweave.core import SearchNode, TeamSpec, TrainingSample, TreeTrajectory
from trajweave.credit.tree_grouping import group_normalized_advantages


def parent_sibling_shaped_rewards(
    nodes: list[SearchNode],
    *,
    gamma: float = 0.3,
    sibling_mix: float = 0.5,
) -> list[float]:
    if not 0.0 <= sibling_mix <= 1.0:
        raise ValueError("sibling_mix must be in [0, 1].")
    by_id = {node.node_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("SearchNode.node_id values must be unique within a tree.")
    children: dict[int | None, list[SearchNode]] = {}
    for node in nodes:
        children.setdefault(node.parent_idx, []).append(node)

    raw = {node.node_id: float(node.reward or 0.0) for node in nodes}
    shaped = dict(raw)
    for siblings in children.values():
        for node in siblings:
            parent_reward = raw.get(node.parent_idx) if node.parent_idx is not None else None
            other_rewards = [raw[sibling.node_id] for sibling in siblings if sibling.node_id != node.node_id]
            sibling_mean = sum(other_rewards) / len(other_rewards) if other_rewards else None
            if parent_reward is not None and sibling_mean is not None:
                baseline = (1.0 - sibling_mix) * parent_reward + sibling_mix * sibling_mean
            elif parent_reward is not None:
                baseline = parent_reward
            elif sibling_mean is not None:
                baseline = sibling_mean
            else:
                continue
            shaped[node.node_id] = raw[node.node_id] + gamma * (raw[node.node_id] - baseline)
    return [shaped[node.node_id] for node in nodes]


def discounted_path_returns(
    nodes: list[SearchNode],
    rewards: list[float],
    *,
    discount: float = 0.3,
) -> list[float]:
    if len(nodes) != len(rewards):
        raise ValueError("nodes and rewards must have the same length.")
    if discount < 0.0:
        raise ValueError("discount must be non-negative.")
    reward_by_id = {node.node_id: float(reward) for node, reward in zip(nodes, rewards, strict=True)}
    node_by_id = {node.node_id: node for node in nodes}
    output: list[float] = []
    for node in nodes:
        path = node.path or _reconstruct_path(node, node_by_id)
        if not path or path[-1] != node.node_id:
            raise ValueError(f"SearchNode {node.node_id} has an invalid path: {path!r}.")
        total = 0.0
        for distance, node_id in enumerate(reversed(path)):
            if node_id not in reward_by_id:
                raise ValueError(f"SearchNode {node.node_id} path references missing node {node_id}.")
            total += (discount**distance) * reward_by_id[node_id]
        output.append(total)
    return output


@dataclass
class TreePathCreditAllocator:
    name: str = "marti_mars2_tree_path_grpo"
    parent_sibling_gamma: float = 0.3
    sibling_mix: float = 0.5
    path_discount: float = 0.3
    normalize_advantage: bool = True

    def assign_tree(self, trajectory: TreeTrajectory, team: TeamSpec) -> list[TrainingSample]:
        trainable = {agent.name for agent in team.trainable_agents()}
        nodes = list(trajectory.nodes)
        shaped = parent_sibling_shaped_rewards(
            nodes,
            gamma=self.parent_sibling_gamma,
            sibling_mix=self.sibling_mix,
        )
        path_returns = discounted_path_returns(nodes, shaped, discount=self.path_discount)
        trainable_indices = [index for index, node in enumerate(nodes) if node.agent_name in trainable]
        trainable_returns = [path_returns[index] for index in trainable_indices]
        groups = [list(range(len(trainable_indices)))]
        advantages = (
            group_normalized_advantages(trainable_returns, groups)
            if self.normalize_advantage
            else list(trainable_returns)
        )
        samples: list[TrainingSample] = []
        for index, advantage in zip(trainable_indices, advantages, strict=True):
            node = nodes[index]
            shaped_reward = shaped[index]
            path_return = path_returns[index]
            node.shaped_reward = shaped_reward
            node.advantage = advantage
            samples.append(
                TrainingSample(
                    sample_id=f"{node.tree_id}:{node.node_id}:{node.agent_name}",
                    episode_id=node.tree_id,
                    task_id=trajectory.task_id,
                    rollout_group=trajectory.rollout_group,
                    turn_id=node.turn_id if node.turn_id is not None else node.node_id,
                    agent_name=node.agent_name,
                    role=node.role,
                    policy_group=node.policy_group,
                    prompt=node.prompt,
                    response=node.action_text,
                    response_token_ids=node.action_token_ids,
                    response_logprobs=node.rollout_logprobs,
                    reward=path_return,
                    advantage=advantage,
                    metadata={
                        "credit": self.name,
                        "tree_id": node.tree_id,
                        "prompt_id": node.prompt_id,
                        "node_id": node.node_id,
                        "parent_idx": node.parent_idx,
                        "path": list(node.path),
                        "raw_reward": float(node.reward or 0.0),
                        "parent_sibling_reward": shaped_reward,
                        "path_return": path_return,
                        "rollout_logprob_available": bool(node.rollout_logprobs),
                    },
                )
            )
        return samples

    def assign(self, trajectories: list[TreeTrajectory], team: TeamSpec) -> list[TrainingSample]:
        return [sample for trajectory in trajectories for sample in self.assign_tree(trajectory, team)]


@dataclass
class TreeGroupCreditAllocator:
    """MARTI-MARS² fidelity credit: raw node reward + tree-group GRPO.

    The official training path verified so far uses node rewards and group
    normalization.  Parent/sibling shaping and discounted path returns live in
    :class:`TreePathCreditAllocator` and must be explicitly opted into.
    """

    name: str = "marti_mars2_fidelity_group_grpo"
    normalize_advantage: bool = True

    def assign_tree(self, trajectory: TreeTrajectory, team: TeamSpec) -> list[TrainingSample]:
        trainable = {agent.name for agent in team.trainable_agents()}
        nodes = list(trajectory.nodes)
        trainable_indices = [index for index, node in enumerate(nodes) if node.agent_name in trainable]
        raw_rewards = [float(nodes[index].reward or 0.0) for index in trainable_indices]
        groups = [list(range(len(trainable_indices)))]
        advantages = (
            group_normalized_advantages(raw_rewards, groups)
            if self.normalize_advantage
            else list(raw_rewards)
        )
        samples: list[TrainingSample] = []
        for index, advantage in zip(trainable_indices, advantages, strict=True):
            node = nodes[index]
            raw_reward = float(node.reward or 0.0)
            node.shaped_reward = raw_reward
            node.advantage = advantage
            samples.append(
                TrainingSample(
                    sample_id=f"{node.tree_id}:{node.node_id}:{node.agent_name}",
                    episode_id=node.tree_id,
                    task_id=trajectory.task_id,
                    rollout_group=trajectory.rollout_group,
                    turn_id=node.turn_id if node.turn_id is not None else node.node_id,
                    agent_name=node.agent_name,
                    role=node.role,
                    policy_group=node.policy_group,
                    prompt=node.prompt,
                    response=node.action_text,
                    response_token_ids=node.action_token_ids,
                    response_logprobs=node.rollout_logprobs,
                    reward=raw_reward,
                    advantage=advantage,
                    metadata={
                        "credit": self.name,
                        "tree_id": node.tree_id,
                        "prompt_id": node.prompt_id,
                        "node_id": node.node_id,
                        "parent_idx": node.parent_idx,
                        "path": list(node.path),
                        "raw_reward": raw_reward,
                        "parent_sibling_reward": raw_reward,
                        "path_return": raw_reward,
                        "rollout_logprob_available": bool(node.rollout_logprobs),
                    },
                )
            )
        return samples

    def assign(self, trajectories: list[TreeTrajectory], team: TeamSpec) -> list[TrainingSample]:
        return [sample for trajectory in trajectories for sample in self.assign_tree(trajectory, team)]


def _reconstruct_path(node: SearchNode, node_by_id: dict[int, SearchNode]) -> tuple[int, ...]:
    path = [node.node_id]
    seen = {node.node_id}
    parent_idx = node.parent_idx
    while parent_idx is not None:
        if parent_idx in seen:
            raise ValueError(f"Cycle detected while reconstructing path for SearchNode {node.node_id}.")
        parent = node_by_id.get(parent_idx)
        if parent is None:
            raise ValueError(f"SearchNode {node.node_id} references missing parent_idx {parent_idx}.")
        seen.add(parent_idx)
        path.append(parent_idx)
        parent_idx = parent.parent_idx
    return tuple(reversed(path))
