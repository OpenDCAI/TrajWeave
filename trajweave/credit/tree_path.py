from __future__ import annotations

from trajweave.core import SearchNode


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
