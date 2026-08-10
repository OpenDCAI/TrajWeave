from __future__ import annotations

from dataclasses import dataclass

from trajweave.core import TeamSpec, TrainingSample, TreeTrajectory
from trajweave.credit.tree_grouping import group_normalized_advantages
from trajweave.credit.tree_path import discounted_path_returns, parent_sibling_shaped_rewards


def _sample(node, trajectory: TreeTrajectory, *, reward: float, advantage: float, name: str, metadata: dict):
    return TrainingSample(
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
        reward=reward,
        advantage=advantage,
        rollout_policy_step=node.rollout_policy_step,
        rollout_global_step=node.rollout_global_step,
        policy_lag=node.policy_lag,
        metadata={
            "credit": name,
            "tree_id": node.tree_id,
            "prompt_id": node.prompt_id,
            "node_id": node.node_id,
            "parent_idx": node.parent_idx,
            "path": list(node.path),
            "rollout_logprob_available": bool(node.rollout_logprobs),
            **metadata,
        },
    )


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
        shaped = parent_sibling_shaped_rewards(nodes, gamma=self.parent_sibling_gamma, sibling_mix=self.sibling_mix)
        path_returns = discounted_path_returns(nodes, shaped, discount=self.path_discount)
        indices = [index for index, node in enumerate(nodes) if node.agent_name in trainable]
        values = [path_returns[index] for index in indices]
        advantages = group_normalized_advantages(values, [list(range(len(indices)))]) if self.normalize_advantage else values
        samples = []
        for index, advantage in zip(indices, advantages, strict=True):
            node = nodes[index]
            node.shaped_reward = shaped[index]
            node.advantage = advantage
            samples.append(
                _sample(
                    node,
                    trajectory,
                    reward=path_returns[index],
                    advantage=advantage,
                    name=self.name,
                    metadata={
                        "raw_reward": float(node.reward or 0.0),
                        "parent_sibling_reward": shaped[index],
                        "path_return": path_returns[index],
                    },
                )
            )
        return samples

    def assign(self, trajectories: list[TreeTrajectory], team: TeamSpec) -> list[TrainingSample]:
        return [sample for trajectory in trajectories for sample in self.assign_tree(trajectory, team)]


@dataclass
class TreeGroupCreditAllocator:
    """MARTI fidelity credit: raw verifier node reward plus tree-group GRPO."""

    name: str = "marti_mars2_fidelity_group_grpo"
    normalize_advantage: bool = True

    def assign_tree(self, trajectory: TreeTrajectory, team: TeamSpec) -> list[TrainingSample]:
        trainable = {agent.name for agent in team.trainable_agents()}
        nodes = list(trajectory.nodes)
        indices = [index for index, node in enumerate(nodes) if node.agent_name in trainable]
        rewards = [float(nodes[index].reward or 0.0) for index in indices]
        advantages = group_normalized_advantages(rewards, [list(range(len(indices)))]) if self.normalize_advantage else rewards
        samples = []
        for index, advantage in zip(indices, advantages, strict=True):
            node = nodes[index]
            reward = float(node.reward or 0.0)
            node.shaped_reward = reward
            node.advantage = advantage
            samples.append(
                _sample(
                    node,
                    trajectory,
                    reward=reward,
                    advantage=advantage,
                    name=self.name,
                    metadata={"raw_reward": reward, "parent_sibling_reward": reward, "path_return": reward},
                )
            )
        return samples

    def assign(self, trajectories: list[TreeTrajectory], team: TeamSpec) -> list[TrainingSample]:
        return [sample for trajectory in trajectories for sample in self.assign_tree(trajectory, team)]
