from __future__ import annotations

from dataclasses import dataclass

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample


@dataclass
class GlobalBroadcastCreditAssigner:
    name: str = "global_broadcast"

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        trainable = {agent.name for agent in team.trainable_agents()}
        samples: list[TrainingSample] = []
        for trajectory in trajectories:
            global_reward = float(trajectory.global_reward or 0.0)
            for turn in trajectory.trainable_turns(trainable):
                # Respect a reward already assigned upstream (e.g. AT-GRPO's mixed
                # global+local verifier reward) instead of unconditionally overwriting it
                # with the trajectory-level global reward.
                if turn.reward is None:
                    turn.reward = global_reward
                reward = turn.reward
                sample = TrainingSample(
                    sample_id=turn.node_id or f"{trajectory.episode_id}:{turn.turn_id}:{turn.agent_name}",
                    episode_id=trajectory.episode_id,
                    task_id=trajectory.task_id,
                    rollout_group=trajectory.rollout_group,
                    turn_id=turn.turn_id,
                    agent_name=turn.agent_name,
                    role=turn.role,
                    policy_group=turn.policy_group,
                    prompt=turn.prompt,
                    response=turn.action_text,
                    response_token_ids=turn.action_token_ids,
                    response_logprobs=turn.action_logprobs,
                    reward=reward,
                    root_id=turn.root_id,
                    node_id=turn.node_id,
                    parent_node_id=turn.parent_node_id,
                    observation_group_id=turn.observation_group_id,
                    branch_index=turn.branch_index,
                    selected_for_expansion=turn.selected_for_expansion,
                    local_score=turn.local_score,
                    completion_id=turn.completion_id,
                    tree_node_id=turn.tree_node_id,
                    joint_action_ids=turn.joint_action_ids,
                    joint_transition_ids=turn.joint_transition_ids,
                    metadata={
                        "credit": self.name,
                        "root_id": turn.root_id,
                        "node_id": turn.node_id,
                        "parent_node_id": turn.parent_node_id,
                        "observation_group_id": turn.observation_group_id,
                        "branch_index": turn.branch_index,
                        "selected_for_expansion": turn.selected_for_expansion,
                        "local_score": turn.local_score,
                        **turn.metadata,
                    },
                )
                samples.append(sample)
        return samples
