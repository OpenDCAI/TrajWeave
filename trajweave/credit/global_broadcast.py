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
            reward = float(trajectory.global_reward or 0.0)
            for turn in trajectory.trainable_turns(trainable):
                turn.reward = reward
                sample = TrainingSample(
                    sample_id=f"{trajectory.episode_id}:{turn.turn_id}:{turn.agent_name}",
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
                    metadata={"credit": self.name, **turn.metadata},
                )
                samples.append(sample)
        return samples
