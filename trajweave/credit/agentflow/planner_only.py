from __future__ import annotations

from dataclasses import dataclass

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample


@dataclass
class FlowGRPOPlannerOnlyCreditAssigner:
    name: str = "agentflow_planner_only_grpo"
    planner_name: str = "planner"
    planner_stage: str = "planner_next_step"

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        trainable = {agent.name for agent in team.trainable_agents()}
        samples: list[TrainingSample] = []
        for trajectory in trajectories:
            reward = float(trajectory.global_reward or 0.0)
            for turn in trajectory.turns:
                if turn.agent_name not in trainable:
                    continue
                if turn.agent_name != self.planner_name:
                    continue
                if turn.metadata.get("agentflow_stage") != self.planner_stage:
                    continue
                turn.reward = reward
                samples.append(
                    TrainingSample(
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
                        completion_id=turn.completion_id,
                        tree_node_id=turn.tree_node_id,
                        joint_action_ids=turn.joint_action_ids,
                        joint_transition_ids=turn.joint_transition_ids,
                        metadata={
                            "credit": self.name,
                            "reward_scope": "final_outcome",
                            "assign_to": self.planner_stage,
                            **turn.metadata,
                        },
                    )
                )
        return samples
