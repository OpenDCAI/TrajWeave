from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory, TrainingSample
from trajweave.credit.comas.scoring import parse_comas_score


@dataclass(frozen=True)
class CoMASInteractionRewards:
    solver: float
    evaluator: float
    scorer: float
    normalized_score: float | None
    score_valid: bool


def allocate_comas_interaction_rewards(score: int | None) -> CoMASInteractionRewards:
    """CoMAS 原始交互奖励真值表。"""

    if score not in {1, 2, 3}:
        return CoMASInteractionRewards(
            solver=0.0,
            evaluator=0.0,
            scorer=-1.0,
            normalized_score=None,
            score_valid=False,
        )
    normalized_score = (float(score) - 1.0) / 2.0
    return CoMASInteractionRewards(
        solver=normalized_score,
        evaluator=1.0 - normalized_score,
        scorer=0.0,
        normalized_score=normalized_score,
        score_valid=True,
    )


@dataclass
class CoMASInteractionCreditAssigner:
    name: str = "comas_interaction_reward"

    def allocate_trajectory(self, trajectory: MultiAgentTrajectory) -> None:
        grouped: dict[str, list[AgentTurn]] = defaultdict(list)
        for turn in trajectory.turns:
            interaction_id = turn.metadata.get("interaction_id")
            if interaction_id is None:
                raise KeyError("CoMAS turn is missing interaction_id.")
            grouped[str(interaction_id)].append(turn)

        for interaction_id, turns in grouped.items():
            by_role = {turn.role: turn for turn in turns}
            expected = {"solver", "evaluator", "scorer"}
            if set(by_role) != expected or len(turns) != 3:
                raise ValueError(
                    f"CoMAS interaction {interaction_id!r} must contain exactly one solver, evaluator and scorer."
                )
            parsed_score = parse_comas_score(by_role["scorer"].action_text)
            rewards = allocate_comas_interaction_rewards(parsed_score)
            role_rewards = {
                "solver": rewards.solver,
                "evaluator": rewards.evaluator,
                "scorer": rewards.scorer,
            }
            for role, turn in by_role.items():
                reward = role_rewards[role]
                turn.reward = reward
                turn.metadata.update(
                    {
                        "generated_score": parsed_score if parsed_score is not None else -1,
                        "score_valid": rewards.score_valid,
                        "normalized_score": (
                            rewards.normalized_score if rewards.normalized_score is not None else -1.0
                        ),
                        "intrinsic_reward": reward,
                        "reward_source": "comas_interaction_only",
                    }
                )

        trajectory.metadata["comas_credit_allocator"] = self.name
        trajectory.metadata["comas_training_uses_ground_truth"] = False

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        trainable = {agent.name for agent in team.trainable_agents()}
        samples: list[TrainingSample] = []
        for trajectory in trajectories:
            self.allocate_trajectory(trajectory)
            for turn in trajectory.trainable_turns(trainable):
                if turn.reward is None:
                    raise RuntimeError(f"CoMAS turn {turn.turn_id} did not receive an intrinsic reward.")
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
                        reward=float(turn.reward),
                        completion_id=turn.completion_id,
                        tree_node_id=turn.tree_node_id,
                        joint_action_ids=turn.joint_action_ids,
                        joint_transition_ids=turn.joint_transition_ids,
                        metadata={"credit": self.name, **turn.metadata},
                    )
                )
        return samples
