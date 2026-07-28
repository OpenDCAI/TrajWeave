from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner


def apply_mixed_reward(
    trajectory: MultiAgentTrajectory,
    team: TeamSpec,
    *,
    alpha: float = 1.0,
    verifier_local_reward_scale: float = 1.0,
    verifier_name: str = "verifier",
) -> None:
    """Give the verifier agent a reward that mixes the shared global outcome with a
    local, agent-specific signal: whether the verifier's own approval judgment agreed
    with the ground-truth correctness of the episode.

    Mirrors PettingLLMs' AT-GRPO mixed reward (``final_reward = alpha * global_reward +
    local_reward``). Solver turns are left untouched (``turn.reward`` stays ``None``) so
    :class:`GlobalBroadcastCreditAssigner` falls back to the plain global reward for
    them, since the solver has no independent local signal beyond the shared outcome.

    The verifier's judgment (``model_approved``/``approved``) is already computed and
    stored in turn metadata by ``SolverVerifierOrchestra`` (see
    ``trajweave/orchestration/solver_verifier/protocol.py``) but was previously
    discarded by the credit assigner -- this function is what actually consumes it.

    Mutates ``turn.reward`` in place on the trajectory's verifier turns. Called by both
    the local smoke path (:class:`ATGRPOCreditAssigner`) and the VERL training path
    (``trajweave/backends/verl/workflow_runtime.py``) so the reward formula cannot
    silently diverge between the two.
    """

    trainable = {agent.name for agent in team.trainable_agents()}
    actual_correct = bool(trajectory.success) if trajectory.success is not None else float(
        trajectory.global_reward or 0.0
    ) > 0
    global_reward = float(trajectory.global_reward or 0.0)
    for turn in trajectory.trainable_turns(trainable):
        if turn.agent_name != verifier_name:
            continue
        model_approved = bool(turn.metadata.get("model_approved", turn.metadata.get("approved", False)))
        judged_correctly = model_approved == actual_correct
        local_reward = verifier_local_reward_scale if judged_correctly else -verifier_local_reward_scale
        turn.reward = alpha * global_reward + local_reward
        turn.metadata["mixed_reward"] = {
            "alpha": alpha,
            "global_reward": global_reward,
            "local_reward": local_reward,
            "model_approved": model_approved,
            "actual_correct": actual_correct,
        }


@dataclass
class ATGRPOCreditAssigner(GlobalBroadcastCreditAssigner):
    """Agent- and Turn-wise GRPO credit assignment.

    Mirrors PettingLLMs' AT-GRPO: rather than normalizing rewards only within
    an agent's role across rollouts (as :class:`DoctorMASCreditAssigner`
    does), samples are grouped jointly by ``(rollout_group, turn_id,
    agent_name)`` so the baseline is computed within trajectories that share
    both the same turn index and the same agent role.
    """

    name: str = "atgrpo_agent_turn_wise_grpo"
    epsilon: float = 1e-6
    normalize_by_std: bool = True
    mixed_reward_enabled: bool = False
    alpha: float = 1.0
    verifier_local_reward_scale: float = 1.0
    verifier_name: str = "verifier"

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        if self.mixed_reward_enabled:
            for trajectory in trajectories:
                apply_mixed_reward(
                    trajectory,
                    team,
                    alpha=self.alpha,
                    verifier_local_reward_scale=self.verifier_local_reward_scale,
                    verifier_name=self.verifier_name,
                )
        samples = super().assign(trajectories, team)
        groups: dict[str, list[TrainingSample]] = defaultdict(list)
        for sample in samples:
            groups[f"{sample.rollout_group}:{sample.turn_id}:{sample.agent_name}"].append(sample)

        for group_samples in groups.values():
            rewards = [sample.reward for sample in group_samples]
            if len(rewards) > 1:
                mean = sum(rewards) / len(rewards)
                variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
                std = sqrt(variance)
            else:
                # Singleton groups have no sibling to compare against. Match VERL's native
                # GRPO convention (verl/trainer/ppo/core_algos.py: id2mean=0, id2std=1 for
                # groups of size 1) instead of self-centering (which would silently zero out
                # the advantage). This keeps the local smoke path consistent with
                # ATGRPOHooks.compute_grpo_outcome_advantage used on the VERL training path.
                mean = 0.0
                std = 1.0
            if std < self.epsilon:
                std = 1.0
            for sample in group_samples:
                advantage = sample.reward - mean
                if self.normalize_by_std:
                    advantage = advantage / (std + self.epsilon)
                sample.advantage = advantage
                sample.metadata["advantage_group"] = f"{sample.rollout_group}:{sample.turn_id}:{sample.agent_name}"
                sample.metadata["reward_mean"] = mean
                sample.metadata["reward_std"] = std
        return samples
