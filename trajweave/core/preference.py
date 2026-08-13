from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any


@dataclass
class JointPreferencePair:
    preference_pair_id: str
    episode_id: str
    tree_node_id: str
    chosen_joint_action_id: str
    rejected_joint_action_id: str
    prompts_by_agent: dict[str, str]
    chosen_by_agent: dict[str, str]
    rejected_by_agent: dict[str, str]
    chosen_reward: float
    rejected_reward: float
    candidate_mean: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        identifiers = {
            "preference_pair_id": self.preference_pair_id,
            "episode_id": self.episode_id,
            "tree_node_id": self.tree_node_id,
            "chosen_joint_action_id": self.chosen_joint_action_id,
            "rejected_joint_action_id": self.rejected_joint_action_id,
        }
        if any(not value for value in identifiers.values()):
            raise ValueError("preference and trajectory identifiers must be non-empty")
        if self.chosen_joint_action_id == self.rejected_joint_action_id:
            raise ValueError("chosen and rejected joint actions must be different")
        prompt_agents = set(self.prompts_by_agent)
        if not prompt_agents:
            raise ValueError("a joint preference pair must contain at least one agent")
        if prompt_agents != set(self.chosen_by_agent) or prompt_agents != set(self.rejected_by_agent):
            raise ValueError("prompts, chosen completions, and rejected completions must have identical agent keys")
        rewards = (self.chosen_reward, self.rejected_reward, self.candidate_mean)
        if not all(isfinite(value) for value in rewards):
            raise ValueError("preference rewards must be finite")
        if self.chosen_reward <= self.rejected_reward:
            raise ValueError("chosen_reward must be strictly greater than rejected_reward")
