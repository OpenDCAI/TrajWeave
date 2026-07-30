from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTurn:
    episode_id: str
    task_id: str
    turn_id: int
    agent_name: str
    role: str
    policy_group: str
    observation: str
    prompt: str
    action_text: str
    action_token_ids: list[int] = field(default_factory=list)
    action_logprobs: list[float] = field(default_factory=list)
    anchor_observation: Any | None = None
    next_observation: Any | None = None
    step_reward: float | None = None
    reward: float | None = None
    advantage: float | None = None
    done: bool = False
    root_id: str | None = None
    node_id: str | None = None
    parent_node_id: str | None = None
    observation_group_id: str | None = None
    branch_index: int | None = None
    selected_for_expansion: bool = False
    local_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentTrajectory:
    episode_id: str
    task_id: str
    rollout_group: str
    team_name: str
    turns: list[AgentTurn] = field(default_factory=list)
    final_answer: str = ""
    global_reward: float | None = None
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_turn(self, turn: AgentTurn) -> None:
        self.turns.append(turn)

    def trainable_turns(self, trainable_agent_names: set[str]) -> list[AgentTurn]:
        return [turn for turn in self.turns if turn.agent_name in trainable_agent_names]


@dataclass
class TrainingSample:
    sample_id: str
    episode_id: str
    task_id: str
    rollout_group: str
    turn_id: int
    agent_name: str
    role: str
    policy_group: str
    prompt: str
    response: str
    response_token_ids: list[int]
    response_logprobs: list[float]
    reward: float
    advantage: float | None = None
    root_id: str | None = None
    node_id: str | None = None
    parent_node_id: str | None = None
    observation_group_id: str | None = None
    branch_index: int | None = None
    selected_for_expansion: bool = False
    local_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
