# Copyright 2025 Junwei Liao, Shanghai Jiao Tong University and Shanghai Innovation Institute.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory, TrainingSample
from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner

MARFTStepRewardFn = Callable[..., float]


@dataclass(frozen=True)
class MARFTStep:
    agent_name: str
    step_index: int
    token_start: int
    token_end: int
    output_text: str
    turn: AgentTurn


@dataclass
class MARFTSharedEnvironmentView:
    """Reward-callback view matching MARFT's ``messages``/``metadata`` contract."""

    messages: list[dict[str, str]]
    metadata: dict[str, Any]


def apply_marft_trajectory_credit(
    trajectory: MultiAgentTrajectory,
    *,
    strategy: str = "equal",
    discount: float = 1.0,
    gamma: float = 1.0,
    step_reward_fn: MARFTStepRewardFn | None = None,
    per_agent_reward_fns: Mapping[str, MARFTStepRewardFn] | None = None,
    environment: Any | None = None,
    data: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    """Project MARFT's token-sequence rewards onto TrajWeave turn rows.

    Upstream MARFT places sparse rewards on the last token of selected agent
    steps and lets GAE propagate them over one concatenated sequence. TrajWeave
    routes one row per trainable role, so this function records the sparse event
    as ``step_reward`` and its reverse discounted return as the row reward.
    """

    normalized_strategy = str(strategy).strip().lower()
    if normalized_strategy not in {"equal", "step_discount", "per_step"}:
        raise ValueError("MARFT credit strategy must be equal, step_discount, or per_step.")
    discount = _unit_interval(discount, field="credit_discount")
    gamma = _unit_interval(gamma, field="gamma")
    turns = list(trajectory.turns)
    if not turns:
        return ()
    team_reward = _finite_float(trajectory.global_reward or 0.0, field="team_reward")
    per_agent_reward_fns = dict(per_agent_reward_fns or {})
    reward_data = _reward_data(data)
    reward_environment = _reward_environment_view(
        trajectory,
        environment=environment,
        data=reward_data,
    )
    events: list[float] = []
    token_offset = 0
    for index, turn in enumerate(turns):
        token_start = token_offset
        token_end = token_start + len(turn.action_token_ids)
        token_offset = token_end
        step = MARFTStep(
            agent_name=turn.agent_name,
            step_index=index,
            token_start=token_start,
            token_end=token_end,
            output_text=turn.action_text,
            turn=turn,
        )
        if normalized_strategy == "equal":
            event = team_reward if index == len(turns) - 1 else 0.0
        elif normalized_strategy == "step_discount":
            event = team_reward * discount ** (len(turns) - 1 - index)
        else:
            reward_fn = per_agent_reward_fns.get(turn.agent_name, step_reward_fn)
            if reward_fn is None:
                raise ValueError(f"MARFT per_step credit has no reward function for agent {turn.agent_name!r}.")
            event = _finite_float(
                reward_fn(step=step, team_reward=team_reward, env=reward_environment, data=reward_data),
                field=f"per_step reward for {turn.agent_name}",
            )
        events.append(float(event))

    returns = [0.0] * len(events)
    running = 0.0
    for index in range(len(events) - 1, -1, -1):
        running = events[index] + gamma * running
        returns[index] = running

    for index, (turn, event, projected_return) in enumerate(zip(turns, events, returns, strict=True)):
        turn.step_reward = event
        turn.reward = projected_return
        turn.metadata.update(
            {
                "marft_credit_strategy": normalized_strategy,
                "marft_credit_discount": discount,
                "marft_return_gamma": gamma,
                "marft_step_reward": event,
                "marft_projected_return": projected_return,
                "joint_reward": event,
                "joint_done": index == len(turns) - 1,
                "joint_truncated": False,
                "joint_stop_reason": "",
            }
        )
    trajectory.metadata.update(
        {
            "marft_credit_strategy": normalized_strategy,
            "marft_credit_discount": discount,
            "marft_return_gamma": gamma,
            "marft_step_rewards": events,
            "marft_projected_returns": returns,
        }
    )
    return tuple(returns)


@dataclass
class MARFTCreditAssigner(GlobalBroadcastCreditAssigner):
    name: str = "marft_ctde"
    strategy: str = "equal"
    discount: float = 1.0
    gamma: float = 1.0
    step_reward_fn: MARFTStepRewardFn | None = None
    per_agent_reward_fns: Mapping[str, MARFTStepRewardFn] | None = None

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        for trajectory in trajectories:
            apply_marft_trajectory_credit(
                trajectory,
                strategy=self.strategy,
                discount=self.discount,
                gamma=self.gamma,
                step_reward_fn=self.step_reward_fn,
                per_agent_reward_fns=self.per_agent_reward_fns,
            )
        samples = super().assign(trajectories, team)
        for sample in samples:
            sample.metadata["credit"] = self.name
        return samples


def _unit_interval(value: Any, *, field: str) -> float:
    parsed = _finite_float(value, field=field)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"MARFT {field} must be in [0, 1].")
    return parsed


def _finite_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MARFT {field} must be a finite number.") from exc
    if not isfinite(parsed):
        raise ValueError(f"MARFT {field} must be a finite number.")
    return parsed


def _reward_data(data: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(data or {})
    if "answer" not in normalized:
        reward_model = normalized.get("reward_model")
        if isinstance(reward_model, Mapping) and reward_model.get("ground_truth") is not None:
            normalized["answer"] = reward_model["ground_truth"]
    return normalized


def _reward_environment_view(
    trajectory: MultiAgentTrajectory,
    *,
    environment: Any | None,
    data: dict[str, Any],
) -> Any:
    if hasattr(environment, "messages") and hasattr(environment, "metadata"):
        return environment

    raw_messages = data.get("messages", data.get("raw_prompt", []))
    messages: list[dict[str, str]] = []
    if isinstance(raw_messages, list | tuple):
        for message in raw_messages:
            if isinstance(message, Mapping) and message.get("content") is not None:
                messages.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": str(message["content"]),
                    }
                )
    if not messages and trajectory.turns:
        messages.append({"role": "user", "content": str(trajectory.turns[0].observation)})
    messages.extend(
        {
            "role": "assistant",
            "content": turn.action_text,
            "agent_name": turn.agent_name,
        }
        for turn in trajectory.turns
    )
    metadata = dict(data)
    metadata["trajectory_metadata"] = dict(trajectory.metadata)
    return MARFTSharedEnvironmentView(messages=messages, metadata=metadata)
