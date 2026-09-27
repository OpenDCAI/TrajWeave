from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from trajweave.envs.math import MathTask, SolverVerifierMathEnvironment


@dataclass(frozen=True)
class JointStepResult:
    team_reward: float
    next_observations_by_agent: dict[str, str]
    done: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class JointEnvironment(Protocol):
    def step_joint(
        self,
        task: Any,
        observations_by_agent: Mapping[str, str],
        actions_by_agent: Mapping[str, str],
        histories_by_agent: Mapping[str, Sequence[str]],
        turn_index: int,
    ) -> JointStepResult: ...


@dataclass
class JointMathEnvironment:
    """Small shared-reward adapter over TrajWeave's existing math evaluator."""

    evaluator: SolverVerifierMathEnvironment = field(default_factory=SolverVerifierMathEnvironment)
    evaluate_task_reward: bool = True

    def initial_observations(self, task: MathTask, agent_names: Sequence[str]) -> dict[str, str]:
        names = _validate_agent_names(agent_names)
        observation = self.evaluator.initial_observation(task)
        return {name: observation for name in names}

    def step_joint(
        self,
        task: MathTask,
        observations_by_agent: Mapping[str, str],
        actions_by_agent: Mapping[str, str],
        histories_by_agent: Mapping[str, Sequence[str]],
        turn_index: int,
    ) -> JointStepResult:
        names = _validate_joint_inputs(
            observations_by_agent,
            actions_by_agent,
            histories_by_agent,
            turn_index,
        )
        if self.evaluate_task_reward:
            rewards_and_success = {name: self.evaluator.evaluate(task, actions_by_agent[name]) for name in names}
            agent_rewards = {name: float(result[0]) for name, result in rewards_and_success.items()}
            agent_correct = {name: bool(result[1]) for name, result in rewards_and_success.items()}
        else:
            agent_rewards = dict.fromkeys(names, 0.0)
            agent_correct = dict.fromkeys(names, False)
        team_reward = sum(agent_rewards.values()) / len(agent_rewards)
        joint_actions = "\n".join(f"{name}: {actions_by_agent[name]}" for name in names)
        next_observations = {
            name: (
                f"Task:\n{task.question}\n\n"
                f"Joint responses from turn {turn_index}:\n{joint_actions}\n\n"
                f"Continue as {name} using the joint responses above."
            )
            for name in names
        }
        return JointStepResult(
            team_reward=float(team_reward),
            next_observations_by_agent=next_observations,
            done=all(agent_correct.values()),
            metadata={
                "turn_index": turn_index,
                "agent_rewards": agent_rewards,
                "agent_correct": agent_correct,
                "task_reward_evaluated": self.evaluate_task_reward,
            },
        )


def _validate_agent_names(agent_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(agent_names, (str, bytes)):
        raise TypeError("agent_names must be a sequence of names, not a string")
    names = tuple(agent_names)
    if not names:
        raise ValueError("agent_names must not be empty")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("agent_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("agent_names must be unique")
    return names


def _validate_joint_inputs(
    observations_by_agent: Mapping[str, str],
    actions_by_agent: Mapping[str, str],
    histories_by_agent: Mapping[str, Sequence[str]],
    turn_index: int,
) -> tuple[str, ...]:
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValueError("turn_index must be a non-negative integer")
    names = _validate_agent_names(tuple(observations_by_agent))
    expected = set(names)
    if set(actions_by_agent) != expected or set(histories_by_agent) != expected:
        raise ValueError("observations, actions, and histories must contain the same agents")
    if any(not isinstance(observations_by_agent[name], str) for name in names):
        raise TypeError("each observation must be a string")
    if any(not isinstance(actions_by_agent[name], str) for name in names):
        raise TypeError("each action must be a string")
    for name in names:
        history = histories_by_agent[name]
        if isinstance(history, (str, bytes)) or any(not isinstance(item, str) for item in history):
            raise TypeError("each agent history must be a sequence of strings")
    return names
