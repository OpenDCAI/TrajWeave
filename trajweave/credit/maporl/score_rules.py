from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample
from trajweave.credit.global_broadcast import GlobalBroadcastCreditAssigner


def shape_maporl_reward(
    global_reward: float,
    *,
    agent_answer: object | None,
    consensus_answer: object | None,
    consensus_reached: bool,
    correct_turn_bonus: float = 0.25,
    consensus_bonus: float = 0.25,
) -> tuple[float, dict[str, float | str]]:
    shaped_reward = float(global_reward)
    task_success = float(global_reward) > 0
    if task_success and _same_answer(agent_answer, consensus_answer):
        shaped_reward += correct_turn_bonus
    if task_success and consensus_reached:
        shaped_reward += consensus_bonus
    return shaped_reward, {
        "score_rule": "final_task_success",
        "bonus_rule": "correct_turn_and_consensus",
        "raw_global_reward": float(global_reward),
        "shaped_reward": shaped_reward,
    }


@dataclass
class MAPoRLScoreBonusCreditAssigner(GlobalBroadcastCreditAssigner):
    name: str = "maporl_score_bonus"
    correct_turn_bonus: float = 0.25
    consensus_bonus: float = 0.25
    epsilon: float = 1e-6
    normalize_by_agent: bool = True
    baseline_scope: str = "policy_group"

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        samples = super().assign(trajectories, team)
        trajectory_by_id = {trajectory.episode_id: trajectory for trajectory in trajectories}
        for sample in samples:
            trajectory = trajectory_by_id[sample.episode_id]
            shaped_reward, reward_metadata = shape_maporl_reward(
                sample.reward,
                agent_answer=sample.metadata.get("agent_answer"),
                consensus_answer=trajectory.metadata.get("consensus_answer"),
                consensus_reached=bool(sample.metadata.get("consensus_reached")),
                correct_turn_bonus=self.correct_turn_bonus,
                consensus_bonus=self.consensus_bonus,
            )
            sample.reward = shaped_reward
            sample.metadata["credit"] = self.name
            sample.metadata.update(reward_metadata)

        if self.normalize_by_agent:
            self._normalize_advantage(samples)
        return samples

    def _normalize_advantage(self, samples: list[TrainingSample]) -> None:
        groups: dict[str, list[TrainingSample]] = defaultdict(list)
        for sample in samples:
            if self.baseline_scope == "agent":
                group_key = f"{sample.rollout_group}:{sample.agent_name}"
            elif self.baseline_scope == "policy_group":
                group_key = f"{sample.rollout_group}:{sample.policy_group}"
            else:
                raise ValueError(f"Unsupported MAPoRL baseline_scope: {self.baseline_scope}")
            groups[group_key].append(sample)

        for group_key, group_samples in groups.items():
            rewards = [sample.reward for sample in group_samples]
            mean = sum(rewards) / len(rewards)
            if len(rewards) > 1:
                variance = sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
                std = sqrt(variance)
            else:
                std = 1.0
            if std < self.epsilon:
                std = 1.0
            for sample in group_samples:
                sample.advantage = (sample.reward - mean) / (std + self.epsilon)
                sample.metadata["advantage_group"] = group_key
                sample.metadata["reward_mean"] = mean
                sample.metadata["reward_std"] = std


def _same_answer(left: object | None, right: object | None) -> bool:
    if left is None or right is None:
        return False
    return str(left).strip() == str(right).strip()


_MAPORL_SCORE_RULE_DEFAULTS: dict[str, object] = {
    "rule_horizon": "discounted_sum",
    "rule_agent_share": "all",
    "rule_discount": 0.3,
    "alpha": (0.0, 0.0, 0.0, 0.0),
    "correct_threshold": 0.7,
    "wrong_threshold": 0.3,
    "bonus_correct_threshold": 0.5,
    "bonus_wrong_threshold": 0.5,
    "include_bonus": True,
}
_MISSING = object()


@dataclass(frozen=True)
class MAPoRLTurnCredit:
    score: float
    bonus: float
    reward: float


def resolve_maporl_score_rule_config(config: Any) -> dict[str, object]:
    """读取 VERL algorithm/trajweave 配置并规范化 MAPoRL credit 参数。"""

    values = dict(_MAPORL_SCORE_RULE_DEFAULTS)
    candidates = (
        _config_path(config, "agent", "orchestra", "maporl"),
        _config_path(config, "maporl"),
        _config_path(config, "credit"),
        _config_path(config, "trajweave", "credit"),
        _config_path(config, "trajweave", "maporl"),
        _config_path(config, "trajweave", "maporl_credit"),
        _config_path(config, "trajweave"),
        _config_path(config, "algorithm", "maporl"),
        _config_path(config, "algorithm", "maporl_credit"),
        _config_path(config, "algorithm"),
        config,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        for key in values:
            value = _config_get(candidate, key, _MISSING)
            if value is not _MISSING:
                values[key] = value

    alpha = values["alpha"]
    if isinstance(alpha, str):
        raise TypeError("MAPoRL alpha must be a sequence of numbers, not a string.")
    try:
        normalized_alpha = tuple(float(value) for value in alpha)  # type: ignore[union-attr]
    except TypeError as exc:
        raise TypeError("MAPoRL alpha must be a sequence of numbers.") from exc

    return {
        "rule_horizon": str(values["rule_horizon"]),
        "rule_agent_share": str(values["rule_agent_share"]),
        "rule_discount": float(values["rule_discount"]),
        "alpha": normalized_alpha,
        "correct_threshold": float(values["correct_threshold"]),
        "wrong_threshold": float(values["wrong_threshold"]),
        "bonus_correct_threshold": float(values["bonus_correct_threshold"]),
        "bonus_wrong_threshold": float(values["bonus_wrong_threshold"]),
        "include_bonus": _as_bool(values["include_bonus"]),
    }


@dataclass
class MAPoRLPPOScoreRuleCreditAssigner:
    """MAPoRL-style per-turn credit allocation.

    This mirrors the native MAPoRL trainer's split between ``score_rule`` and
    ``bonus_rule``: each agent turn receives a verifier/rule score aggregated
    over future turns, plus optional collaboration incentives based on answer
    improvement or deterioration across rounds.
    """

    name: str = "maporl_ppo_score_rule"
    rule_horizon: str = "discounted_sum"
    rule_agent_share: str = "all"
    rule_discount: float = 0.3
    alpha: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    correct_threshold: float = 0.7
    wrong_threshold: float = 0.3
    bonus_correct_threshold: float = 0.5
    bonus_wrong_threshold: float = 0.5
    include_bonus: bool = True
    metadata_defaults: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Any) -> MAPoRLPPOScoreRuleCreditAssigner:
        return cls(**resolve_maporl_score_rule_config(config))

    def assign(self, trajectories: list[MultiAgentTrajectory], team: TeamSpec) -> list[TrainingSample]:
        trainable = {agent.name for agent in team.trainable_agents()}
        samples: list[TrainingSample] = []
        agent_order = {agent.name: idx for idx, agent in enumerate(team.agents)}

        for trajectory in trajectories:
            turns = [turn for turn in trajectory.turns if turn.agent_name in trainable]
            if not turns:
                continue
            total_rounds = _trajectory_round_count(trajectory, turns)
            total_agents = len(team.agents)
            finished_round = int(trajectory.metadata.get("finished_round", -1))
            round_ids = [int(turn.metadata.get("round_id", turn.turn_id)) for turn in turns]
            agent_indices = [
                int(turn.metadata.get("agent_index", agent_order.get(turn.agent_name, 0))) for turn in turns
            ]
            credits = self.score_turns(
                round_ids=round_ids,
                agent_indices=agent_indices,
                raw_scores=[
                    float(turn.metadata.get("raw_score", float(trajectory.global_reward or 0.0))) for turn in turns
                ],
                correctnesses=[
                    float(turn.metadata.get("correctness", float(trajectory.success or 0.0))) for turn in turns
                ],
                finished_round=finished_round,
                total_rounds=total_rounds,
                total_agents=total_agents,
            )

            for turn, credit in zip(turns, credits, strict=True):
                turn.reward = credit.reward
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
                    reward=credit.reward,
                    completion_id=turn.completion_id,
                    tree_node_id=turn.tree_node_id,
                    joint_action_ids=turn.joint_action_ids,
                    joint_transition_ids=turn.joint_transition_ids,
                    metadata={
                        **self.metadata_defaults,
                        **turn.metadata,
                        "credit": self.name,
                        "score_rule": self.rule_horizon,
                        "rule_agent_share": self.rule_agent_share,
                        "rule_discount": self.rule_discount,
                        "maporl_score": credit.score,
                        "maporl_bonus": credit.bonus,
                        "raw_global_reward": float(trajectory.global_reward or 0.0),
                        "finished_round": finished_round,
                    },
                )
                samples.append(sample)
        return samples

    def score_turns(
        self,
        *,
        round_ids: Sequence[int],
        agent_indices: Sequence[int],
        raw_scores: Sequence[float],
        correctnesses: Sequence[float],
        finished_round: int,
        total_rounds: int | None = None,
        total_agents: int | None = None,
    ) -> list[MAPoRLTurnCredit]:
        """按输入顺序计算一条轨迹中每个 agent turn 的 MAPoRL credit。"""

        lengths = {len(round_ids), len(agent_indices), len(raw_scores), len(correctnesses)}
        if len(lengths) != 1:
            raise ValueError("MAPoRL turn fields must have the same number of rows.")
        if len(round_ids) == 0:
            return []

        normalized_round_ids = [int(value) for value in round_ids]
        normalized_agent_indices = [int(value) for value in agent_indices]
        normalized_raw_scores = [float(value) for value in raw_scores]
        normalized_correctnesses = [float(value) for value in correctnesses]
        inferred_rounds = max(normalized_round_ids) + 1
        inferred_agents = max(normalized_agent_indices) + 1
        total_rounds = inferred_rounds if total_rounds is None else int(total_rounds)
        total_agents = inferred_agents if total_agents is None else int(total_agents)
        if total_rounds <= 0 or total_agents <= 0:
            raise ValueError("MAPoRL total_rounds and total_agents must be positive.")

        raw_score_matrix = dict(
            zip(
                zip(normalized_round_ids, normalized_agent_indices, strict=True),
                normalized_raw_scores,
                strict=True,
            )
        )
        correctness_matrix = dict(
            zip(
                zip(normalized_round_ids, normalized_agent_indices, strict=True),
                normalized_correctnesses,
                strict=True,
            )
        )
        credits: list[MAPoRLTurnCredit] = []
        for round_id, agent_index in zip(normalized_round_ids, normalized_agent_indices, strict=True):
            score = _score_rule(
                raw_score_matrix,
                total_rounds=total_rounds,
                total_agents=total_agents,
                finished_round=int(finished_round),
                round_id=round_id,
                agent_index=agent_index,
                rule_horizon=self.rule_horizon,
                rule_agent_share=self.rule_agent_share,
                rule_discount=self.rule_discount,
                correct_threshold=self.correct_threshold,
                wrong_threshold=self.wrong_threshold,
            )
            bonus = (
                _bonus_rule(
                    correctness_matrix,
                    total_rounds=total_rounds,
                    total_agents=total_agents,
                    finished_round=int(finished_round),
                    round_id=round_id,
                    agent_index=agent_index,
                    alpha=self._alpha4(),
                    correct_threshold=self.bonus_correct_threshold,
                    wrong_threshold=self.bonus_wrong_threshold,
                )
                if self.include_bonus
                else 0.0
            )
            credits.append(MAPoRLTurnCredit(score=score, bonus=bonus, reward=score + bonus))
        return credits

    def _alpha4(self) -> tuple[float, float, float, float]:
        values = tuple(float(value) for value in self.alpha)
        if len(values) >= 4:
            return values[:4]
        return values + (0.0,) * (4 - len(values))


def _trajectory_round_count(trajectory: MultiAgentTrajectory, turns: list) -> int:
    configured = trajectory.metadata.get("round_num") or trajectory.metadata.get("max_rounds")
    if configured is not None:
        return int(configured)
    round_ids = [int(turn.metadata.get("round_id", 0)) for turn in turns]
    return max(round_ids, default=0) + 1


def _get_matrix_score(
    matrix: dict[tuple[int, int], float],
    *,
    total_agents: int,
    finished_round: int,
    round_id: int,
    agent_index: int,
) -> float:
    if finished_round != -1 and round_id > finished_round:
        return -1.0
    if agent_index < 0 or agent_index >= total_agents:
        return -1.0
    return float(matrix.get((round_id, agent_index), -1.0))


def _mean_other(
    matrix: dict[tuple[int, int], float],
    *,
    total_agents: int,
    finished_round: int,
    round_id: int,
    agent_index: int,
) -> float:
    values = [
        _get_matrix_score(
            matrix,
            total_agents=total_agents,
            finished_round=finished_round,
            round_id=round_id,
            agent_index=idx,
        )
        for idx in range(total_agents)
        if idx != agent_index
    ]
    valid = [value for value in values if value != -1.0]
    if not valid:
        return -1.0
    return sum(valid) / len(valid)


def _score_rule(
    matrix: dict[tuple[int, int], float],
    *,
    total_rounds: int,
    total_agents: int,
    finished_round: int,
    round_id: int,
    agent_index: int,
    rule_horizon: str,
    rule_agent_share: str,
    rule_discount: float,
    correct_threshold: float,
    wrong_threshold: float,
) -> float:
    del correct_threshold, wrong_threshold

    def score_at(t: int, a: int) -> float:
        return _get_matrix_score(
            matrix,
            total_agents=total_agents,
            finished_round=finished_round,
            round_id=t,
            agent_index=a,
        )

    final_round = total_rounds - 1 if finished_round == -1 else finished_round
    if round_id > final_round:
        return -1.0

    if rule_horizon == "last" and rule_agent_share == "all":
        values = [score_at(final_round, idx) for idx in range(total_agents)]
        valid = [value for value in values if value != -1.0]
        return sum(valid) / len(valid) if valid else -1.0
    if rule_horizon == "last" and rule_agent_share == "individual":
        return score_at(final_round, agent_index)
    if rule_horizon == "current" and rule_agent_share == "all":
        values = [score_at(round_id, idx) for idx in range(total_agents)]
        valid = [value for value in values if value != -1.0]
        return sum(valid) / len(valid) if valid else -1.0
    if rule_horizon == "current" and rule_agent_share == "individual":
        return score_at(round_id, agent_index)
    if rule_horizon != "discounted_sum":
        raise ValueError(f"Unsupported MAPoRL rule_horizon: {rule_horizon}")

    discounted_factors = [
        rule_discount ** (future_round - round_id) for future_round in range(round_id, final_round + 1)
    ]
    discount_sum = sum(discounted_factors) or 1.0
    if rule_agent_share == "individual":
        total = 0.0
        for future_round, factor in zip(range(round_id, final_round + 1), discounted_factors, strict=True):
            value = score_at(future_round, agent_index)
            total += 0.0 if value == -1.0 else factor * value
        return total / discount_sum
    if rule_agent_share == "all":
        current = score_at(round_id, agent_index)
        current = 0.0 if current == -1.0 else current
        future_total = 0.0
        for future_round in range(round_id + 1, final_round + 1):
            values = [score_at(future_round, idx) for idx in range(total_agents)]
            valid = [value for value in values if value != -1.0]
            future_total += (rule_discount ** (future_round - round_id)) * (sum(valid) / len(valid) if valid else 0.0)
        return (current + future_total) / discount_sum
    raise ValueError(f"Unsupported MAPoRL rule_agent_share: {rule_agent_share}")


def _bonus_rule(
    matrix: dict[tuple[int, int], float],
    *,
    total_rounds: int,
    total_agents: int,
    finished_round: int,
    round_id: int,
    agent_index: int,
    alpha: tuple[float, float, float, float],
    correct_threshold: float,
    wrong_threshold: float,
) -> float:
    score = 0.0

    def current(t: int, a: int) -> float:
        return _get_matrix_score(
            matrix,
            total_agents=total_agents,
            finished_round=finished_round,
            round_id=t,
            agent_index=a,
        )

    if round_id > 0:
        prev_score = current(round_id - 1, agent_index)
        current_score = current(round_id, agent_index)
        prev_others_score = _mean_other(
            matrix,
            total_agents=total_agents,
            finished_round=finished_round,
            round_id=round_id - 1,
            agent_index=agent_index,
        )
        if prev_score > correct_threshold and current_score < wrong_threshold:
            score -= alpha[1] if prev_others_score > correct_threshold else alpha[0]
        if prev_score < wrong_threshold and current_score > correct_threshold:
            score += alpha[1] if prev_others_score < wrong_threshold else alpha[0]

    final_round = total_rounds - 1 if finished_round == -1 else finished_round
    if round_id < final_round and total_agents > 1:
        next_others_score = _mean_other(
            matrix,
            total_agents=total_agents,
            finished_round=finished_round,
            round_id=round_id + 1,
            agent_index=agent_index,
        )
        current_others_score = _mean_other(
            matrix,
            total_agents=total_agents,
            finished_round=finished_round,
            round_id=round_id,
            agent_index=agent_index,
        )
        current_score = current(round_id, agent_index)
        if next_others_score > correct_threshold and current_others_score < wrong_threshold:
            score -= alpha[3] if current_score > correct_threshold else alpha[2]
        if next_others_score < wrong_threshold and current_others_score > correct_threshold:
            score += alpha[3] if current_score < wrong_threshold else alpha[2]

    return score


def _config_path(config: Any, *keys: str) -> Any:
    current = config
    for key in keys:
        current = _config_get(current, key, _MISSING)
        if current is _MISSING:
            return None
    return current


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Unsupported boolean value: {value!r}")
    return bool(value)
