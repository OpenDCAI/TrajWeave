from __future__ import annotations

import random
from collections.abc import Sequence
from math import isfinite
from statistics import fmean

from trajweave.core.joint_trajectory import JointAction, JointCompletion
from trajweave.core.preference import JointPreferencePair
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory, TrainingSample


def build_joint_preference_pairs(
    trajectory: MultiAgentTrajectory,
    *,
    pair_selection: str = "reward_gap",
    pairs_per_sample: int = 16,
    random_seed: int = 0,
) -> list[JointPreferencePair]:
    selection = str(pair_selection).strip().lower()
    if selection not in {"reward_gap", "random", "all"}:
        raise ValueError("pair_selection must be one of: reward_gap, random, all")
    if isinstance(pairs_per_sample, bool) or not isinstance(pairs_per_sample, int) or pairs_per_sample < 1:
        raise ValueError("pairs_per_sample must be a positive integer")
    mode = _joint_mode(trajectory)
    if mode != "aligned":
        raise ValueError("MADPO/MARLHF preference generation requires aligned joint sampling")

    nodes = {node.tree_node_id: node for node in trajectory.joint_nodes}
    completions = {completion.completion_id: completion for completion in trajectory.joint_completions}
    actions_by_node: dict[str, list[JointAction]] = {}
    for action in trajectory.joint_actions:
        node = nodes.get(action.tree_node_id)
        if node is None:
            raise ValueError(f"preference action {action.joint_action_id!r} references an unknown node")
        if node.depth != 0 or action.child_tree_node_id is not None:
            raise ValueError("MADPO/MARLHF v1.4.1 preference generation requires one-turn joint actions")
        _validate_aligned_action(action, completions)
        actions_by_node.setdefault(action.tree_node_id, []).append(action)

    output: list[JointPreferencePair] = []
    for node_id, actions in actions_by_node.items():
        ordered = sorted(actions, key=_aligned_candidate_index)
        candidate_indices = [_aligned_candidate_index(action) for action in ordered]
        if len(candidate_indices) != len(set(candidate_indices)):
            raise ValueError(f"aligned preference node {node_id!r} has duplicate candidate indices")
        rewards = [_finite_reward(action) for action in ordered]
        candidate_mean = fmean(rewards)
        candidates: list[tuple[int, int, float]] = []
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                if rewards[left] == rewards[right]:
                    continue
                winner, loser = (left, right) if rewards[left] > rewards[right] else (right, left)
                candidates.append((winner, loser, rewards[winner] - rewards[loser]))
        if selection == "reward_gap":
            candidates.sort(
                key=lambda item: (
                    -item[2],
                    candidate_indices[item[0]],
                    candidate_indices[item[1]],
                )
            )
        elif selection == "random":
            random.Random(f"{random_seed}:{trajectory.episode_id}:{node_id}").shuffle(candidates)
        else:
            candidates.sort(key=lambda item: (candidate_indices[item[0]], candidate_indices[item[1]]))
        selected = candidates if selection == "all" else candidates[:pairs_per_sample]
        for pair_index, (winner_index, loser_index, gap) in enumerate(selected):
            winner = ordered[winner_index]
            loser = ordered[loser_index]
            prompts, chosen, rejected = _pair_text(winner, loser, completions)
            output.append(
                JointPreferencePair(
                    preference_pair_id=(
                        f"{trajectory.episode_id}:{node_id}:preference:"
                        f"{candidate_indices[winner_index]}:{candidate_indices[loser_index]}:{pair_index}"
                    ),
                    episode_id=trajectory.episode_id,
                    tree_node_id=node_id,
                    chosen_joint_action_id=winner.joint_action_id,
                    rejected_joint_action_id=loser.joint_action_id,
                    prompts_by_agent=prompts,
                    chosen_by_agent=chosen,
                    rejected_by_agent=rejected,
                    chosen_reward=rewards[winner_index],
                    rejected_reward=rewards[loser_index],
                    candidate_mean=candidate_mean,
                    metadata={
                        "pair_selection": selection,
                        "winner_candidate_index": candidate_indices[winner_index],
                        "loser_candidate_index": candidate_indices[loser_index],
                        "reward_gap": gap,
                        "candidate_rewards": rewards,
                    },
                )
            )
    return output


def preference_pairs_to_training_samples(
    trajectory: MultiAgentTrajectory,
    pairs: Sequence[JointPreferencePair],
    team: TeamSpec,
) -> list[TrainingSample]:
    actions = {action.joint_action_id: action for action in trajectory.joint_actions}
    completions = {completion.completion_id: completion for completion in trajectory.joint_completions}
    agent_order = tuple(agent.name for agent in team.agents)
    output: list[TrainingSample] = []
    for pair in pairs:
        pair.validate()
        if pair.episode_id != trajectory.episode_id:
            raise ValueError(f"preference pair {pair.preference_pair_id!r} belongs to another episode")
        chosen_action = actions.get(pair.chosen_joint_action_id)
        rejected_action = actions.get(pair.rejected_joint_action_id)
        if chosen_action is None or rejected_action is None:
            raise ValueError(f"preference pair {pair.preference_pair_id!r} references an unknown joint action")
        if chosen_action.tree_node_id != pair.tree_node_id or rejected_action.tree_node_id != pair.tree_node_id:
            raise ValueError(f"preference pair {pair.preference_pair_id!r} actions must share tree_node_id")
        if set(pair.prompts_by_agent) != set(agent_order):
            raise ValueError(f"preference pair {pair.preference_pair_id!r} must cover every team agent")
        for agent_name in agent_order:
            agent = team.agent(agent_name)
            for side, action, text, reward in (
                ("chosen", chosen_action, pair.chosen_by_agent[agent_name], pair.chosen_reward),
                ("rejected", rejected_action, pair.rejected_by_agent[agent_name], pair.rejected_reward),
            ):
                completion = completions[action.completion_ids[agent_name]]
                if completion.text != text:
                    raise ValueError(
                        f"preference pair {pair.preference_pair_id!r} {side} text disagrees with completion"
                    )
                output.append(
                    TrainingSample(
                        sample_id=f"{pair.preference_pair_id}:{agent_name}:{side}",
                        episode_id=trajectory.episode_id,
                        task_id=trajectory.task_id,
                        rollout_group=trajectory.rollout_group,
                        turn_id=0,
                        agent_name=agent_name,
                        role=agent.role,
                        policy_group=agent.policy_group,
                        prompt=pair.prompts_by_agent[agent_name],
                        response=text,
                        response_token_ids=list(completion.token_ids),
                        response_logprobs=list(completion.logprobs),
                        reward=reward,
                        completion_id=completion.completion_id,
                        tree_node_id=pair.tree_node_id,
                        joint_action_ids=[action.joint_action_id],
                        joint_transition_ids=[action.joint_transition_id],
                        metadata={
                            "preference_pair_id": pair.preference_pair_id,
                            "preference_side": side,
                            "chosen_reward": pair.chosen_reward,
                            "rejected_reward": pair.rejected_reward,
                            "candidate_mean": pair.candidate_mean,
                            "preference_loss_mask": 1.0,
                            "joint_reward": reward,
                            "joint_done": bool(action.done),
                            "joint_truncated": bool(action.truncated),
                            "joint_stop_reason": action.stop_reason or "",
                            "joint_sampling_mode": "aligned",
                        },
                    )
                )
    return output


def _validate_aligned_action(action: JointAction, completions: dict[str, JointCompletion]) -> None:
    if not action.completion_ids or set(action.completion_ids) != set(action.candidate_indices):
        raise ValueError(f"aligned action {action.joint_action_id!r} has inconsistent agent keys")
    candidate_indices = set(action.candidate_indices.values())
    if len(candidate_indices) != 1:
        raise ValueError(f"aligned action {action.joint_action_id!r} must use one shared candidate index")
    for agent_name, completion_id in action.completion_ids.items():
        completion = completions.get(completion_id)
        if completion is None:
            raise ValueError(f"aligned action {action.joint_action_id!r} references an unknown completion")
        if completion.agent_name != agent_name or completion.tree_node_id != action.tree_node_id:
            raise ValueError(f"aligned action {action.joint_action_id!r} references a mismatched completion")
        if completion.candidate_index != action.candidate_indices[agent_name]:
            raise ValueError(f"aligned action {action.joint_action_id!r} has a mismatched candidate index")


def _aligned_candidate_index(action: JointAction) -> int:
    indices = set(action.candidate_indices.values())
    if len(indices) != 1:
        raise ValueError(f"aligned action {action.joint_action_id!r} must use one candidate index")
    return next(iter(indices))


def _finite_reward(action: JointAction) -> float:
    if action.shared_reward is None:
        raise ValueError(f"preference action {action.joint_action_id!r} is missing shared_reward")
    reward = float(action.shared_reward)
    if not isfinite(reward):
        raise ValueError(f"preference action {action.joint_action_id!r} has a non-finite reward")
    return reward


def _pair_text(
    winner: JointAction,
    loser: JointAction,
    completions: dict[str, JointCompletion],
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if set(winner.completion_ids) != set(loser.completion_ids):
        raise ValueError("preference actions must contain identical agent keys")
    prompts: dict[str, str] = {}
    chosen: dict[str, str] = {}
    rejected: dict[str, str] = {}
    for agent_name in winner.completion_ids:
        winner_completion = completions[winner.completion_ids[agent_name]]
        loser_completion = completions[loser.completion_ids[agent_name]]
        winner_prompt = str(winner_completion.metadata.get("prompt", ""))
        loser_prompt = str(loser_completion.metadata.get("prompt", ""))
        if winner_prompt != loser_prompt:
            raise ValueError("chosen and rejected completions must share the same prompt for each agent")
        prompts[agent_name] = winner_prompt
        chosen[agent_name] = winner_completion.text
        rejected[agent_name] = loser_completion.text
    return prompts, chosen, rejected


def _joint_mode(trajectory: MultiAgentTrajectory) -> str:
    joint_tree = trajectory.metadata.get("joint_tree")
    value = joint_tree.get("joint_mode") if isinstance(joint_tree, dict) else None
    mode = str(trajectory.metadata.get("joint_sampling_mode", value or "aligned")).strip().lower()
    return {"align": "aligned", "aligned": "aligned", "cross": "cross", "crossed": "cross"}.get(mode, mode)


__all__ = ["build_joint_preference_pairs", "preference_pairs_to_training_samples"]
