from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from trajweave.core import AgentSpec, AgentTurn, MultiAgentTrajectory, PolicyGroupSpec, SearchNode, TeamSpec
from trajweave.core.trajectory import TrainingSample
from trajweave.credit import TreeGroupBuilder, group_normalized_advantages, rewards_from_nodes
from trajweave.rollout.engine import RolloutResult


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str
    rewards: tuple[float, ...]


def default_marti_mars2_team(max_num_nodes: int = 2) -> TeamSpec:
    return TeamSpec(
        name="marti_mars2_single_mcts",
        agents=(
            AgentSpec(
                name="generator",
                role="generator",
                policy_group="shared",
                trainable=True,
                generation_config={"max_num_nodes": max_num_nodes},
            ),
        ),
        policy_groups=(PolicyGroupSpec(name="shared", backend="local", trainable=True),),
        orchestra="mcts_tree_search",
        reward="code_verifier",
        credit="tree_group_norm",
        max_turns=max_num_nodes,
        metadata={
            "control": "mcts_selection_expansion_refinement_termination",
            "communication_graph": "single_agent_tree",
            "aggregation": "best_path_or_mcts_eval",
            "training_target": "single_trainable_policy",
        },
    )


def default_code_tasks(max_num_nodes: int) -> list[CodeTask]:
    if max_num_nodes < 2:
        raise ValueError("MARTI-MARS2 smoke requires max_num_nodes >= 2 for group advantage.")
    return [
        CodeTask(
            task_id="code-add-one",
            prompt="Implement add_one(x).",
            rewards=tuple([1.0, 0.0] + [0.0] * (max_num_nodes - 2)),
        ),
        CodeTask(
            task_id="code-square",
            prompt="Implement square(x).",
            rewards=tuple([1.0] * max_num_nodes),
        ),
    ]


def run_single_mcts_smoke(*, max_num_nodes: int = 2, rollouts_per_task: int = 1) -> tuple[object, RolloutResult]:
    team = default_marti_mars2_team(max_num_nodes=max_num_nodes)
    nodes: list[SearchNode] = []
    trajectories: list[MultiAgentTrajectory] = []
    for rollout_idx in range(rollouts_per_task):
        for task in default_code_tasks(max_num_nodes):
            tree_id = f"{task.task_id}:rollout-{rollout_idx}"
            trajectory = MultiAgentTrajectory(
                episode_id=tree_id,
                task_id=task.task_id,
                rollout_group=tree_id,
                team_name=team.name,
                final_answer=f"candidate-{task.rewards.index(max(task.rewards))}",
                global_reward=max(task.rewards),
                success=max(task.rewards) > 0,
                metadata={"tree_id": tree_id, "prompt_id": task.task_id, "max_num_nodes": max_num_nodes},
            )
            for node_id, reward in enumerate(task.rewards):
                node = SearchNode(
                    tree_id=tree_id,
                    prompt_id=task.task_id,
                    node_id=node_id,
                    parent_idx=None if node_id == 0 else 0,
                    turn_id=node_id,
                    agent_name="generator",
                    role="generator",
                    policy_group="shared",
                    prompt=task.prompt,
                    action_text=f"candidate-{node_id}",
                    action_token_ids=[100 + node_id],
                    rollout_logprobs=[-0.1 * (node_id + 1)],
                    reward=reward,
                    path=(0, node_id) if node_id else (0,),
                )
                nodes.append(node)
                trajectory.add_turn(
                    AgentTurn(
                        episode_id=trajectory.episode_id,
                        task_id=task.task_id,
                        turn_id=node_id,
                        agent_name=node.agent_name,
                        role=node.role,
                        policy_group=node.policy_group,
                        observation=task.prompt,
                        prompt=task.prompt,
                        action_text=node.action_text,
                        action_token_ids=node.action_token_ids,
                        action_logprobs=node.rollout_logprobs,
                        reward=reward,
                        metadata={
                            "tree_id": tree_id,
                            "prompt_id": task.task_id,
                            "node_id": node_id,
                            "parent_idx": node.parent_idx,
                            "path": list(node.path),
                            "credit": "tree_group_norm",
                        },
                    )
                )
            trajectories.append(trajectory)

    groups = TreeGroupBuilder().group_indices(nodes)
    advantages = group_normalized_advantages(rewards_from_nodes(nodes), groups)
    samples = [
        TrainingSample(
            sample_id=f"{node.tree_id}:{node.node_id}:generator",
            episode_id=node.tree_id,
            task_id=node.prompt_id,
            rollout_group=node.tree_id,
            turn_id=node.turn_id if node.turn_id is not None else node.node_id,
            agent_name=node.agent_name,
            role=node.role,
            policy_group=node.policy_group,
            prompt=node.prompt,
            response=node.action_text,
            response_token_ids=node.action_token_ids,
            response_logprobs=node.rollout_logprobs,
            reward=node.effective_reward,
            advantage=advantages[index],
            metadata={
                "credit": "tree_group_norm",
                "tree_id": node.tree_id,
                "prompt_id": node.prompt_id,
                "node_id": node.node_id,
                "parent_idx": node.parent_idx,
                "path": list(node.path),
                "rollout_logprob_available": bool(node.rollout_logprobs),
            },
        )
        for index, node in enumerate(nodes)
    ]

    summary = SimpleNamespace(
        trajectories=len(trajectories),
        samples=len(samples),
        success_rate=sum(1 for item in trajectories if item.success) / len(trajectories),
        dataproto_rows=None,
        dataproto_status="skipped",
    )
    return summary, RolloutResult(trajectories=trajectories, samples=samples)
