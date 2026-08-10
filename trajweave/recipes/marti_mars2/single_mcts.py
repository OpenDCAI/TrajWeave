from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from trajweave.backends.policy import PolicyRequest, PolicyResponse
from trajweave.core import AgentSpec, AgentTurn, MultiAgentTrajectory, PolicyGroupSpec, TeamSpec, TreeTrajectory
from trajweave.credit import TreeGroupCreditAllocator, TreePathCreditAllocator
from trajweave.orchestration import TreeSearchProtocol
from trajweave.rollout.engine import RolloutResult
from trajweave.verifiers import CallableVerifierAdapter, VerifierRequest


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str
    rewards: tuple[float, ...]


class _SmokePolicyBackend:
    def generate(self, request: PolicyRequest) -> PolicyResponse:
        node_id = int(request.metadata["node_id"])
        return PolicyResponse(
            text=f"candidate-{node_id}",
            token_ids=[100 + node_id],
            logprobs=[-0.1 * (node_id + 1)],
            metadata={"backend": "marti_mars2_smoke", **request.metadata},
        )


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


def run_single_mcts_smoke(
    *, max_num_nodes: int = 2, rollouts_per_task: int = 1, credit_mode: str = "fidelity"
) -> tuple[object, RolloutResult]:
    team = default_marti_mars2_team(max_num_nodes=max_num_nodes)
    policy_backend = _SmokePolicyBackend()
    protocol = TreeSearchProtocol(max_num_nodes=max_num_nodes, initial_candidates=min(2, max_num_nodes))
    if credit_mode == "fidelity":
        credit_allocator = TreeGroupCreditAllocator()
    elif credit_mode == "experimental":
        credit_allocator = TreePathCreditAllocator()
    else:
        raise ValueError(f"Unsupported MARTI-MARS² credit_mode: {credit_mode!r}.")
    trees: list[TreeTrajectory] = []
    trajectories: list[MultiAgentTrajectory] = []
    for rollout_idx in range(rollouts_per_task):
        for task in default_code_tasks(max_num_nodes):
            tree_id = f"{task.task_id}:rollout-{rollout_idx}"
            verifier = CallableVerifierAdapter(
                lambda request, rewards=task.rewards: _verify_smoke_candidate(request, rewards),
                name="deterministic_code_verifier",
            )
            tree = protocol.run(
                tree_id=tree_id,
                prompt_id=task.task_id,
                task=task,
                team=team,
                observation=task.prompt,
                policy_backend=policy_backend,
                verifier=verifier,
            )
            trees.append(tree)
            trajectory = MultiAgentTrajectory(
                episode_id=tree.tree_id,
                task_id=tree.task_id,
                rollout_group=tree.rollout_group,
                team_name=team.name,
                final_answer=tree.final_answer,
                global_reward=tree.global_reward,
                success=tree.success,
                metadata={**tree.metadata, "tree_id": tree.tree_id, "prompt_id": tree.prompt_id},
            )
            for node in tree.nodes:
                trajectory.add_turn(
                    AgentTurn(
                        episode_id=trajectory.episode_id,
                        task_id=task.task_id,
                        turn_id=node.turn_id if node.turn_id is not None else node.node_id,
                        agent_name=node.agent_name,
                        role=node.role,
                        policy_group=node.policy_group,
                        observation=task.prompt,
                        prompt=task.prompt,
                        action_text=node.action_text,
                        action_token_ids=node.action_token_ids,
                        action_logprobs=node.rollout_logprobs,
                        reward=node.reward,
                        metadata={
                            "tree_id": tree_id,
                            "prompt_id": task.task_id,
                            "node_id": node.node_id,
                            "parent_idx": node.parent_idx,
                            "path": list(node.path),
                            "credit": credit_allocator.name,
                            **node.metadata,
                        },
                    )
                )
            trajectories.append(trajectory)

    samples = credit_allocator.assign(trees, team)

    summary = SimpleNamespace(
        trajectories=len(trajectories),
        samples=len(samples),
        success_rate=sum(1 for item in trajectories if item.success) / len(trajectories),
        dataproto_rows=None,
        dataproto_status="skipped",
    )
    return summary, RolloutResult(trajectories=trajectories, samples=samples)


def _verify_smoke_candidate(request: VerifierRequest, rewards: tuple[float, ...]) -> dict[str, object]:
    score = float(rewards[request.node_id])
    return {
        "score": score,
        "success": score > 0.0,
        "terminal": False,
        "feedback": "passes deterministic tests" if score > 0.0 else "fails deterministic tests",
        "verifier": "deterministic_code_verifier",
    }
