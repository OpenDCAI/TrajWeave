from __future__ import annotations

import asyncio
import inspect
import math
from dataclasses import dataclass
from typing import Any

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core import SearchNode, TeamSpec, TreeTrajectory
from trajweave.verifiers import VerifierAdapter, VerifierRequest


@dataclass
class TreeSearchProtocol:
    max_num_nodes: int = 2
    initial_candidates: int = 2
    exploration_constant: float = 1.0
    stop_on_success: bool = False
    min_num_nodes: int = 2
    refinement_concurrency: int = 2

    def run(
        self,
        *,
        tree_id: str,
        prompt_id: str,
        task: Any,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        verifier: VerifierAdapter,
    ) -> TreeTrajectory:
        if self.max_num_nodes <= 0:
            raise ValueError("TreeSearchProtocol.max_num_nodes must be positive.")
        if not team.agents:
            raise ValueError("TreeSearchProtocol requires at least one agent.")

        task_id = str(getattr(task, "task_id", prompt_id))
        trajectory = TreeTrajectory(
            tree_id=tree_id,
            prompt_id=prompt_id,
            task_id=task_id,
            rollout_group=tree_id,
            metadata={
                "control": "mcts_selection_expansion_refinement_termination",
                "max_num_nodes": self.max_num_nodes,
                "verifier": verifier.name,
            },
        )
        visits: dict[int, int] = {}
        value_sums: dict[int, float] = {}

        for node_id in range(self.max_num_nodes):
            parent = self.select(trajectory, visits=visits, value_sums=value_sums)
            agent = team.agents[node_id % len(team.agents)]
            prompt = self.build_expansion_prompt(observation, parent=parent)
            response = policy_backend.generate(
                PolicyRequest(
                    agent=agent,
                    task_id=task_id,
                    observation=observation,
                    prompt=prompt,
                    metadata={
                        "tree_id": tree_id,
                        "prompt_id": prompt_id,
                        "node_id": node_id,
                        "parent_idx": None if parent is None else parent.node_id,
                        "search_stage": "expand" if parent is None else "refine",
                    },
                )
            )
            result = verifier.verify(
                VerifierRequest(
                    task_id=task_id,
                    prompt=observation,
                    candidate=response.text,
                    node_id=node_id,
                    parent_idx=None if parent is None else parent.node_id,
                    metadata=response.metadata,
                )
            )
            path = (node_id,) if parent is None else (*parent.path, node_id)
            node = SearchNode(
                tree_id=tree_id,
                prompt_id=prompt_id,
                node_id=node_id,
                parent_idx=None if parent is None else parent.node_id,
                turn_id=node_id,
                agent_name=agent.name,
                role=agent.role,
                policy_group=agent.policy_group,
                prompt=prompt,
                action_text=response.text,
                action_token_ids=response.token_ids,
                rollout_logprobs=response.logprobs,
                reward=result.score,
                path=path,
                is_terminal=result.terminal,
                rollout_policy_step=_as_int(response.metadata.get("rollout_policy_step")),
                rollout_global_step=_as_int(response.metadata.get("rollout_global_step")),
                policy_lag=_as_int(response.metadata.get("policy_lag")),
                metadata=response.metadata
                | result.metadata
                | {
                    "verifier_feedback": result.feedback,
                    "verifier_success": result.success,
                    "search_stage": "expand" if parent is None else "refine",
                },
            )
            trajectory.add_node(node)
            self.backpropagate(node, result.score, trajectory, visits=visits, value_sums=value_sums)
            if self.should_stop(trajectory, latest=node, success=result.success):
                break

        best = self.aggregate(trajectory)
        trajectory.final_answer = best.action_text
        trajectory.global_reward = best.effective_reward
        trajectory.success = any(bool(node.metadata.get("verifier_success")) for node in trajectory.nodes)
        trajectory.metadata["best_node_id"] = best.node_id
        trajectory.metadata["node_count"] = len(trajectory.nodes)
        return trajectory

    async def run_async(
        self,
        *,
        tree_id: str,
        prompt_id: str,
        task: Any,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        verifier: VerifierAdapter,
    ) -> TreeTrajectory:
        """Run initial candidates and refinement waves concurrently.

        Every request reserves its node id before awaiting.  Completed futures
        are sorted by that id before insertion, so network completion order can
        never change tree topology or aggregation.
        """
        if self.max_num_nodes <= 0 or not team.agents:
            raise ValueError("TreeSearchProtocol requires positive max_num_nodes and a non-empty team")
        concurrency = max(1, int(self.refinement_concurrency))
        task_id = str(getattr(task, "task_id", prompt_id))
        trajectory = TreeTrajectory(
            tree_id=tree_id,
            prompt_id=prompt_id,
            task_id=task_id,
            rollout_group=tree_id,
            metadata={"control": "async_wave_mcts", "max_num_nodes": self.max_num_nodes, "verifier": verifier.name},
        )
        visits: dict[int, int] = {}
        value_sums: dict[int, float] = {}
        next_id = 0
        # Initial candidates are independent and can be generated together.
        while next_id < self.max_num_nodes:
            wave_ids = list(
                range(
                    next_id,
                    min(next_id + (self.initial_candidates if next_id == 0 else concurrency), self.max_num_nodes),
                )
            )
            next_id += len(wave_ids)
            requests = []
            for node_id in wave_ids:
                parent = (
                    None
                    if node_id < self.initial_candidates
                    else self.select(trajectory, visits=visits, value_sums=value_sums)
                )
                agent = team.agents[node_id % len(team.agents)]
                prompt = self.build_expansion_prompt(observation, parent=parent)
                requests.append((node_id, parent, agent, prompt))

            async def produce(item):
                node_id, parent, agent, prompt = item
                response = await _maybe_await(
                    policy_backend.generate,
                    PolicyRequest(
                        agent=agent,
                        task_id=task_id,
                        observation=observation,
                        prompt=prompt,
                        metadata={
                            "tree_id": tree_id,
                            "prompt_id": prompt_id,
                            "node_id": node_id,
                            "parent_idx": None if parent is None else parent.node_id,
                            "search_stage": "expand" if parent is None else "refine",
                        },
                    ),
                )
                result = await _maybe_await(
                    verifier.verify,
                    VerifierRequest(
                        task_id=task_id,
                        prompt=observation,
                        candidate=response.text,
                        node_id=node_id,
                        parent_idx=None if parent is None else parent.node_id,
                        metadata=response.metadata,
                    ),
                )
                path = (node_id,) if parent is None else (*parent.path, node_id)
                return SearchNode(
                    tree_id=tree_id,
                    prompt_id=prompt_id,
                    node_id=node_id,
                    parent_idx=None if parent is None else parent.node_id,
                    turn_id=node_id,
                    agent_name=agent.name,
                    role=agent.role,
                    policy_group=agent.policy_group,
                    prompt=prompt,
                    action_text=response.text,
                    action_token_ids=response.token_ids,
                    rollout_logprobs=response.logprobs,
                    reward=result.score,
                    path=path,
                    is_terminal=result.terminal,
                    rollout_policy_step=_as_int(response.metadata.get("rollout_policy_step")),
                    rollout_global_step=_as_int(response.metadata.get("rollout_global_step")),
                    policy_lag=_as_int(response.metadata.get("policy_lag")),
                    metadata=response.metadata
                    | result.metadata
                    | {
                        "verifier_feedback": result.feedback,
                        "verifier_success": result.success,
                        "search_stage": "expand" if parent is None else "refine",
                    },
                )

            nodes = await asyncio.gather(*(produce(item) for item in requests))
            for node in sorted(nodes, key=lambda item: item.node_id):
                trajectory.add_node(node)
                self.backpropagate(node, float(node.reward or 0.0), trajectory, visits=visits, value_sums=value_sums)
            if (
                nodes
                and any(bool(node.metadata.get("verifier_success")) for node in nodes)
                and self.stop_on_success
                and len(trajectory.nodes) >= self.min_num_nodes
            ):
                break
        best = self.aggregate(trajectory)
        trajectory.final_answer, trajectory.global_reward = best.action_text, best.effective_reward
        trajectory.success = any(bool(node.metadata.get("verifier_success")) for node in trajectory.nodes)
        trajectory.metadata.update(
            {"best_node_id": best.node_id, "node_count": len(trajectory.nodes), "completion_order_independent": True}
        )
        return trajectory

    def select(
        self,
        trajectory: TreeTrajectory,
        *,
        visits: dict[int, int],
        value_sums: dict[int, float],
    ) -> SearchNode | None:
        if len(trajectory.nodes) < min(self.initial_candidates, self.max_num_nodes):
            return None
        candidates = [node for node in trajectory.nodes if not node.is_terminal]
        if not candidates:
            return None
        total_visits = max(1, sum(visits.values()))

        def ucb(node: SearchNode) -> tuple[float, int]:
            node_visits = visits.get(node.node_id, 0)
            mean_value = value_sums.get(node.node_id, 0.0) / max(node_visits, 1)
            exploration = self.exploration_constant * math.sqrt(math.log(total_visits + 1) / (node_visits + 1))
            return mean_value + exploration, -node.node_id

        return max(candidates, key=ucb)

    def build_expansion_prompt(self, observation: str, *, parent: SearchNode | None) -> str:
        if parent is None:
            return observation
        feedback = str(parent.metadata.get("verifier_feedback", "")).strip()
        suffix = f"\n\nVerifier feedback:\n{feedback}" if feedback else ""
        return (
            f"{observation}\n\nPrevious candidate:\n{parent.action_text}{suffix}\n\n"
            "Refine the candidate and return a complete replacement answer."
        )

    def backpropagate(
        self,
        node: SearchNode,
        score: float,
        trajectory: TreeTrajectory,
        *,
        visits: dict[int, int],
        value_sums: dict[int, float],
    ) -> None:
        nodes_by_id = {item.node_id: item for item in trajectory.nodes}
        current: SearchNode | None = node
        while current is not None:
            visits[current.node_id] = visits.get(current.node_id, 0) + 1
            value_sums[current.node_id] = value_sums.get(current.node_id, 0.0) + float(score)
            current = nodes_by_id.get(current.parent_idx) if current.parent_idx is not None else None

    def should_stop(self, trajectory: TreeTrajectory, *, latest: SearchNode, success: bool) -> bool:
        if len(trajectory.nodes) >= self.max_num_nodes:
            return True
        return self.stop_on_success and success and len(trajectory.nodes) >= self.min_num_nodes and latest.is_terminal

    def aggregate(self, trajectory: TreeTrajectory) -> SearchNode:
        if not trajectory.nodes:
            raise ValueError("Cannot aggregate an empty tree trajectory.")
        return max(trajectory.nodes, key=lambda node: (node.effective_reward, -node.node_id))


async def _maybe_await(function, *args, **kwargs):
    value = function(*args, **kwargs)
    return await value if inspect.isawaitable(value) else value


def _as_int(value: Any) -> int | None:
    return None if value is None else int(value)
