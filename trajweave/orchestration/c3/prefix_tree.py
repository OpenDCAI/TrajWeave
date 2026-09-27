from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.math.c3_text_sanitize import sanitize_math_solution_text


@dataclass(frozen=True)
class _PrefixState:
    parent_node_id: str | None
    role_outputs: dict[str, str]
    role_prompts: dict[str, str]


@dataclass(frozen=True)
class C3PrefixTreeOrchestra:
    """Build C3's nested role-prefix tree with fixed-context sibling replay."""

    fanout: tuple[int, ...] = (2, 2)
    role_order: tuple[str, ...] = ("reasoner", "actor")
    allow_singleton_fanout: bool = False

    def run_tree(
        self,
        *,
        episode_id: str,
        rollout_group: str,
        task: Any,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        environment: Any,
        branch_factor: int | None = None,
    ) -> MultiAgentTrajectory:
        del branch_factor
        roles, fanout = self._validate(team, environment)
        root_id = f"{episode_id}:c3-root"
        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=str(task.task_id),
            rollout_group=rollout_group,
            team_name=team.name,
            metadata={
                "root_id": root_id,
                "sampling_mode": "c3_nested_prefix_tree",
                "c3_fanout": list(fanout),
                "c3_role_order": list(self.role_order),
            },
        )
        states = [_PrefixState(parent_node_id=None, role_outputs={}, role_prompts={})]
        children_by_parent: dict[str, list[str]] = {}
        turns_by_node: dict[str, AgentTurn] = {}
        leaf_rewards: dict[str, float] = {}

        for depth, (role_name, count) in enumerate(zip(self.role_order, fanout, strict=True)):
            agent = roles[role_name]
            next_states: list[_PrefixState] = []
            for parent_index, state in enumerate(states):
                visible_context = self._render_visible_role_outputs(role=role_name, role_outputs=state.role_outputs)
                prompt = self._build_prompt(
                    role=role_name,
                    observation=observation,
                    role_outputs=state.role_outputs,
                )
                group_id = f"{root_id}:group:{depth}:{state.parent_node_id or 'root'}:{role_name}"
                siblings: list[AgentTurn] = []
                for branch_index in range(count):
                    node_id = f"{root_id}:node:{depth}:{parent_index}:{branch_index}"
                    response = policy_backend.generate(
                        PolicyRequest(
                            agent=agent,
                            task_id=str(task.task_id),
                            observation=observation,
                            prompt=prompt,
                            team_context=visible_context,
                            metadata={
                                "answer": task.answer,
                                "c3_depth": depth,
                                "c3_branch_index": branch_index,
                                "c3_parent_node_id": state.parent_node_id,
                            },
                        )
                    )
                    outputs = {**state.role_outputs, role_name: response.text}
                    prompts = {**state.role_prompts, role_name: prompt}
                    is_leaf = depth == len(self.role_order) - 1
                    reward = None
                    success = False
                    if is_leaf:
                        reward, success = environment.evaluate(task, response.text)
                        leaf_rewards[node_id] = float(reward)
                    turn = AgentTurn(
                        episode_id=episode_id,
                        task_id=str(task.task_id),
                        turn_id=depth,
                        agent_name=agent.name,
                        role=agent.role,
                        policy_group=agent.policy_group,
                        observation=observation,
                        prompt=prompt,
                        action_text=response.text,
                        action_token_ids=response.token_ids,
                        action_logprobs=response.logprobs,
                        reward=float(reward) if reward is not None else None,
                        done=is_leaf,
                        root_id=root_id,
                        node_id=node_id,
                        parent_node_id=state.parent_node_id,
                        observation_group_id=group_id,
                        branch_index=branch_index,
                        selected_for_expansion=not is_leaf,
                        local_score=float(reward) if reward is not None else None,
                        metadata={
                            **response.metadata,
                            "c3_group_id": group_id,
                            "c3_depth": depth,
                            "c3_role_index": depth,
                            "c3_parent_id": state.parent_node_id or "",
                            "c3_is_leaf": is_leaf,
                            "c3_leaf_success": bool(success),
                            "c3_question": observation,
                            "c3_prefix_outputs": outputs,
                            "c3_prefix_prompts": prompts,
                            "c3_prefix_text": self.format_prefix_text(observation, outputs),
                        },
                    )
                    trajectory.add_turn(turn)
                    siblings.append(turn)
                    turns_by_node[node_id] = turn
                    next_states.append(_PrefixState(node_id, outputs, prompts))
                self._assert_frozen_sibling_context(siblings, expected_size=count)
                children_by_parent[state.parent_node_id or root_id] = [
                    turn.node_id for turn in siblings if turn.node_id
                ]
            states = next_states

        subtree = self._materialize_subtree_returns(
            root_id=root_id,
            turns_by_node=turns_by_node,
            children_by_parent=children_by_parent,
            leaf_rewards=leaf_rewards,
        )
        for node_id, turn in turns_by_node.items():
            node_return, leaf_count = subtree[node_id]
            turn.reward = node_return
            turn.metadata["c3_subtree_return"] = node_return
            turn.metadata["c3_leaf_count"] = leaf_count

        best_leaf = min(
            (turn for turn in trajectory.turns if bool(turn.metadata["c3_is_leaf"])),
            key=lambda turn: (-float(turn.reward or 0.0), str(turn.node_id)),
        )
        trajectory.final_answer = best_leaf.action_text
        trajectory.global_reward = float(best_leaf.reward or 0.0)
        trajectory.success = bool(best_leaf.metadata["c3_leaf_success"])
        trajectory.metadata.update(
            {
                "selected_leaf_node_id": best_leaf.node_id,
                "leaf_count": len(leaf_rewards),
                "tree_node_count": len(turns_by_node),
            }
        )
        return trajectory

    def _validate(self, team: TeamSpec, environment: Any) -> tuple[dict[str, Any], tuple[int, ...]]:
        if not callable(getattr(environment, "evaluate", None)):
            raise TypeError("C3 prefix replay requires an environment with evaluate().")
        if len(self.fanout) != len(self.role_order):
            raise ValueError("C3 fanout must contain exactly one value per role.")
        minimum = 1 if self.allow_singleton_fanout else 2
        if any(isinstance(value, bool) or not isinstance(value, int) or value < minimum for value in self.fanout):
            if self.allow_singleton_fanout:
                raise ValueError("C3 evaluation fanout values must be positive integers.")
            raise ValueError("C3 requires integer fanout >= 2 at every role for leave-one-out credit.")
        fanout = tuple(self.fanout)
        roles = {agent.name: agent for agent in team.agents}
        missing = [name for name in self.role_order if name not in roles]
        if missing:
            raise ValueError(f"C3 role_order references unknown team agents: {missing}.")
        if len(roles) != len(self.role_order):
            raise ValueError("C3 currently requires the team agents to exactly match role_order.")
        return roles, fanout

    @staticmethod
    def _build_prompt(*, role: str, observation: str, role_outputs: dict[str, str]) -> str:
        context = C3PrefixTreeOrchestra._render_visible_role_outputs(role=role, role_outputs=role_outputs)
        if role.lower() == "reasoner":
            instruction = "Give the Actor a concise plan. Do not provide the final answer."
        elif role.lower() == "actor":
            instruction = (
                "Use the Reasoner's plan and put the final answer in \\boxed{...}; "
                "an explicit 'Final answer: ...' line is also accepted."
            )
        else:
            instruction = f"Act as {role} and produce the next collaboration output."
        return f"Question:\n{observation}\n\nRole prefix:\n{context}\n\n{instruction}"

    @staticmethod
    def _render_role_outputs(role_outputs: dict[str, str]) -> str:
        if not role_outputs:
            return "<empty>"
        return "\n\n".join(f"--- {role}'s Answer ---\n{text}" for role, text in role_outputs.items())

    @staticmethod
    def _render_visible_role_outputs(*, role: str, role_outputs: dict[str, str]) -> str:
        if role.lower() != "actor":
            return C3PrefixTreeOrchestra._render_role_outputs(role_outputs)
        sanitized = {name: sanitize_math_solution_text(text) for name, text in role_outputs.items()}
        return C3PrefixTreeOrchestra._render_role_outputs(sanitized)

    def format_prefix_text(self, question: str, role_outputs: dict[str, str]) -> str:
        parts = [f"Question: {question}\n"]
        for role in self.role_order:
            if role not in role_outputs:
                break
            parts.append(f"\n--- {role}'s Answer ---\n{role_outputs[role]}\n")
        return "".join(parts)

    @staticmethod
    def _assert_frozen_sibling_context(siblings: list[AgentTurn], *, expected_size: int) -> None:
        if len(siblings) != expected_size:
            raise RuntimeError(f"C3 sibling group expected {expected_size} rows, got {len(siblings)}.")
        prompts = {turn.prompt for turn in siblings}
        parents = {turn.parent_node_id for turn in siblings}
        groups = {turn.observation_group_id for turn in siblings}
        if len(prompts) != 1 or len(parents) != 1 or len(groups) != 1:
            raise RuntimeError("C3 sibling alternatives must share one frozen prompt, parent, and group id.")

    @staticmethod
    def _materialize_subtree_returns(
        *,
        root_id: str,
        turns_by_node: dict[str, AgentTurn],
        children_by_parent: dict[str, list[str]],
        leaf_rewards: dict[str, float],
    ) -> dict[str, tuple[float, int]]:
        cache: dict[str, tuple[float, int]] = {}

        def visit(node_id: str) -> tuple[float, int]:
            if node_id in cache:
                return cache[node_id]
            children = children_by_parent.get(node_id, [])
            if not children:
                if node_id not in leaf_rewards:
                    raise RuntimeError(f"C3 non-expanded node {node_id!r} has no leaf reward.")
                result = (float(leaf_rewards[node_id]), 1)
            else:
                values = [visit(child) for child in children]
                total_leaves = sum(count for _, count in values)
                result = (
                    sum(value * count for value, count in values) / float(total_leaves),
                    total_leaves,
                )
            cache[node_id] = result
            return result

        roots = children_by_parent.get(root_id, [])
        if not roots:
            raise RuntimeError("C3 tree did not produce root role candidates.")
        for node_id in turns_by_node:
            visit(node_id)
        return cache
