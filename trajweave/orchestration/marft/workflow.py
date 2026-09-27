# Copyright 2025 Junwei Liao, Shanghai Jiao Tong University and Shanghai Innovation Institute.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory


@dataclass(frozen=True)
class MARFTGraphNode:
    node_id: str
    role_name: str
    transition_message: str | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MARFTGraphNode:
        if not isinstance(config, dict):
            raise TypeError("MARFT graph nodes must be mappings.")
        node_id = str(config.get("id", config.get("node_id", ""))).strip()
        role_name = str(config.get("role_name", "")).strip()
        if not node_id or not role_name:
            raise ValueError("MARFT graph nodes require non-empty id and role_name values.")
        transition = config.get("transition_message")
        return cls(
            node_id=node_id,
            role_name=role_name,
            transition_message=None if transition is None else str(transition),
        )


@dataclass(frozen=True)
class MARFTWorkflowGraph:
    nodes: tuple[MARFTGraphNode, ...]
    edges: tuple[tuple[str, str], ...]

    @classmethod
    def sequential(
        cls,
        role_names: tuple[str, ...] | list[str],
        transition_messages: tuple[str | None, ...] | list[str | None] | None = None,
    ) -> MARFTWorkflowGraph:
        roles = tuple(str(name).strip() for name in role_names)
        if not roles or any(not name for name in roles):
            raise ValueError("MARFT sequential role_names must contain non-empty values.")
        if transition_messages is None:
            transitions = (None,) * len(roles)
        else:
            transitions = tuple(transition_messages)
            if len(transitions) != len(roles):
                raise ValueError("MARFT transition_messages must have the same length as role_names.")
        nodes = tuple(
            MARFTGraphNode(
                node_id=f"{role_name}_{index}",
                role_name=role_name,
                transition_message=None if transitions[index] is None else str(transitions[index]),
            )
            for index, role_name in enumerate(roles)
        )
        edges = tuple((nodes[index - 1].node_id, nodes[index].node_id) for index in range(1, len(nodes)))
        graph = cls(nodes=nodes, edges=edges)
        graph.validate()
        return graph

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MARFTWorkflowGraph:
        if not isinstance(config, dict):
            raise TypeError("MARFT graph_config must be a mapping.")
        raw_nodes = config.get("nodes", [])
        raw_edges = config.get("edges", [])
        if not isinstance(raw_nodes, list | tuple) or not isinstance(raw_edges, list | tuple):
            raise ValueError("MARFT graph_config nodes and edges must be sequences.")
        nodes = tuple(MARFTGraphNode.from_config(dict(node)) for node in raw_nodes)
        edges: list[tuple[str, str]] = []
        for edge in raw_edges:
            if not isinstance(edge, list | tuple) or len(edge) != 2:
                raise ValueError("Each MARFT graph edge must contain exactly [source, target].")
            edges.append((str(edge[0]), str(edge[1])))
        graph = cls(nodes=nodes, edges=tuple(edges))
        graph.validate()
        return graph

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("MARFT workflow graph must contain at least one node.")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("MARFT workflow graph node ids must be unique.")
        known = set(node_ids)
        for source, target in self.edges:
            if source not in known or target not in known:
                raise ValueError(f"MARFT graph edge {source!r}->{target!r} references an unknown node.")
            if source == target:
                raise ValueError("MARFT workflow graph cannot contain self edges.")
        self.execution_layers()

    def execution_layers(self) -> tuple[tuple[MARFTGraphNode, ...], ...]:
        by_id = {node.node_id: node for node in self.nodes}
        order = {node.node_id: index for index, node in enumerate(self.nodes)}
        in_degree = dict.fromkeys(by_id, 0)
        children: dict[str, list[str]] = defaultdict(list)
        for source, target in self.edges:
            in_degree[target] += 1
            children[source].append(target)
        queue = deque(node.node_id for node in self.nodes if in_degree[node.node_id] == 0)
        layers: list[tuple[MARFTGraphNode, ...]] = []
        while queue:
            current_ids = tuple(sorted(queue, key=order.__getitem__))
            queue.clear()
            layers.append(tuple(by_id[node_id] for node_id in current_ids))
            next_ids: list[str] = []
            for node_id in current_ids:
                for child in children[node_id]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_ids.append(child)
            queue.extend(sorted(next_ids, key=order.__getitem__))
        if sum(len(layer) for layer in layers) != len(self.nodes):
            raise ValueError("MARFT workflow graph must be acyclic.")
        return tuple(layers)


@dataclass
class MARFTWorkflowOrchestra:
    graph: MARFTWorkflowGraph
    role_prompts: dict[str, str]

    def run(
        self,
        *,
        episode_id: str,
        rollout_group: str,
        task: Any,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        environment: Any | None = None,
    ) -> MultiAgentTrajectory:
        del environment
        self.graph.validate()
        team_roles = {agent.name for agent in team.agents}
        graph_roles = {node.role_name for node in self.graph.nodes}
        unknown = sorted(graph_roles - team_roles)
        if unknown:
            raise ValueError(f"MARFT graph references roles absent from TeamSpec: {unknown}.")

        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=str(task.task_id),
            rollout_group=rollout_group,
            team_name=team.name,
            metadata={
                "coordination_protocol": "marft_static_dag",
                "marft_graph_nodes": [node.node_id for node in self.graph.nodes],
                "marft_graph_edges": [list(edge) for edge in self.graph.edges],
            },
        )
        shared_history: list[tuple[str, str]] = []
        turn_id = 0
        final_answer = ""
        layers = self.graph.execution_layers()
        for layer_index, layer in enumerate(layers):
            layer_history = tuple(shared_history)
            pending_messages: list[tuple[str, str]] = []
            for node in layer:
                agent = team.agent(node.role_name)
                prompt = self._build_prompt(
                    agent_name=agent.name,
                    observation=observation,
                    history=layer_history,
                    transition_message=node.transition_message,
                )
                response = policy_backend.generate(
                    PolicyRequest(
                        agent=agent,
                        task_id=str(task.task_id),
                        observation=observation,
                        prompt=prompt,
                        team_context=self._render_history(layer_history),
                        metadata={
                            "stage": "marft_agent",
                            "answer": getattr(task, "answer", None),
                            "marft_node_id": node.node_id,
                            "marft_layer": layer_index,
                        },
                    )
                )
                done = layer_index == len(layers) - 1 and node == layer[-1]
                action_id = f"{episode_id}:{turn_id}:action"
                transition_id = f"{episode_id}:{turn_id}:transition"
                trajectory.add_turn(
                    AgentTurn(
                        episode_id=episode_id,
                        task_id=str(task.task_id),
                        turn_id=turn_id,
                        agent_name=agent.name,
                        role=agent.role,
                        policy_group=agent.policy_group,
                        observation=observation,
                        prompt=prompt,
                        action_text=response.text,
                        action_token_ids=list(response.token_ids),
                        action_logprobs=list(response.logprobs),
                        done=done,
                        joint_action_ids=[action_id],
                        joint_transition_ids=[transition_id],
                        metadata={
                            **response.metadata,
                            "marft_node_id": node.node_id,
                            "marft_layer": layer_index,
                            "marft_role_index": turn_id,
                            "marft_transition_message": node.transition_message or "",
                            "marft_context_snapshot": self._render_history(layer_history),
                            "joint_reward": 0.0,
                            "joint_done": done,
                            "joint_truncated": False,
                            "joint_stop_reason": "",
                        },
                    )
                )
                pending_messages.append((agent.name, response.text))
                final_answer = response.text
                turn_id += 1
            shared_history.extend(pending_messages)

        # Upstream MARFT evaluates the concatenated multi-role completion. This
        # also lets a reviewer/verifier finish without restating a valid solver answer.
        trajectory.final_answer = self._render_history(tuple(shared_history))
        trajectory.metadata["shared_history"] = list(shared_history)
        trajectory.metadata["marft_last_output"] = final_answer
        return trajectory

    def _build_prompt(
        self,
        *,
        agent_name: str,
        observation: str,
        history: tuple[tuple[str, str], ...],
        transition_message: str | None,
    ) -> str:
        system_prompt = str(self.role_prompts.get(agent_name, "")).strip()
        if not system_prompt:
            raise ValueError(f"MARFT role {agent_name!r} requires a non-empty system_prompt.")
        sections = [system_prompt, f"Task:\n{observation}"]
        if transition_message:
            sections.append(f"Transition instruction:\n{transition_message}")
        sections.append(f"Shared conversation:\n{self._render_history(history)}")
        return "\n\n".join(sections)

    @staticmethod
    def _render_history(history: tuple[tuple[str, str], ...]) -> str:
        if not history:
            return "(empty)"
        return "\n".join(f"{agent_name}: {text}" for agent_name, text in history)
