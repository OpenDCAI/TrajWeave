from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core.specs import AgentSpec, TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.credit.comas.scoring import parse_comas_score
from trajweave.orchestration.comas.prompts import (
    render_evaluator_prompt,
    render_scorer_prompt,
    render_solver_prompt,
)


@dataclass
class CoMASPeerReviewOrchestra:
    """CoMAS 原生的 Solver -> Evaluator -> Scorer 固定交互协议。"""

    num_rounds: int = 2
    num_references: int = 2
    task_name: str = "math"
    assignment_seed: int = 0

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
        self._validate(team)
        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=str(task.task_id),
            rollout_group=rollout_group,
            team_name=team.name,
            metadata={
                "comas_task_name": self.task_name,
                "comas_num_rounds": self.num_rounds,
                "comas_num_references": self.num_references,
                "communication_graph": "peer_review",
                "aggregation": "final_round_population_accuracy",
            },
        )
        rng = random.Random(self._episode_seed(episode_id))
        previous_discussions: list[dict[str, Any]] = []
        turn_id = 0

        for round_id in range(self.num_rounds):
            references = self._select_references(previous_discussions, rng)
            rendered_discussion = self._render_discussion(references)
            discussions: list[dict[str, Any]] = []

            for discussion_id, agent in enumerate(team.agents):
                prompt = render_solver_prompt(
                    self.task_name,
                    problem=observation,
                    discussion=rendered_discussion,
                )
                turn = self._generate_turn(
                    trajectory=trajectory,
                    policy_backend=policy_backend,
                    agent=agent,
                    observation=observation,
                    prompt=prompt,
                    role="solver",
                    turn_id=turn_id,
                    round_id=round_id,
                    discussion_id=discussion_id,
                    metadata={
                        "reference_interaction_ids": [item["interaction_id"] for item in references],
                    },
                )
                turn_id += 1
                discussions.append(
                    {
                        "interaction_id": self._interaction_id(episode_id, round_id, discussion_id),
                        "solver": turn,
                    }
                )

            evaluator_agents = list(team.agents)
            rng.shuffle(evaluator_agents)
            for discussion, evaluator in zip(discussions, evaluator_agents, strict=True):
                solver_turn = discussion["solver"]
                prompt = render_evaluator_prompt(
                    self.task_name,
                    problem=observation,
                    solution=solver_turn.action_text,
                )
                discussion["evaluator"] = self._generate_turn(
                    trajectory=trajectory,
                    policy_backend=policy_backend,
                    agent=evaluator,
                    observation=observation,
                    prompt=prompt,
                    role="evaluator",
                    turn_id=turn_id,
                    round_id=round_id,
                    discussion_id=int(solver_turn.metadata["discussion_id"]),
                )
                turn_id += 1

            scorer_agents = list(team.agents)
            rng.shuffle(scorer_agents)
            for discussion, scorer in zip(discussions, scorer_agents, strict=True):
                solver_turn = discussion["solver"]
                evaluator_turn = discussion["evaluator"]
                prompt = render_scorer_prompt(
                    self.task_name,
                    problem=observation,
                    solution=solver_turn.action_text,
                    evaluation=evaluator_turn.action_text,
                )
                scorer_turn = self._generate_turn(
                    trajectory=trajectory,
                    policy_backend=policy_backend,
                    agent=scorer,
                    observation=observation,
                    prompt=prompt,
                    role="scorer",
                    turn_id=turn_id,
                    round_id=round_id,
                    discussion_id=int(solver_turn.metadata["discussion_id"]),
                )
                turn_id += 1
                discussion["scorer"] = scorer_turn
                self._finalize_discussion(discussion)

            previous_discussions = discussions

        final_solutions = [item["solver"].action_text for item in previous_discussions]
        trajectory.final_answer = final_solutions[0] if final_solutions else ""
        trajectory.metadata["comas_final_solutions"] = final_solutions
        trajectory.metadata["comas_interaction_count"] = self.num_rounds * len(team.agents)
        trajectory.metadata["comas_role_counts"] = {
            role: sum(turn.role == role for turn in trajectory.turns) for role in ("solver", "evaluator", "scorer")
        }
        return trajectory

    def _generate_turn(
        self,
        *,
        trajectory: MultiAgentTrajectory,
        policy_backend: PolicyBackend,
        agent: AgentSpec,
        observation: str,
        prompt: str,
        role: str,
        turn_id: int,
        round_id: int,
        discussion_id: int,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTurn:
        interaction_id = self._interaction_id(trajectory.episode_id, round_id, discussion_id)
        response = policy_backend.generate(
            PolicyRequest(
                agent=agent,
                task_id=trajectory.task_id,
                observation=observation,
                prompt=prompt,
                metadata={
                    "comas_stage": role,
                    "round_id": round_id,
                    "discussion_id": discussion_id,
                    "interaction_id": interaction_id,
                },
            )
        )
        turn = AgentTurn(
            episode_id=trajectory.episode_id,
            task_id=trajectory.task_id,
            turn_id=turn_id,
            agent_name=agent.name,
            role=role,
            policy_group=agent.policy_group,
            observation=observation,
            prompt=prompt,
            action_text=response.text,
            action_token_ids=response.token_ids,
            action_logprobs=response.logprobs,
            metadata=response.metadata
            | {
                "comas_stage": role,
                "round_id": round_id,
                "discussion_id": discussion_id,
                "interaction_id": interaction_id,
                **(metadata or {}),
            },
        )
        trajectory.add_turn(turn)
        return turn

    def _finalize_discussion(self, discussion: dict[str, Any]) -> None:
        solver: AgentTurn = discussion["solver"]
        evaluator: AgentTurn = discussion["evaluator"]
        scorer: AgentTurn = discussion["scorer"]
        score = parse_comas_score(scorer.action_text)
        shared = {
            "interaction_id": discussion["interaction_id"],
            "solver_agent_id": solver.agent_name,
            "evaluator_agent_id": evaluator.agent_name,
            "scorer_agent_id": scorer.agent_name,
            "generated_score": score if score is not None else -1,
            "score_valid": score is not None,
        }
        for turn in (solver, evaluator, scorer):
            turn.metadata.update(shared)

    def _select_references(
        self,
        discussions: list[dict[str, Any]],
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        count = min(self.num_references, len(discussions))
        return rng.sample(discussions, count) if count else []

    @staticmethod
    def _render_discussion(discussions: list[dict[str, Any]]) -> str:
        if not discussions:
            return "(Empty discussion)"
        parts: list[str] = []
        for index, discussion in enumerate(discussions, start=1):
            parts.append(
                f"=== Discussion {index} ===\n"
                f"Solution: {discussion['solver'].action_text.strip()}\n"
                f"Evaluation: {discussion['evaluator'].action_text.strip()}"
            )
        return "\n\n".join(parts).strip()

    def _episode_seed(self, episode_id: str) -> int:
        digest = hashlib.sha256(f"{self.assignment_seed}:{episode_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    @staticmethod
    def _interaction_id(episode_id: str, round_id: int, discussion_id: int) -> str:
        return f"{episode_id}:round-{round_id}:discussion-{discussion_id}"

    def _validate(self, team: TeamSpec) -> None:
        if len(team.agents) < 2:
            raise ValueError("CoMAS requires at least two agents.")
        if self.num_rounds <= 0:
            raise ValueError("CoMAS num_rounds must be positive.")
        if self.num_references < 0:
            raise ValueError("CoMAS num_references must be zero or positive.")
        if self.task_name not in {"math", "coding", "science"}:
            raise ValueError(f"Unsupported CoMAS task type: {self.task_name!r}.")
