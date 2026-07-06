from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from trajweave.backends.local import extract_final_int
from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.math import MathTask
from trajweave.orchestration.base import TeamContext


@dataclass
class MAPoRLDebateOrchestra:
    consensus_threshold: int = 2
    early_stop: bool = True

    def run(
        self,
        *,
        episode_id: str,
        rollout_group: str,
        task: MathTask,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        environment: object | None = None,
    ) -> MultiAgentTrajectory:
        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            rollout_group=rollout_group,
            team_name=team.name,
        )
        context = TeamContext()
        final_answer = ""
        turn_id = 0
        consensus_reached = False
        consensus_answer: int | None = None

        for round_id in range(team.max_turns):
            round_answers: list[int | None] = []
            for agent in team.agents:
                prompt = self._build_prompt(agent.name, observation, context, round_id)
                response = policy_backend.generate(
                    PolicyRequest(
                        agent=agent,
                        task_id=task.task_id,
                        observation=observation,
                        prompt=prompt,
                        team_context=context.render(),
                        metadata={"answer": task.answer, "round_id": round_id},
                    )
                )
                parsed_answer = extract_final_int(response.text)
                round_answers.append(parsed_answer)
                final_answer = response.text
                context.append(agent.name, response.text)
                trajectory.add_turn(
                    AgentTurn(
                        episode_id=episode_id,
                        task_id=task.task_id,
                        turn_id=turn_id,
                        agent_name=agent.name,
                        role=agent.role,
                        policy_group=agent.policy_group,
                        observation=observation,
                        prompt=prompt,
                        action_text=response.text,
                        action_token_ids=response.token_ids,
                        action_logprobs=response.logprobs,
                        metadata=response.metadata
                        | {
                            "round_id": round_id,
                            "agent_answer": parsed_answer,
                            "communication_graph": "fully_connected",
                            "aggregation": "consensus",
                        },
                    )
                )
                turn_id += 1

            consensus_answer = self._consensus_answer(round_answers)
            consensus_reached = consensus_answer is not None
            if consensus_reached:
                final_answer = f"Final answer: {consensus_answer}"
                for turn in trajectory.turns:
                    if turn.metadata.get("round_id") == round_id:
                        turn.metadata["consensus_reached"] = True
                        turn.metadata["consensus_answer"] = consensus_answer
                if self.early_stop:
                    break

        trajectory.final_answer = final_answer
        trajectory.metadata["team_context"] = context.render()
        trajectory.metadata["consensus_reached"] = consensus_reached
        if consensus_answer is not None:
            trajectory.metadata["consensus_answer"] = consensus_answer
        return trajectory

    def _build_prompt(self, agent_name: str, observation: str, context: TeamContext, round_id: int) -> str:
        rendered = context.render()
        if not rendered:
            rendered = "No previous agent messages."
        return (
            f"Task:\n{observation}\n\n"
            f"Round: {round_id}\n"
            f"Agent: {agent_name}\n\n"
            f"Other agent messages:\n{rendered}\n\n"
            "Solve the task. Return exactly one line with 'Final answer: <number>'."
        )

    def _consensus_answer(self, answers: list[int | None]) -> int | None:
        filtered = [answer for answer in answers if answer is not None]
        if not filtered:
            return None
        answer, count = Counter(filtered).most_common(1)[0]
        return answer if count >= self.consensus_threshold else None
