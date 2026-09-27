from __future__ import annotations

from dataclasses import dataclass

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.search import SearchAnswerEnvironment, SearchTask
from trajweave.orchestration.base import TeamContext, is_approved_response


@dataclass
class SearchAnswerOrchestra:
    verifier_name: str = "verifier"
    searcher_name: str = "searcher"
    answer_name: str = "answer"
    approval_keyword: str = "APPROVED"

    def run(
        self,
        *,
        episode_id: str,
        rollout_group: str,
        task: SearchTask,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        environment: SearchAnswerEnvironment | None = None,
    ) -> MultiAgentTrajectory:
        if environment is None or not hasattr(environment, "search"):
            raise ValueError("SearchAnswerOrchestra requires an environment with a search() method.")

        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            rollout_group=rollout_group,
            team_name=team.name,
        )
        context = TeamContext()
        turn_id = 0

        for loop_index in range(team.max_turns):
            verifier = team.agent(self.verifier_name)
            verifier_prompt = self._build_prompt(verifier.role, observation, context)
            verifier_response = policy_backend.generate(
                PolicyRequest(
                    agent=verifier,
                    task_id=task.task_id,
                    observation=observation,
                    prompt=verifier_prompt,
                    team_context=context.render(),
                    metadata={"answer": task.answer, "loop_index": loop_index},
                )
            )
            context.append(verifier.name, verifier_response.text)
            approved = is_approved_response(verifier_response.text, self.approval_keyword)
            trajectory.add_turn(
                AgentTurn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=verifier.name,
                    role=verifier.role,
                    policy_group=verifier.policy_group,
                    observation=observation,
                    prompt=verifier_prompt,
                    action_text=verifier_response.text,
                    action_token_ids=verifier_response.token_ids,
                    action_logprobs=verifier_response.logprobs,
                    metadata=verifier_response.metadata | {"approved": approved, "loop_index": loop_index},
                )
            )
            turn_id += 1
            if approved:
                break

            searcher = team.agent(self.searcher_name)
            searcher_prompt = self._build_prompt(searcher.role, observation, context)
            searcher_response = policy_backend.generate(
                PolicyRequest(
                    agent=searcher,
                    task_id=task.task_id,
                    observation=observation,
                    prompt=searcher_prompt,
                    team_context=context.render(),
                    metadata={"answer": task.answer, "search_query": task.search_query, "loop_index": loop_index},
                )
            )
            tool_result = environment.search(task, searcher_response.text)
            context.append(searcher.name, searcher_response.text)
            context.append("search_tool", tool_result)
            trajectory.add_turn(
                AgentTurn(
                    episode_id=episode_id,
                    task_id=task.task_id,
                    turn_id=turn_id,
                    agent_name=searcher.name,
                    role=searcher.role,
                    policy_group=searcher.policy_group,
                    observation=observation,
                    prompt=searcher_prompt,
                    action_text=searcher_response.text,
                    action_token_ids=searcher_response.token_ids,
                    action_logprobs=searcher_response.logprobs,
                    metadata=searcher_response.metadata
                    | {
                        "loop_index": loop_index,
                        "tool_name": "search",
                        "tool_result": tool_result,
                    },
                )
            )
            turn_id += 1

        answer = team.agent(self.answer_name)
        answer_prompt = self._build_prompt(answer.role, observation, context)
        answer_response = policy_backend.generate(
            PolicyRequest(
                agent=answer,
                task_id=task.task_id,
                observation=observation,
                prompt=answer_prompt,
                team_context=context.render(),
                metadata={"answer": task.answer},
            )
        )
        context.append(answer.name, answer_response.text)
        trajectory.add_turn(
            AgentTurn(
                episode_id=episode_id,
                task_id=task.task_id,
                turn_id=turn_id,
                agent_name=answer.name,
                role=answer.role,
                policy_group=answer.policy_group,
                observation=observation,
                prompt=answer_prompt,
                action_text=answer_response.text,
                action_token_ids=answer_response.token_ids,
                action_logprobs=answer_response.logprobs,
                done=True,
                metadata=answer_response.metadata,
            )
        )
        trajectory.final_answer = answer_response.text
        trajectory.metadata["team_context"] = context.render()
        return trajectory

    def _build_prompt(self, role: str, observation: str, context: TeamContext) -> str:
        rendered = context.render()
        if role.lower() == "verifier":
            return (
                f"Question:\n{observation}\n\nTeam context:\n{rendered}\n\n"
                "Decide whether evidence is enough. Return APPROVED or SEARCH."
            )
        if role.lower() == "searcher":
            return f"Question:\n{observation}\n\nTeam context:\n{rendered}\n\nWrite one search query."
        if role.lower() == "answer":
            return f"Question:\n{observation}\n\nTeam context:\n{rendered}\n\nAnswer with 'Final answer: <answer>'."
        return f"Question:\n{observation}\n\nTeam context:\n{rendered}"
