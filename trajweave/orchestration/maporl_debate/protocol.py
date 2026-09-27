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
    reward_feedback: bool = False
    criteria_for_consensus_percentage: float | None = None
    criteria_for_consensus_reward_threshold: float = 0.7
    dataset_name: str = "GSM8k"

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
        finished_round = -1
        agent_contexts: dict[str, list[dict[str, str]]] = {
            agent.name: [{"role": "user", "content": self._format_question(agent.name, observation)}]
            for agent in team.agents
        }

        for round_id in range(team.max_turns):
            round_answers: list[int | None] = []
            round_scores: list[float] = []
            for agent in team.agents:
                agent_index = tuple(item.name for item in team.agents).index(agent.name)
                if round_id > 0:
                    agent_contexts[agent.name].append(
                        {
                            "role": "user",
                            "content": self._build_peer_feedback(
                                agent.name, observation, agent_contexts, team, round_id
                            ),
                        }
                    )
                prompt = self._render_messages(agent_contexts[agent.name])
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
                raw_score = 1.0 if parsed_answer == task.answer else 0.0
                round_answers.append(parsed_answer)
                round_scores.append(raw_score)
                final_answer = response.text
                context.append(agent.name, response.text)
                agent_contexts[agent.name].append({"role": "assistant", "content": response.text})
                if self.reward_feedback:
                    agent_contexts[agent.name].append(
                        {
                            "role": "user",
                            "content": self._reward_feedback_message(raw_score),
                        }
                    )
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
                            "agent_index": agent_index,
                            "agent_answer": parsed_answer,
                            "raw_score": raw_score,
                            "correctness": raw_score,
                            "communication_graph": "fully_connected",
                            "aggregation": "consensus",
                            "reward_feedback": self.reward_feedback,
                            "policy_separation": team.metadata.get("policy_separation", False),
                            "collaboration_separation": team.metadata.get("collaboration_separation", False),
                            "task_training": team.metadata.get("task_training", False),
                        },
                    )
                )
                turn_id += 1

            consensus_answer = self._consensus_answer(round_answers, round_scores)
            consensus_reached = consensus_answer is not None
            finished_round = round_id if consensus_reached else -1
            if consensus_reached:
                final_answer = f"Final answer: {consensus_answer}"
                for turn in trajectory.turns:
                    if turn.metadata.get("round_id") == round_id:
                        turn.metadata["consensus_reached"] = True
                        turn.metadata["consensus_answer"] = consensus_answer
                        turn.metadata["finished_round"] = finished_round
                if self.early_stop:
                    break

        trajectory.final_answer = final_answer
        trajectory.metadata["team_context"] = context.render()
        trajectory.metadata["agent_contexts"] = agent_contexts
        trajectory.metadata["consensus_reached"] = consensus_reached
        trajectory.metadata["finished_round"] = finished_round
        trajectory.metadata["round_num"] = team.max_turns
        trajectory.metadata["maporl_protocol"] = {
            "reward_feedback": self.reward_feedback,
            "criteria_for_consensus_percentage": self._consensus_percentage_threshold(len(team.agents)),
            "criteria_for_consensus_reward_threshold": self.criteria_for_consensus_reward_threshold,
        }
        if consensus_answer is not None:
            trajectory.metadata["consensus_answer"] = consensus_answer
        for turn in trajectory.turns:
            turn.metadata["finished_round"] = finished_round
        return trajectory

    def _format_question(self, agent_name: str, observation: str) -> str:
        del agent_name
        return (
            f"Question: {observation}\n\n"
            "Provide a short but precise reasoning for your solution. "
            "At the end, you MUST write the answer in the following format:\n\n"
            "Final answer: <number>"
        )

    def _build_peer_feedback(
        self,
        agent_name: str,
        observation: str,
        agent_contexts: dict[str, list[dict[str, str]]],
        team: TeamSpec,
        round_id: int,
    ) -> str:
        others = [agent for agent in team.agents if agent.name != agent_name]
        if not others:
            return self._format_question(agent_name, observation)

        parts = ["These are the solutions to the problem from other agents:"]
        for idx, other in enumerate(others, start=1):
            messages = agent_contexts[other.name]
            assistant_messages = [message["content"] for message in messages if message["role"] == "assistant"]
            response = assistant_messages[round_id - 1] if len(assistant_messages) >= round_id else ""
            parts.append(f"Agent {idx} solution: ```{response}```")
            if self.reward_feedback:
                rewards = [
                    message["content"]
                    for message in messages
                    if message["role"] == "user" and message["content"].startswith("Reward from a verifier")
                ]
                if len(rewards) >= round_id:
                    parts.append(rewards[round_id - 1].replace("your", f"agent {idx}'s"))

        parts.append(
            "Focus on providing a well-reasoned response that considers your own previous solution "
            "and the answers from other agents. If your previous answer was wrong, revise it. "
            f"Once again, the question is: {observation}"
        )
        return "\n\n".join(parts)

    def _reward_feedback_message(self, score: float) -> str:
        if score < 0.3:
            feedback = "Your answer is highly likely wrong."
        elif score < 0.6:
            feedback = "Your answer might be wrong, or your reasoning needs a stronger argument."
        elif score < 0.8:
            feedback = "Your answer seems right, but check your reasoning again."
        else:
            feedback = "Your answer is likely right with high probability."
        return f"Reward from a verifier of your answer: {score:.3f} out of 1.0, which means {feedback}"

    def _render_messages(self, messages: list[dict[str, str]]) -> str:
        return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages)

    def _consensus_answer(self, answers: list[int | None], rewards: list[float]) -> int | None:
        filtered = [answer for answer in answers if answer is not None]
        if not filtered:
            return None
        answer, count = Counter(filtered).most_common(1)[0]
        majority_percentage = count / len(answers)
        threshold = self._consensus_percentage_threshold(len(answers))
        if majority_percentage < threshold:
            return None
        majority_rewards = [reward for item, reward in zip(answers, rewards, strict=True) if item == answer]
        avg_reward = sum(majority_rewards) / len(majority_rewards) if majority_rewards else 0.0
        return answer if avg_reward > self.criteria_for_consensus_reward_threshold else None

    def _consensus_percentage_threshold(self, agent_count: int) -> float:
        if self.criteria_for_consensus_percentage is not None:
            return self.criteria_for_consensus_percentage
        return (self.consensus_threshold - 1e-9) / max(agent_count, 1)
