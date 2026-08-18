from __future__ import annotations

from dataclasses import dataclass

from trajweave.backends.policy import PolicyBackend, PolicyRequest
from trajweave.core.specs import TeamSpec
from trajweave.core.trajectory import AgentTurn, MultiAgentTrajectory
from trajweave.envs.strategic import TicTacToeEnvironment, TicTacToeTask


@dataclass
class MARSHALSelfPlayOrchestra:
    player_ids: tuple[str, str] = ("player_0", "player_1")
    format_reward: float = 0.05
    max_actions: int = 9
    allow_bare_actions: bool = False

    def run(
        self,
        *,
        episode_id: str,
        rollout_group: str,
        task: TicTacToeTask,
        team: TeamSpec,
        observation: str,
        policy_backend: PolicyBackend,
        environment: TicTacToeEnvironment | None = None,
    ) -> MultiAgentTrajectory:
        del observation
        game = environment or TicTacToeEnvironment()
        if len(self.player_ids) != 2 or {agent.name for agent in team.agents} != set(self.player_ids):
            raise ValueError("MARSHAL self-play requires exactly the configured two players.")
        if self.max_actions < 1 or self.max_actions > 9:
            raise ValueError("MARSHAL max_actions must be in [1, 9].")

        trajectory = MultiAgentTrajectory(
            episode_id=episode_id,
            task_id=task.task_id,
            rollout_group=rollout_group,
            team_name=team.name,
        )
        state = game.initial_state(task)
        history: list[str] = []
        player_turn_counts = [0, 0]
        latest_turn_by_player: dict[int, AgentTurn] = {}

        for turn_id in range(self.max_actions):
            player_id = state.current_player
            agent_name = self.player_ids[player_id]
            agent = team.agent(agent_name)
            legal_actions = game.legal_actions(state)
            board_before = game.render(state)
            prompt = self._prompt(
                board=board_before,
                player_id=player_id,
                legal_actions=legal_actions,
                history=history,
            )
            response = policy_backend.generate(
                PolicyRequest(
                    agent=agent,
                    task_id=task.task_id,
                    observation=board_before,
                    prompt=prompt,
                    team_context="\n".join(history),
                    metadata={
                        "stage": "marshal_action",
                        "marshal_player_id": player_id,
                        "marshal_player_turn": player_turn_counts[player_id],
                        "legal_actions": legal_actions,
                        "strategy": task.strategy,
                    },
                )
            )
            parsed_action = game.parse_response(response.text, allow_bare_action=self.allow_bare_actions)
            action = parsed_action.action
            action_valid = action is not None and action in legal_actions
            transition = game.step(state, action, format_valid=parsed_action.format_valid)
            turn_reward = self.format_reward if transition.format_valid else 0.0
            if transition.state.done:
                turn_reward += transition.rewards[player_id]

            turn = AgentTurn(
                episode_id=episode_id,
                task_id=task.task_id,
                turn_id=turn_id,
                agent_name=agent.name,
                role=agent.role,
                policy_group=agent.policy_group,
                observation=board_before,
                prompt=prompt,
                action_text=response.text,
                action_token_ids=response.token_ids,
                action_logprobs=response.logprobs,
                anchor_observation={
                    "board": state.board,
                    "player_id": player_id,
                    "legal_actions": legal_actions,
                },
                next_observation={
                    "board": transition.state.board,
                    "next_player": transition.state.current_player,
                    "done": transition.state.done,
                },
                step_reward=float(turn_reward),
                reward=float(turn_reward),
                done=transition.state.done,
                metadata=response.metadata
                | {
                    "marshal_episode_id": episode_id,
                    "marshal_player_id": player_id,
                    "marshal_player_turn": player_turn_counts[player_id],
                    "marshal_action": -1 if action is None else action,
                    "marshal_action_valid": action_valid,
                    "marshal_legal_actions": list(legal_actions),
                    "marshal_format_valid": transition.format_valid,
                    "marshal_turn_reward": float(turn_reward),
                    "marshal_terminal": transition.state.done,
                    "marshal_shared_policy": True,
                    "active_mask": 1.0,
                },
            )
            trajectory.add_turn(turn)
            latest_turn_by_player[player_id] = turn
            player_turn_counts[player_id] += 1
            history.append(
                f"Turn {turn_id + 1}: player_{player_id} returned {response.text!r}; "
                f"action={action if action_valid else 'invalid'}."
            )

            if transition.state.done:
                opponent = 1 - player_id
                opponent_turn = latest_turn_by_player.get(opponent)
                if opponent_turn is not None:
                    opponent_reward = float(opponent_turn.step_reward or 0.0) + transition.rewards[opponent]
                    opponent_turn.step_reward = opponent_reward
                    opponent_turn.reward = opponent_reward
                    opponent_turn.metadata["marshal_turn_reward"] = opponent_reward
                    opponent_turn.metadata["marshal_terminal_payoff"] = transition.rewards[opponent]
                turn.metadata["marshal_terminal_payoff"] = transition.rewards[player_id]
                state = transition.state
                break
            state = transition.state

        completed = state.done
        payoffs = [0.0, 0.0]
        if completed and state.winner is not None:
            payoffs[state.winner] = 1.0
            payoffs[1 - state.winner] = -1.0
        if not completed:
            history.append(f"Episode truncated after {self.max_actions} actions.")

        trajectory.final_answer = "draw" if state.winner is None else f"winner: player_{state.winner}"
        trajectory.metadata.update(
            {
                "marshal_game": "tictactoe",
                "marshal_completed": completed,
                "marshal_invalid_action": state.invalid_action,
                "marshal_winner": state.winner,
                "marshal_payoffs": payoffs,
                "marshal_action_history": history,
                "marshal_player_turn_counts": player_turn_counts,
                "marshal_shared_policy": True,
                "marshal_per_player_subtrajectories": {
                    self.player_ids[player]: f"{episode_id}:p{player}" for player in range(2)
                },
            }
        )
        return trajectory

    @staticmethod
    def _prompt(
        *,
        board: str,
        player_id: int,
        legal_actions: tuple[int, ...],
        history: list[str],
    ) -> str:
        visible_history = "\n".join(history) if history else "No actions have been played."
        answer_options = ", ".join(f"<answer>{value}</answer>" for value in legal_actions)
        return (
            "You are playing Tic-Tac-Toe through self-play. Cells are numbered 0 through 8 row-wise.\n"
            f"You are player_{player_id} ({'X' if player_id == 0 else 'O'}).\n"
            f"Board:\n{board}\n"
            f"Legal responses: {answer_options}\n"
            f"Game history:\n{visible_history}\n"
            "Copy exactly one complete legal response from the list above and output nothing else. "
            "Do not output placeholders such as N, row, or column."
        )
