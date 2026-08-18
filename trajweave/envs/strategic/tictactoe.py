from __future__ import annotations

import re
from dataclasses import dataclass

from trajweave.core.trajectory import MultiAgentTrajectory

_WINNING_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)
_ACTION_RE = re.compile(r"\s*(?:<think>.*?</think>\s*)?<answer>\s*([0-8])\s*</answer>\s*", re.DOTALL)
_BARE_ACTION_RE = re.compile(r"\s*([0-8])\s*")


@dataclass(frozen=True)
class TicTacToeTask:
    task_id: str
    strategy: str = "player0_win"
    starting_player: int = 0


@dataclass(frozen=True)
class TicTacToeState:
    board: tuple[int, ...] = (-1,) * 9
    current_player: int = 0
    done: bool = False
    winner: int | None = None
    invalid_action: bool = False


@dataclass(frozen=True)
class TicTacToeStep:
    state: TicTacToeState
    rewards: tuple[float, float]
    format_valid: bool


@dataclass(frozen=True)
class TicTacToeAction:
    action: int | None
    format_valid: bool


@dataclass(frozen=True)
class TicTacToeEnvironment:
    name: str = "marshal_tictactoe"

    def initial_state(self, task: TicTacToeTask) -> TicTacToeState:
        if task.starting_player not in {0, 1}:
            raise ValueError("TicTacToe starting_player must be 0 or 1.")
        return TicTacToeState(current_player=task.starting_player)

    def initial_observation(self, task: TicTacToeTask) -> str:
        return self.render(self.initial_state(task))

    def legal_actions(self, state: TicTacToeState) -> tuple[int, ...]:
        if state.done:
            return ()
        return tuple(index for index, value in enumerate(state.board) if value < 0)

    def parse_action(self, text: str) -> int | None:
        match = _ACTION_RE.fullmatch(text)
        return int(match.group(1)) if match is not None else None

    def parse_response(self, text: str, *, allow_bare_action: bool = False) -> TicTacToeAction:
        action = self.parse_action(text)
        if action is not None:
            return TicTacToeAction(action=action, format_valid=True)
        if allow_bare_action:
            match = _BARE_ACTION_RE.fullmatch(text)
            if match is not None:
                return TicTacToeAction(action=int(match.group(1)), format_valid=False)
        return TicTacToeAction(action=None, format_valid=False)

    def step(
        self,
        state: TicTacToeState,
        action: int | None,
        *,
        format_valid: bool = True,
    ) -> TicTacToeStep:
        if state.done:
            raise ValueError("Cannot step a terminal TicTacToe state.")
        player = state.current_player
        opponent = 1 - player
        if action is None or action not in self.legal_actions(state):
            rewards = [-1.0, -1.0]
            rewards[opponent] = 1.0
            return TicTacToeStep(
                state=TicTacToeState(
                    board=state.board,
                    current_player=opponent,
                    done=True,
                    winner=opponent,
                    invalid_action=True,
                ),
                rewards=(rewards[0], rewards[1]),
                format_valid=False,
            )

        board = list(state.board)
        board[action] = player
        board_tuple = tuple(board)
        winner = player if self._is_winner(board_tuple, player) else None
        done = winner is not None or all(value >= 0 for value in board_tuple)
        rewards = [0.0, 0.0]
        if winner is not None:
            rewards[winner] = 1.0
            rewards[1 - winner] = -1.0
        return TicTacToeStep(
            state=TicTacToeState(
                board=board_tuple,
                current_player=opponent,
                done=done,
                winner=winner,
            ),
            rewards=(rewards[0], rewards[1]),
            format_valid=format_valid,
        )

    def render(self, state: TicTacToeState) -> str:
        symbols = {0: "X", 1: "O", -1: "."}
        rows = [" ".join(symbols[state.board[row * 3 + column]] for column in range(3)) for row in range(3)]
        return "\n".join(rows)

    def evaluate(self, task: TicTacToeTask, final_answer: str) -> tuple[float, bool]:
        del task
        completed = final_answer.startswith("winner:") or final_answer == "draw"
        return float(completed), completed

    def evaluate_trajectory(
        self,
        task: TicTacToeTask,
        trajectory: MultiAgentTrajectory,
    ) -> tuple[float, bool]:
        del task
        completed = bool(trajectory.metadata.get("marshal_completed", False))
        invalid = bool(trajectory.metadata.get("marshal_invalid_action", False))
        return float(completed and not invalid), bool(completed and not invalid)

    @staticmethod
    def _is_winner(board: tuple[int, ...], player: int) -> bool:
        return any(all(board[index] == player for index in line) for line in _WINNING_LINES)
