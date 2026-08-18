from __future__ import annotations

from dataclasses import dataclass

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.marshal import MARSHALCreditAssigner
from trajweave.envs.strategic import TicTacToeEnvironment, TicTacToeTask
from trajweave.orchestration.marshal import MARSHALSelfPlayOrchestra
from trajweave.rollout.engine import RolloutEngine, RolloutResult

_MOVE_PREFERENCES = {
    "player0_win": {
        0: (0, 1, 2, 3, 4, 5, 6, 7, 8),
        1: (8, 7, 6, 5, 4, 3, 2, 1, 0),
    },
    "player1_win": {
        0: (0, 1, 8, 2, 6, 5, 7, 3, 4),
        1: (4, 3, 5, 2, 6, 7, 8, 1, 0),
    },
}


@dataclass(frozen=True)
class MARSHALSmokeSummary:
    trajectories: int
    samples: int
    success_rate: float
    player_0_wins: int
    player_1_wins: int
    draws: int


@dataclass
class RuleBasedMARSHALPolicyBackend:
    tokenizer: StableByteTokenizer = StableByteTokenizer()

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        if request.metadata.get("stage") != "marshal_action":
            raise ValueError("RuleBasedMARSHALPolicyBackend only handles MARSHAL action turns.")
        strategy = str(request.metadata.get("strategy", "player0_win"))
        if strategy not in _MOVE_PREFERENCES:
            raise ValueError(f"Unknown MARSHAL smoke strategy: {strategy!r}.")
        player = int(request.metadata["marshal_player_id"])
        legal_actions = {int(value) for value in request.metadata["legal_actions"]}
        action = next(value for value in _MOVE_PREFERENCES[strategy][player] if value in legal_actions)
        text = f"<answer>{action}</answer>"
        token_ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))


def default_marshal_team(
    *,
    player_ids: tuple[str, str] = ("player_0", "player_1"),
    shared_model_id: str = "shared_policy",
    max_actions: int = 9,
) -> TeamSpec:
    if len(player_ids) != 2 or len(set(player_ids)) != 2:
        raise ValueError("MARSHAL requires two distinct player IDs.")
    agents = tuple(
        AgentSpec(name=name, role="strategic_player", policy_group=shared_model_id, trainable=True)
        for name in player_ids
    )
    return TeamSpec(
        name="marshal_tictactoe_selfplay",
        agents=agents,
        policy_groups=(PolicyGroupSpec(name=shared_model_id, backend="local", trainable=True),),
        orchestra="marshal_self_play",
        reward="zero_sum_game_payoff",
        credit="marshal_turn_level_reinforce",
        max_turns=max_actions,
        metadata={
            "control": "alternating_self_play",
            "communication_graph": "game_state",
            "training_target": "shared_policy_both_players",
            "per_player_subtrajectories": True,
        },
    )


def default_marshal_tasks() -> list[TicTacToeTask]:
    return [
        TicTacToeTask(task_id="marshal_tictactoe_player0_win", strategy="player0_win"),
        TicTacToeTask(task_id="marshal_tictactoe_player1_win", strategy="player1_win"),
    ]


def run_marshal_smoke(
    *,
    rollouts_per_task: int = 1,
    player_ids: tuple[str, str] = ("player_0", "player_1"),
    shared_model_id: str = "shared_policy",
    max_actions: int = 9,
    format_reward: float = 0.05,
    allow_bare_actions: bool = False,
    gamma: float = 1.0,
    reward_normalization: str = "mean",
    advantage_normalization: str = "mean",
    whiten_rewards: bool = True,
    whiten_advantages: bool = True,
) -> tuple[MARSHALSmokeSummary, RolloutResult]:
    engine = RolloutEngine(
        team=default_marshal_team(
            player_ids=player_ids,
            shared_model_id=shared_model_id,
            max_actions=max_actions,
        ),
        orchestra=MARSHALSelfPlayOrchestra(
            player_ids=player_ids,
            format_reward=format_reward,
            max_actions=max_actions,
            allow_bare_actions=allow_bare_actions,
        ),
        environment=TicTacToeEnvironment(),
        policy_backend=RuleBasedMARSHALPolicyBackend(),
        credit_assigner=MARSHALCreditAssigner(
            gamma=gamma,
            reward_normalization=reward_normalization,
            advantage_normalization=advantage_normalization,
            whiten_rewards=whiten_rewards,
            whiten_advantages=whiten_advantages,
        ),
    )
    result = engine.run(default_marshal_tasks(), rollouts_per_task=rollouts_per_task)
    winners = [trajectory.metadata.get("marshal_winner") for trajectory in result.trajectories]
    return (
        MARSHALSmokeSummary(
            trajectories=len(result.trajectories),
            samples=len(result.samples),
            success_rate=result.success_rate,
            player_0_wins=sum(winner == 0 for winner in winners),
            player_1_wins=sum(winner == 1 for winner in winners),
            draws=sum(winner is None for winner in winners),
        ),
        result,
    )
