from __future__ import annotations

import re
from dataclasses import dataclass

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.credit.comas import CoMASInteractionCreditAssigner
from trajweave.envs.comas import CoMASMathEnvironment
from trajweave.orchestration.comas import CoMASPeerReviewOrchestra
from trajweave.recipes.doctor_mas.math_smoke import default_math_tasks
from trajweave.rollout.engine import RolloutEngine, RolloutResult


def default_comas_team(
    *,
    agent_ids: tuple[str, ...] = ("agent_0", "agent_1"),
    model_ids: tuple[str, ...] = ("policy_0", "policy_1"),
    num_rounds: int = 2,
) -> TeamSpec:
    if len(agent_ids) < 2:
        raise ValueError("CoMAS requires at least two agents.")
    if len(agent_ids) != len(model_ids):
        raise ValueError("CoMAS agent_ids and model_ids must have the same length.")
    agents = tuple(
        AgentSpec(name=agent_id, role="peer", policy_group=model_id, trainable=True)
        for agent_id, model_id in zip(agent_ids, model_ids, strict=True)
    )
    policy_groups = tuple(
        PolicyGroupSpec(name=model_id, backend="local", trainable=True) for model_id in dict.fromkeys(model_ids)
    )
    return TeamSpec(
        name="comas_peer_review_math",
        agents=agents,
        policy_groups=policy_groups,
        orchestra="comas_peer_review",
        reward="interaction_intrinsic",
        credit="comas_interaction_reward",
        max_turns=num_rounds,
        metadata={
            "control": "fixed_protocol",
            "communication_graph": "peer_review",
            "training_target": "all_agents",
            "aggregation": "final_round_population_accuracy",
            "ground_truth_used_for_training": False,
        },
    )


@dataclass
class CoMASRulePolicyBackend:
    """仅用于 CPU smoke 的可重复后端，不代替 HF/VERL 训练路径。"""

    tokenizer: StableByteTokenizer = StableByteTokenizer()

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        stage = str(request.metadata["comas_stage"])
        if stage == "solver":
            answer = _parse_arithmetic(request.observation)
            text = f"Compute the expression step by step. \\boxed{{{answer if answer is not None else 0}}}"
        elif stage == "evaluator":
            text = "No fatal error is found in the proposed arithmetic solution."
        elif stage == "scorer":
            text = "The proposed solution is consistent with the evaluation.\n<score>3</score>"
        else:
            raise ValueError(f"Unsupported CoMAS stage: {stage!r}.")
        token_ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))


def run_comas_math_smoke(
    *,
    agent_ids: tuple[str, ...] = ("agent_0", "agent_1"),
    model_ids: tuple[str, ...] = ("policy_0", "policy_1"),
    num_rounds: int = 2,
    num_references: int = 2,
    assignment_seed: int = 0,
    rollouts_per_task: int = 1,
) -> tuple[object, RolloutResult]:
    engine = RolloutEngine(
        team=default_comas_team(agent_ids=agent_ids, model_ids=model_ids, num_rounds=num_rounds),
        orchestra=CoMASPeerReviewOrchestra(
            num_rounds=num_rounds,
            num_references=num_references,
            task_name="math",
            assignment_seed=assignment_seed,
        ),
        environment=CoMASMathEnvironment(),
        policy_backend=CoMASRulePolicyBackend(),
        credit_assigner=CoMASInteractionCreditAssigner(),
    )
    result = engine.run(default_math_tasks(), rollouts_per_task=rollouts_per_task)

    class Summary:
        trajectories = len(result.trajectories)
        samples = len(result.samples)
        success_rate = result.success_rate
        dataproto_rows = None
        dataproto_status = "skipped"

    return Summary(), result


def _parse_arithmetic(text: str) -> int | None:
    match = re.search(r"(-?\d+)\s*([+\-*x])\s*(-?\d+)", text)
    if match is None:
        return None
    left, operator, right = int(match.group(1)), match.group(2), int(match.group(3))
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    return left * right
