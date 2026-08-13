from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from trajweave.backends.policy import StableByteTokenizer
from trajweave.backends.verl.emitters.comlrl import CoMLRLEmitterMixin

_SMOKE_RUNTIME_ALGORITHM = {
    "marlhf": "magrpo",
    "madpo_iter": "madpo",
    "marlhf_iter": "madpo",
}


class _SmokeWorker(CoMLRLEmitterMixin):
    def __init__(self, algorithm: str, *, num_candidates: int) -> None:
        runtime_algorithm = _SMOKE_RUNTIME_ALGORITHM.get(algorithm, algorithm)
        self.config = {
            "trajweave": {
                "comlrl": {
                    "algorithm": runtime_algorithm,
                    "joint_mode": "aligned",
                    "max_turns": 1,
                    "normalize_advantages": False,
                }
            },
            "agent": {
                "agent_ids": ["agent_0", "agent_1"],
                "model_ids": ["policy_0", "policy_1"],
            },
        }
        if runtime_algorithm in {"iac", "maac"}:
            topology = "independent" if runtime_algorithm == "iac" else "centralized"
            self.config["trajweave"]["comlrl"]["actor_critic"] = {
                "topology": topology,
                "critic_type": "v",
            }
        self.tokenizer = StableByteTokenizer()
        self.model_config = SimpleNamespace(local_path="/smoke/model")
        self.num_candidates = num_candidates

    def _encode_prompt(self, value: Any) -> list[int]:
        return self.tokenizer.encode(str(value))

    def _encode_prompt_text(self, value: str) -> list[int]:
        return self.tokenizer.encode(value)

    def _encode_text(self, value: str) -> list[int]:
        return self.tokenizer.encode(value)

    @staticmethod
    def _local_policy_version() -> int:
        return 0

    @staticmethod
    def _worker_group_model_path(group_id: str) -> str:
        return f"/smoke/{group_id}"


def run_comlrl_joint_math_smoke(algorithm: str) -> dict[str, Any]:
    normalized = str(algorithm).strip().lower()
    candidates = 1 if normalized in {"iac", "maac"} else 2
    worker = _SmokeWorker(normalized, num_candidates=candidates)
    outputs = worker._build_comlrl_joint_math_outputs(
        {
            "uid": f"smoke-{normalized}",
            "raw_prompt": [{"role": "user", "content": "What is 1 + 1?"}],
            "reward_model": {"ground_truth": "2"},
            "__comlrl_num_candidates__": candidates,
        }
    )
    active_preferences = sum(float(output.extra_fields.get("preference_loss_mask", 0.0)) for output in outputs)
    return {
        "algorithm": normalized,
        "runtime_algorithm": _SMOKE_RUNTIME_ALGORITHM.get(normalized, normalized),
        "trajectories": 1,
        "samples": len(outputs),
        "worker_groups": sorted({output.extra_fields["worker_group"] for output in outputs}),
        "preference_rows": int(active_preferences),
        "reward_scores": [float(output.reward_score or 0.0) for output in outputs],
    }


__all__ = ["run_comlrl_joint_math_smoke"]
