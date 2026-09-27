from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from trajweave.backends.verl.workers.scalar_head import (
    RewardModelWorker,
    serialize_preference_pair,
)
from trajweave.core.preference import JointPreferencePair
from trajweave.core.specs import TeamSpec


@dataclass(frozen=True)
class JointPreferenceBatch:
    pairs: tuple[JointPreferencePair, ...]
    chosen_texts: tuple[str, ...]
    rejected_texts: tuple[str, ...]

    @classmethod
    def from_pairs(
        cls,
        team: TeamSpec,
        pairs: Sequence[JointPreferencePair],
    ) -> JointPreferenceBatch:
        if not pairs:
            raise ValueError("A joint preference batch cannot be empty.")
        stable_pairs = tuple(pairs)
        return cls(
            pairs=stable_pairs,
            chosen_texts=tuple(serialize_preference_pair(team, pair, chosen=True) for pair in stable_pairs),
            rejected_texts=tuple(serialize_preference_pair(team, pair, chosen=False) for pair in stable_pairs),
        )


class JointRewardModelScorer:
    """Frozen joint-action scorer using rollout-provided prompts verbatim."""

    def __init__(self, worker: RewardModelWorker) -> None:
        if not worker.is_frozen_for_evaluation:
            raise ValueError("JointRewardModelScorer requires a frozen evaluation reward worker.")
        self.worker = worker

    @property
    def is_frozen(self) -> bool:
        return self.worker.is_frozen_for_evaluation and not any(
            parameter.requires_grad for parameter in self.worker.model.parameters()
        )

    def score(
        self,
        prompts_by_agent: Mapping[str, str],
        responses_by_agent: Mapping[str, str],
    ) -> float:
        if not self.is_frozen:
            raise RuntimeError("Joint reward model scorer is no longer frozen.")
        with torch.no_grad():
            scores = self.worker.score_joint_actions([prompts_by_agent], [responses_by_agent])
        if scores.shape != (1,):
            raise RuntimeError(f"Joint reward model must return exactly one score; got shape {tuple(scores.shape)}.")
        return float(scores.item())

    __call__ = score

    def score_many(
        self,
        prompts: Sequence[Mapping[str, str]],
        responses: Sequence[Mapping[str, str]],
    ) -> list[float]:
        if not self.is_frozen:
            raise RuntimeError("Joint reward model scorer is no longer frozen.")
        with torch.no_grad():
            scores = self.worker.score_joint_actions(prompts, responses)
        return [float(value) for value in scores.cpu().tolist()]


__all__ = ["JointPreferenceBatch", "JointRewardModelScorer"]
