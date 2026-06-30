from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from trajweave.core.trajectory import TrainingSample
from verl.protocol import DataProto

DRMAS_AGENT_IDS = {
    "solver": "Solver Agent",
    "verifier": "Verifier Agent",
    "searcher": "Search Agent",
    "search": "Search Agent",
    "answer": "Answer Agent",
}


def _pad(sequences: list[list[int]], pad_value: int = 0) -> torch.Tensor:
    width = max(len(item) for item in sequences) if sequences else 1
    tensor = torch.full((len(sequences), width), pad_value, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long)
        tensor[row, : len(sequence)] = values
    return tensor


@dataclass
class VerlDataProtoAdapter:
    pad_token_id: int = 0

    def build(self, samples: list[TrainingSample]) -> DataProto:
        if not samples:
            raise ValueError("Cannot build DataProto from empty samples.")
        prompt_ids = [[ord(ch) % 255 + 1 for ch in sample.prompt] or [1] for sample in samples]
        response_ids = [sample.response_token_ids or [ord(ch) % 255 + 1 for ch in sample.response] or [1] for sample in samples]
        prompts = _pad(prompt_ids, self.pad_token_id)
        responses = _pad(response_ids, self.pad_token_id)
        input_ids = torch.cat([prompts, responses], dim=1)
        attention_mask = torch.cat([(prompts != self.pad_token_id).long(), (responses != self.pad_token_id).long()], dim=1)
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0).repeat(input_ids.shape[0], 1)
        response_mask = (responses != self.pad_token_id).long()
        token_level_rewards = torch.zeros_like(responses, dtype=torch.float32)
        advantages = torch.zeros_like(responses, dtype=torch.float32)

        for row, sample in enumerate(samples):
            valid = int(response_mask[row].sum().item())
            last = max(valid - 1, 0)
            token_level_rewards[row, last] = float(sample.reward)
            if sample.advantage is not None:
                advantages[row, :valid] = float(sample.advantage)

        non_tensors = {
            "uid": np.array([sample.rollout_group for sample in samples], dtype=object),
            "traj_uid": np.array([sample.episode_id for sample in samples], dtype=object),
            "sample_id": np.array([sample.sample_id for sample in samples], dtype=object),
            "task_id": np.array([sample.task_id for sample in samples], dtype=object),
            "agent_name": np.array([sample.agent_name for sample in samples], dtype=object),
            "agent_id": np.array(
                [DRMAS_AGENT_IDS.get(sample.agent_name, sample.agent_name) for sample in samples],
                dtype=object,
            ),
            "role": np.array([sample.role for sample in samples], dtype=object),
            "policy_group": np.array([sample.policy_group for sample in samples], dtype=object),
            "turn_id": np.array([sample.turn_id for sample in samples], dtype=object),
        }
        tensors = {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "token_level_rewards": token_level_rewards,
            "advantages": advantages,
            "returns": advantages.clone(),
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info={"source": "trajweave"})
