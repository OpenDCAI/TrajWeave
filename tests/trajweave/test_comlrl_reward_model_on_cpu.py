from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trajweave.backends.verl.workers.scalar_head import (
    JointRewardModel,
    RewardModelWorker,
    bradley_terry_loss,
    serialize_joint_reward_text,
)
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec


class TinyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 4, *, config_field: str = "hidden_size") -> None:
        super().__init__()
        self.config = SimpleNamespace(**{config_field: hidden_size})
        self.embed = nn.Embedding(32, hidden_size)
        self.lm_head = nn.Linear(hidden_size, 32, bias=False)

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache, position_ids=None):
        assert output_hidden_states is True
        assert use_cache is False
        if position_ids is not None:
            assert position_ids.shape == input_ids.shape
        hidden = self.embed(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,))


class PositionAwareTinyCausalLM(TinyCausalLM):
    def __init__(self) -> None:
        super().__init__(hidden_size=2)
        self.position_embed = nn.Embedding(16, 2)

    def forward(self, input_ids, attention_mask, output_hidden_states, use_cache, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        hidden = self.embed(input_ids) + self.position_embed(position_ids)
        return SimpleNamespace(logits=self.lm_head(hidden), hidden_states=(hidden,))


def _team() -> TeamSpec:
    return TeamSpec(
        name="rm-team",
        agents=(
            AgentSpec("alice", "solver", "actor-a"),
            AgentSpec("bob", "reviewer", "actor-b"),
        ),
        policy_groups=(PolicyGroupSpec("actor-a"), PolicyGroupSpec("actor-b")),
        orchestra="joint",
        reward="joint",
        credit="marlhf",
    )


@pytest.mark.parametrize("config_field", ["hidden_size", "n_embd", "d_model"])
def test_hidden_size_inference_preserves_causal_lm_head(config_field: str):
    backbone = TinyCausalLM(hidden_size=7, config_field=config_field)
    original_lm_head = backbone.lm_head

    model = JointRewardModel(backbone)

    assert model.reward_head.in_features == 7
    assert model.backbone.lm_head is original_lm_head


def test_last_non_pad_pooling_handles_left_right_and_sparse_padding():
    backbone = TinyCausalLM(hidden_size=2)
    with torch.no_grad():
        values = torch.arange(32, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
        backbone.embed.weight.copy_(values)
    model = JointRewardModel(backbone)
    with torch.no_grad():
        model.reward_head.weight.fill_(1.0)
        model.reward_head.bias.zero_()

    input_ids = torch.tensor([[4, 7, 0, 0], [0, 0, 4, 7], [4, 0, 7, 0]])
    attention_mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]])

    assert model(input_ids, attention_mask).tolist() == pytest.approx([14.0, 14.0, 14.0])
    with pytest.raises(ValueError, match="all-padding"):
        model(torch.zeros((1, 3), dtype=torch.long), torch.zeros((1, 3), dtype=torch.long))


def test_padding_aware_position_ids_make_left_and_right_padding_equivalent():
    torch.manual_seed(0)
    model = JointRewardModel(PositionAwareTinyCausalLM())
    input_ids = torch.tensor([[4, 7, 0, 0], [0, 0, 4, 7]])
    attention_mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]])

    scores = model(input_ids, attention_mask)

    assert scores[0].item() == pytest.approx(scores[1].item())


def test_non_finite_reward_scores_fail_fast():
    model = JointRewardModel(TinyCausalLM())
    with torch.no_grad():
        model.reward_head.weight.fill_(float("inf"))
    with pytest.raises(FloatingPointError, match="non-finite"):
        model(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))


def test_freeze_backbone_only_trains_scalar_head():
    model = JointRewardModel(TinyCausalLM(), freeze_backbone=True)
    model.train()
    loss = model(torch.tensor([[1, 2], [3, 4]]), torch.ones((2, 2), dtype=torch.long)).sum()
    loss.backward()

    assert model.backbone.training is False
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert model.reward_head.weight.grad is not None


def test_bradley_terry_loss_is_finite_and_decreases():
    chosen = nn.Parameter(torch.tensor([0.0, -0.5]))
    rejected = nn.Parameter(torch.tensor([0.0, 0.5]))
    optimizer = torch.optim.SGD([chosen, rejected], lr=0.25)
    losses = []
    for _ in range(12):
        loss = bradley_terry_loss(chosen, rejected)
        losses.append(float(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert losses[-1] < losses[0]
    assert torch.isfinite(torch.tensor(losses)).all()
    with pytest.raises(FloatingPointError, match="finite"):
        bradley_terry_loss(torch.tensor([float("nan")]), torch.tensor([0.0]))


def test_joint_serializer_uses_team_order_not_mapping_order():
    text = serialize_joint_reward_text(
        _team(),
        {"bob": "prompt-b", "alice": "prompt-a"},
        {"bob": "answer-b", "alice": "answer-a"},
    )

    assert text.index("Agent 1 prompt") < text.index("Agent 2 prompt")
    assert text.index("prompt-a") < text.index("prompt-b")
    assert text.index("answer-a") < text.index("answer-b")


@pytest.mark.parametrize("freeze_backbone", [False, True])
def test_reward_worker_checkpoint_roundtrip_respects_freeze_configuration(tmp_path, freeze_backbone: bool):
    torch.manual_seed(1)
    source = JointRewardModel(TinyCausalLM(), freeze_backbone=freeze_backbone)
    torch.manual_seed(2)
    target = JointRewardModel(TinyCausalLM(), freeze_backbone=freeze_backbone)
    source_worker = RewardModelWorker(source, object(), _team())
    target_worker = RewardModelWorker(target, object(), _team())
    path = tmp_path / "reward-model.pt"

    source_worker.save_checkpoint(path)
    target_worker.load_checkpoint(path)

    assert target.reward_head.state_dict().keys() == source.reward_head.state_dict().keys()
    for name, value in source.reward_head.state_dict().items():
        assert torch.equal(target.reward_head.state_dict()[name], value)
    checkpoint = source_worker.checkpoint_state_dict()
    assert "backbone" in checkpoint
    for name, value in source.backbone.state_dict().items():
        assert torch.equal(target.backbone.state_dict()[name], value)
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    assert torch.equal(source(input_ids, attention_mask), target(input_ids, attention_mask))
