from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from trajweave.core.preference import JointPreferencePair
from trajweave.core.specs import TeamSpec


class JointRewardModel(nn.Module):
    """Causal LM with an independent scalar head over the last real token."""

    def __init__(self, backbone: nn.Module, *, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        config = getattr(backbone, "config", None)
        hidden_size = (
            getattr(config, "hidden_size", None) or getattr(config, "n_embd", None) or getattr(config, "d_model", None)
        )
        if hidden_size is None:
            raise ValueError(
                "Could not infer reward model hidden size from config.hidden_size, config.n_embd, or config.d_model."
            )
        if isinstance(hidden_size, bool) or int(hidden_size) < 1:
            raise ValueError("Reward model hidden size must be a positive integer.")

        self.reward_head = nn.Linear(int(hidden_size), 1)
        reference = next(backbone.parameters(), None)
        if reference is not None and reference.is_floating_point():
            self.reward_head.to(device=reference.device, dtype=reference.dtype)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)

    def train(self, mode: bool = True) -> JointRewardModel:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must both be rank-2 tensors.")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have identical shapes.")
        if input_ids.shape[1] == 0:
            raise ValueError("Reward model inputs must contain at least one token position.")
        valid_tokens = attention_mask.to(dtype=torch.bool)
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only zero and one values.")
        if not torch.all(valid_tokens.any(dim=1)):
            raise ValueError("Reward model inputs cannot contain an all-padding row.")

        backbone_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
        }
        if _supports_position_ids(self.backbone):
            position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
            backbone_kwargs["position_ids"] = position_ids.masked_fill(~valid_tokens, 0)
        outputs = self.backbone(**backbone_kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError("Causal LM backbone must return hidden_states when requested.")
        final_hidden = hidden_states[-1]
        expected_shape = (*input_ids.shape, self.reward_head.in_features)
        if tuple(final_hidden.shape) != expected_shape:
            raise ValueError(
                f"Backbone final hidden state must have shape {expected_shape}; got {tuple(final_hidden.shape)}."
            )

        positions = torch.arange(input_ids.shape[1], device=attention_mask.device).expand_as(input_ids)
        last_non_pad = positions.masked_fill(~valid_tokens, -1).amax(dim=1)
        rows = torch.arange(input_ids.shape[0], device=final_hidden.device)
        pooled = final_hidden[rows, last_non_pad.to(device=final_hidden.device)]
        scores = self.reward_head(pooled).squeeze(-1)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Joint reward model produced a non-finite score.")
        return scores


def bradley_terry_loss(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> torch.Tensor:
    if chosen_scores.shape != rejected_scores.shape:
        raise ValueError("chosen_scores and rejected_scores must have identical shapes.")
    if chosen_scores.numel() == 0:
        raise ValueError("Bradley-Terry loss requires at least one preference pair.")
    if not torch.isfinite(chosen_scores).all() or not torch.isfinite(rejected_scores).all():
        raise FloatingPointError("Bradley-Terry scores must be finite.")
    loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Bradley-Terry loss is non-finite.")
    return loss


def serialize_joint_reward_text(
    team: TeamSpec,
    prompts_by_agent: Mapping[str, str],
    responses_by_agent: Mapping[str, str],
) -> str:
    """Serialize one joint action in immutable ``TeamSpec.agents`` order."""

    agent_names = tuple(agent.name for agent in team.agents)
    if not agent_names:
        raise ValueError("Joint reward text requires a team with at least one agent.")
    _require_exact_agents(prompts_by_agent, agent_names, "prompts")
    _require_exact_agents(responses_by_agent, agent_names, "responses")
    parts = ["Joint multi-agent response"]
    for index, agent_name in enumerate(agent_names, start=1):
        prompt = prompts_by_agent[agent_name]
        response = responses_by_agent[agent_name]
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise TypeError("Joint reward prompts and responses must be strings.")
        parts.append(f"Agent {index} prompt:\n{prompt}")
        parts.append(f"Agent {index} response:\n{response}")
    return "\n\n".join(parts)


def serialize_preference_pair(
    team: TeamSpec,
    pair: JointPreferencePair,
    *,
    chosen: bool,
) -> str:
    pair.validate()
    responses = pair.chosen_by_agent if chosen else pair.rejected_by_agent
    return serialize_joint_reward_text(team, pair.prompts_by_agent, responses)


class RewardModelWorker:
    """Standalone adapter for pairwise RM training, scoring, and checkpoints.

    This intentionally does not inherit VERL's token-value ``TrainingWorker``:
    a joint causal-LM reward head consumes complete multi-agent text and emits
    one scalar per joint action, rather than one critic value per token.
    """

    checkpoint_version = 1

    def __init__(
        self,
        model: JointRewardModel,
        tokenizer: Any,
        team: TeamSpec,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        max_length: int | None = None,
    ) -> None:
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be positive when provided.")
        self.model = model
        self.tokenizer = tokenizer
        self.team = team
        self.optimizer = optimizer
        self.max_length = max_length
        self._frozen_for_evaluation = False

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        team: TeamSpec,
        *,
        checkpoint_path: str | Path | None = None,
        freeze_backbone: bool = False,
        learning_rate: float | None = None,
        max_length: int | None = None,
        device: str | torch.device = "cpu",
        torch_dtype: str | torch.dtype | None = None,
        frozen_for_evaluation: bool = False,
    ) -> RewardModelWorker:
        if not str(model_name).strip():
            raise ValueError("Reward model_name must be non-empty.")
        if learning_rate is not None and learning_rate <= 0:
            raise ValueError("Reward model learning_rate must be positive when provided.")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs = {}
        resolved_dtype = _resolve_torch_dtype(torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                backbone.resize_token_embeddings(len(tokenizer))
        model = JointRewardModel(backbone, freeze_backbone=freeze_backbone).to(device)
        optimizer = None
        if learning_rate is not None:
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=float(learning_rate))
        worker = cls(model, tokenizer, team, optimizer=optimizer, max_length=max_length)
        if checkpoint_path is not None:
            worker.load_checkpoint(checkpoint_path)
        if frozen_for_evaluation:
            worker.freeze_for_evaluation()
        return worker

    @property
    def device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    @property
    def is_frozen_for_evaluation(self) -> bool:
        return self._frozen_for_evaluation

    def score_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, device=self.device)
        encoded = self._tokenize(texts)
        scores = self.model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Reward worker produced non-finite scores.")
        return scores

    def score_joint_actions(
        self,
        prompts: Sequence[Mapping[str, str]],
        responses: Sequence[Mapping[str, str]],
    ) -> torch.Tensor:
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must contain the same number of joint actions.")
        texts = [
            serialize_joint_reward_text(self.team, prompt, response)
            for prompt, response in zip(prompts, responses, strict=True)
        ]
        return self.score_texts(texts)

    def train_preference_batch(self, pairs: Sequence[JointPreferencePair]) -> torch.Tensor:
        if self._frozen_for_evaluation:
            raise RuntimeError("A frozen evaluation reward worker cannot be trained.")
        if self.optimizer is None:
            raise RuntimeError("Reward worker training requires an optimizer.")
        if not pairs:
            raise ValueError("Reward worker training requires at least one preference pair.")
        chosen = [serialize_preference_pair(self.team, pair, chosen=True) for pair in pairs]
        rejected = [serialize_preference_pair(self.team, pair, chosen=False) for pair in pairs]
        self.model.train()
        chosen_scores = self.score_texts(chosen)
        rejected_scores = self.score_texts(rejected)
        loss = bradley_terry_loss(chosen_scores, rejected_scores)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def freeze_for_evaluation(self) -> RewardModelWorker:
        if self._frozen_for_evaluation:
            raise RuntimeError("Reward worker is already frozen for evaluation.")
        self.model.eval()
        self.model.requires_grad_(False)
        self.optimizer = None
        self._frozen_for_evaluation = True
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.checkpoint_state_dict()

    def load_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        self.load_checkpoint_state_dict(checkpoint)

    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "version": self.checkpoint_version,
            "freeze_backbone": self.model.freeze_backbone,
            "reward_head": self.model.reward_head.state_dict(),
            "backbone": self.model.backbone.state_dict(),
        }

    def load_checkpoint_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        if int(checkpoint.get("version", -1)) != self.checkpoint_version:
            raise ValueError("Unsupported reward model checkpoint version.")
        checkpoint_freeze = checkpoint.get("freeze_backbone")
        if not isinstance(checkpoint_freeze, bool) or checkpoint_freeze != self.model.freeze_backbone:
            raise ValueError("Reward model checkpoint freeze_backbone configuration does not match the worker.")
        if "reward_head" not in checkpoint:
            raise ValueError("Reward model checkpoint is missing reward_head state.")
        if "backbone" not in checkpoint:
            raise ValueError("Reward model checkpoint is missing backbone state.")
        self.model.reward_head.load_state_dict(checkpoint["reward_head"], strict=True)
        self.model.backbone.load_state_dict(checkpoint["backbone"], strict=True)

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(self.checkpoint_state_dict(), Path(path))

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Reward model checkpoint must contain a mapping.")
        self.load_checkpoint_state_dict(checkpoint)

    def _tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        kwargs: dict[str, Any] = {
            "padding": True,
            "truncation": self.max_length is not None,
            "return_tensors": "pt",
        }
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        encoded = self.tokenizer(list(texts), **kwargs)
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise ValueError("Reward tokenizer must return input_ids and attention_mask.")
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }


def _resolve_torch_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return aliases[str(value).strip().lower()]
    except KeyError as error:
        raise ValueError("Reward model torch_dtype must be one of fp32, fp16, or bf16.") from error


def _supports_position_ids(backbone: nn.Module) -> bool:
    try:
        parameters = signature(backbone.forward).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == "position_ids" or parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters)


def _require_exact_agents(values: Mapping[str, str], agent_names: tuple[str, ...], field: str) -> None:
    expected = set(agent_names)
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Joint {field} must exactly match TeamSpec agents; missing={missing}, extra={extra}.")


__all__ = [
    "JointRewardModel",
    "RewardModelWorker",
    "bradley_terry_loss",
    "serialize_joint_reward_text",
    "serialize_preference_pair",
]
