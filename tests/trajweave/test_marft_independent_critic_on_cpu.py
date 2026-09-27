from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from trajweave.backends.verl.trainers.marft_critic_sync import (
    MARFTIndependentCriticSyncTrainer,
    MARFTMultiActorSyncTrainer,
    MARFTSharedCriticSyncTrainer,
)
from trajweave.backends.verl.workers.value_lora import (
    MARFTValueCriticTrainingWorker,
    _build_value_model_lora_module,
    _install_value_model_lora_builder,
)
from trajweave.recipes.marft.config import build_marft_launch_overrides, resolve_marft_settings


def _independent_config(mode: str, *, use_multi_lora: bool = True, overrides: list[str] | None = None) -> dict:
    groups = {
        role: {
            "model_path": f"/models/{role}",
            "tokenizer_path": "/tokenizer",
            "trainable": True,
            "gpus": 1,
        }
        for role in ("planner", "solver")
    }
    return {
        "marft": {
            "role_names": ["planner", "solver"],
            "model_ids": ["planner", "solver"],
            "shared_policy": False,
            "use_multi_lora": use_multi_lora,
            "shared_lora": False,
            "lora_rank": 4,
            "lora_alpha": 8,
            "independent_critic": mode,
            "kl_coef": 0.0,
            "agent_loop_backend": "synthetic_tq",
            "worker_groups": groups,
        },
        "verl": {"overrides": list(overrides or ())},
    }


def test_marft_value_critic_lora_uses_token_classification_and_trains_score_head():
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    model_config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_labels=1,
    )
    value_model = transformers.Qwen2ForTokenClassification(model_config)
    critic_model_config = SimpleNamespace(
        lora_rank=2,
        lora_alpha=4,
        target_modules="all-linear",
        target_parameters=None,
        exclude_modules=None,
        lora_adapter_path=None,
    )
    engine = SimpleNamespace(model_config=critic_model_config, _build_lora_module=lambda module: module)
    _install_value_model_lora_builder(engine, model_config=critic_model_config)
    wrapped = engine._build_lora_module(value_model)

    assert wrapped.peft_config["default"].task_type == peft.TaskType.TOKEN_CLS
    trainable_score = [
        parameter for name, parameter in wrapped.named_parameters() if "score" in name and parameter.requires_grad
    ]
    assert trainable_score

    output = wrapped(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
    )
    output.logits.square().mean().backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in trainable_score
    )


def test_marft_value_critic_rejects_causal_lm_adapter_before_wrapping(tmp_path):
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")

    model_config = transformers.Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_labels=1,
    )
    actor = peft.get_peft_model(
        transformers.Qwen2ForCausalLM(model_config),
        peft.LoraConfig(task_type=peft.TaskType.CAUSAL_LM, r=2, target_modules="all-linear"),
    )
    actor.save_pretrained(tmp_path)
    critic_model_config = SimpleNamespace(
        lora_rank=2,
        lora_alpha=4,
        target_modules="all-linear",
        target_parameters=None,
        exclude_modules=None,
        lora_adapter_path=str(tmp_path),
        use_shm=False,
    )

    with pytest.raises(ValueError, match="must use PEFT TaskType.TOKEN_CLS"):
        _build_value_model_lora_module(
            SimpleNamespace(model_config=critic_model_config),
            transformers.Qwen2ForTokenClassification(model_config),
        )


def test_marft_independent_critic_trainer_selects_value_lora_worker():
    trainer_classes = (
        MARFTSharedCriticSyncTrainer,
        MARFTMultiActorSyncTrainer,
        MARFTIndependentCriticSyncTrainer,
    )

    assert all(
        object.__new__(trainer_class)._critic_training_worker_cls() is MARFTValueCriticTrainingWorker
        for trainer_class in trainer_classes
    )


def test_marft_lora_critic_requires_multi_lora_and_separate_clears_stale_rank():
    with pytest.raises(ValueError, match="independent_critic=lora requires use_multi_lora=true"):
        resolve_marft_settings(_independent_config("lora", use_multi_lora=False))

    lora_overrides = build_marft_launch_overrides(_independent_config("lora"), config_path="marft.yaml")
    assert [item for item in lora_overrides if item.startswith("critic.model.lora_rank=")] == [
        "critic.model.lora_rank=4"
    ]

    separate_overrides = build_marft_launch_overrides(
        _independent_config("separate", overrides=["critic.model.lora_rank=99"]),
        config_path="marft.yaml",
    )
    assert [item for item in separate_overrides if item.startswith("critic.model.lora_rank=")] == [
        "critic.model.lora_rank=0"
    ]
