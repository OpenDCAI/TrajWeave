from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, Qwen2Config

from trajweave.backends.verl.extensions.drmas.agent_wise_grpo import (
    TrajWeaveActorRolloutRefWorker,
    _collect_merged_lora_snapshot_state_dict,
)
from verl.utils.checkpoint import fsdp_checkpoint_manager
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


class _FakeFSDPModel:
    def __init__(self, model):
        self._fsdp_wrapped_module = model


class _FakeCheckpointManager:
    def __init__(self, model):
        self.model = model
        self.checkpoint_save_contents = ["model", "optimizer", "extra"]
        self.previous_saved_paths = ["previous"]
        self.previous_global_step = 7


class _FakeEngine:
    def __init__(self, manager, config):
        self.checkpoint_manager = manager
        self.config = config

    def save_checkpoint(self, *, local_path, global_step, max_ckpt_to_keep):
        assert self.checkpoint_manager.checkpoint_save_contents == ["hf_model"]
        assert global_step == 8
        assert max_ckpt_to_keep is None
        state_dict = fsdp_checkpoint_manager.get_fsdp_full_state_dict(
            self.checkpoint_manager.model,
            offload_to_cpu=True,
            rank0_only=True,
        )
        output = Path(local_path) / "huggingface"
        output.mkdir(parents=True)
        AutoModelForCausalLM.from_config(self.config).save_pretrained(output, state_dict=state_dict)


def test_lora_rollout_snapshot_saves_merged_hf_weights(monkeypatch, tmp_path):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    actor = get_peft_model(
        AutoModelForCausalLM.from_config(config),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    ).eval()
    with torch.no_grad():
        for name, parameter in actor.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.25)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected_logits = actor(input_ids).logits
    merged_model = copy.deepcopy(actor).merge_and_unload().eval()
    merged_state = {name: value.detach().clone() for name, value in merged_model.state_dict().items()}

    wrapped = _FakeFSDPModel(actor)
    manager = _FakeCheckpointManager(wrapped)
    engine = _FakeEngine(manager, config)
    worker = object.__new__(TrajWeaveActorRolloutRefWorker)
    worker.actor = SimpleNamespace(engine=engine)
    collected = []

    def collect_merged(candidate):
        collected.append(candidate)
        return merged_state

    monkeypatch.setattr("verl.utils.fsdp_utils.collect_merged_lora_params", collect_merged)
    original_get_state_dict = fsdp_checkpoint_manager.get_fsdp_full_state_dict

    worker.export_hf_rollout_snapshot(str(tmp_path), global_step=8)

    assert collected == [wrapped]
    assert fsdp_checkpoint_manager.get_fsdp_full_state_dict is original_get_state_dict
    assert manager.checkpoint_save_contents == ["model", "optimizer", "extra"]
    assert manager.previous_saved_paths == ["previous"]
    assert manager.previous_global_step == 7

    reloaded, loading_info = AutoModelForCausalLM.from_pretrained(
        tmp_path / "huggingface",
        output_loading_info=True,
    )
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert not loading_info["mismatched_keys"]
    assert all("lora_" not in key and "base_model.model." not in key for key in reloaded.state_dict())
    with torch.no_grad():
        actual_logits = reloaded(input_ids).logits
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)


def test_cpu_fsdp_lora_collection_copies_merged_weights_and_restores_actor(tmp_path):
    if dist.is_initialized():
        pytest.skip("CPU FSDP snapshot test requires an isolated process group.")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'fsdp-rendezvous'}",
        rank=0,
        world_size=1,
    )
    try:
        torch.manual_seed(1)
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        config.architectures = ["Qwen2ForCausalLM"]
        actor = get_peft_model(
            AutoModelForCausalLM.from_config(config),
            LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        ).eval()
        with torch.no_grad():
            for name, parameter in actor.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.25)
        actor = FSDP(actor, use_orig_params=True, device_id=torch.device("cpu")).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        adapter_before = {
            name: parameter.detach().clone() for name, parameter in actor.named_parameters() if "lora_" in name
        }
        with torch.no_grad():
            logits_before = actor(input_ids).logits

        merged_state = _collect_merged_lora_snapshot_state_dict(actor)

        assert merged_state
        assert all(value.device.type == "cpu" for value in merged_state.values())
        assert all("lora_" not in name and "base_model.model." not in name for name in merged_state)
        assert all(not layer.merged for layer in actor.modules() if isinstance(layer, LoraLayer))
        for name, parameter in actor.named_parameters():
            if name in adapter_before:
                torch.testing.assert_close(parameter, adapter_before[name], rtol=0, atol=0)
        with torch.no_grad():
            logits_after = actor(input_ids).logits
        torch.testing.assert_close(logits_after, logits_before, rtol=0, atol=0)

        reloaded = AutoModelForCausalLM.from_config(config).eval()
        reloaded.load_state_dict(merged_state, strict=True)
        with torch.no_grad():
            reloaded_logits = reloaded(input_ids).logits
        torch.testing.assert_close(reloaded_logits, logits_before, rtol=1e-5, atol=1e-6)

        manager = FSDPCheckpointManager(
            model=actor,
            optimizer=None,
            checkpoint_config=OmegaConf.create(
                {
                    "load_contents": ["model", "optimizer", "extra"],
                    "save_contents": ["model", "optimizer", "extra"],
                }
            ),
        )
        worker = object.__new__(TrajWeaveActorRolloutRefWorker)
        worker.actor = SimpleNamespace(
            engine=SimpleNamespace(
                checkpoint_manager=manager,
                save_checkpoint=manager.save_checkpoint,
            )
        )
        snapshot_path = tmp_path / "snapshot"

        worker.export_hf_rollout_snapshot(str(snapshot_path), global_step=1)

        assert (snapshot_path / "lora_train_meta.json").is_file()
        assert not (snapshot_path / "huggingface" / "adapter_config.json").exists()
        disk_model, loading_info = AutoModelForCausalLM.from_pretrained(
            snapshot_path / "huggingface",
            dtype=torch.float32,
            output_loading_info=True,
        )
        assert not loading_info["missing_keys"]
        assert not loading_info["unexpected_keys"]
        assert not loading_info["mismatched_keys"]
        with torch.no_grad():
            disk_logits = disk_model(input_ids).logits
        torch.testing.assert_close(disk_logits, logits_before, rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()
