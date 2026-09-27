from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from trajweave.backends.verl.agent_loops.registry import validate_agent_loop_backend
from trajweave.backends.verl.async_buffer import PolicyBufferCoordinator
from trajweave.backends.verl.emitters.registry import supported_emitter_recipes
from trajweave.backends.verl.trainers.mrlx_async import TrajWeaveMrlXAsyncTrainer
from trajweave.backends.verl.weight_sync import MultiActorWeightSyncContract


def test_atgrpo_padding_does_not_change_real_sibling_baseline():
    from trajweave.backends.verl.extensions.common.hooks import ATGRPOHooks

    hook = ATGRPOHooks()
    real, _ = hook.compute_grpo_outcome_advantage(
        token_level_rewards=torch.tensor([[0.0], [1.0]]),
        response_mask=torch.ones(2, 1), index=np.array(["obs", "obs"]),
        group_by_agent_id=True,
    )
    padded, _ = hook.compute_grpo_outcome_advantage(
        token_level_rewards=torch.tensor([[0.0], [1.0], [0.0]]),
        response_mask=torch.tensor([[1.0], [1.0], [0.0]]),
        index=np.array(["obs", "obs", "obs"]), group_by_agent_id=True,
    )
    torch.testing.assert_close(padded[:2], real)
    assert padded[-1].item() == 0.0


def test_all_merged_recipes_have_backend_routes():
    recipes = supported_emitter_recipes()
    assert len(recipes) == 15
    assert {"comlrl_joint_math", "marti_mars2_single_mcts", "atgrpo_solver_verifier_math"} <= recipes
    for recipe in recipes:
        validate_agent_loop_backend(recipe, "hf_local_tq")
        validate_agent_loop_backend(recipe, "synthetic_tq")
    validate_agent_loop_backend("marti_mars2_single_mcts", "vllm_marti_tq")
    with pytest.raises(ValueError, match="does not support"):
        validate_agent_loop_backend("comlrl_joint_math", "vllm_marti_tq")


def test_mrlx_actor_update_records_lz_policy_version():
    trainer = object.__new__(TrajWeaveMrlXAsyncTrainer)
    trainer.config = OmegaConf.create({
        "trajweave": {"recipe": "mrlx_research_qa"},
        "actor_rollout_ref": {"actor": {
            "calculate_entropy": False, "entropy_coeff": 0.0,
            "ppo_epochs": 1, "data_loader_seed": 0, "shuffle": False,
        }, "rollout": {"temperature": 1.0}},
    })
    trainer.global_steps = 2
    trainer.maporl_weight_sync = MultiActorWeightSyncContract(("explorer", "adapter"))
    trainer.maporl_buffer_coordinator = PolicyBufferCoordinator(("explorer", "adapter"))
    updated = []
    trainer.actor_rollout_wgs = {"adapter": SimpleNamespace(
        update_actor=lambda batch: updated.append(batch) or {"metrics": {"grad_norm": 0.5}},
    )}
    batch = SimpleNamespace(keys=["sample"], extra_info={})
    trainer._update_group("adapter", batch, phase="delayed")
    trainer._update_group("adapter", batch, phase="drain")
    assert len(updated) == 2
    assert trainer.maporl_weight_sync.as_dict()["versions"]["adapter"]["global_step"] == 2
    assert trainer.maporl_buffer_coordinator.states["adapter"].actor_step == 2
    assert "explorer" not in trainer.maporl_weight_sync.as_dict()["versions"]


def test_mrlx_does_not_inherit_model_only_checkpoint_resume(tmp_path):
    trainer = object.__new__(TrajWeaveMrlXAsyncTrainer)
    trainer.config = OmegaConf.create({"trainer": {
        "resume_mode": "auto", "resume_from_path": None, "default_local_dir": str(tmp_path),
    }})
    trainer._load_checkpoint()
    assert trainer.global_steps == 0
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("1\n")
    with pytest.raises(ValueError, match="Adapter replay"):
        trainer._load_checkpoint()


def test_marti_hook_supports_omitted_default_fields_from_shared_runtime():
    from trajweave.backends.verl.extensions.common.hooks import PPOExtensionHooks
    from trajweave.backends.verl.extensions.marti_mars2.tree_grpo import MARTIMARS2TreeGRPOHooks

    base = PPOExtensionHooks()
    assert base.tq_select_fields("actor") == ()
    assert base.tq_select_fields("advantage", default_fields=()) == ()
    fields = MARTIMARS2TreeGRPOHooks().tq_select_fields("advantage", config={})
    assert {"uid", "response_mask", "rm_scores", "values", "tree_id", "node_id", "path"} <= set(fields)
    assert len(fields) == len(set(fields))
