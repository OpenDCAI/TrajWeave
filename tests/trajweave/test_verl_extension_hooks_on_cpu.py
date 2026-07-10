import numpy as np
import torch
from omegaconf import OmegaConf

from trajweave.backends.verl.extensions.common.hooks import (
    AgentFlowPlannerGRPOHooks,
    AgentWiseGRPOHooks,
    MAPoRLFullPPOHooks,
    PPOExtensionHooks,
    extension_hooks_for_config,
)
from trajweave.backends.verl.extensions.maporl.full_ppo import _install_maporl_algorithm_config
from trajweave.credit.maporl import MAPoRLPPOScoreRuleCreditAssigner
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories


def _batch() -> DataProto:
    tensors = {
        "token_level_rewards": torch.tensor([[1.0], [0.0], [0.0], [10.0]], dtype=torch.float32),
        "response_mask": torch.ones(4, 1, dtype=torch.long),
    }
    non_tensors = {
        "uid": np.array(["prompt"] * 4, dtype=object),
        "agent_id": np.array(["Solver Agent", "Verifier Agent", "Solver Agent", "Verifier Agent"], dtype=object),
        "traj_uid": np.array(["traj-0", "traj-0", "traj-1", "traj-1"], dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _maporl_batch(
    *,
    raw_scores: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    correctnesses: tuple[float, float, float, float] | None = None,
) -> DataProto:
    if correctnesses is None:
        correctnesses = raw_scores
    response_mask = torch.tensor([[1, 1, 0]] * 4, dtype=torch.long)
    tensors = {
        "token_level_scores": torch.zeros(4, 3, dtype=torch.float32),
        "response_mask": response_mask,
        "values": torch.zeros(4, 3, dtype=torch.float32),
    }
    non_tensors = {
        "uid": np.array(["prompt"] * 4, dtype=object),
        "traj_uid": np.array(["traj-0"] * 4, dtype=object),
        "round_id": np.array([0, 0, 1, 1], dtype=object),
        "agent_index": np.array([0, 1, 0, 1], dtype=object),
        "raw_score": np.array(raw_scores, dtype=object),
        "correctness": np.array(correctnesses, dtype=object),
        "finished_round": np.array([-1] * 4, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def _run_maporl_gae(
    config: dict,
    *,
    raw_scores: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    correctnesses: tuple[float, float, float, float] | None = None,
) -> DataProto:
    hooks = MAPoRLFullPPOHooks()
    data = hooks.prepare_dataproto(
        _maporl_batch(raw_scores=raw_scores, correctnesses=correctnesses),
        stage="advantage",
        config=config,
    )
    return hooks.compute_advantage(
        data,
        batch_keys=[f"prompt_0_{index}" for index in range(4)],
        adv_estimator=AdvantageEstimator.GAE,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config=config,
        fallback=compute_advantage_for_multi_trajectories,
    )


def test_agent_wise_hook_declares_stable_batch_contract():
    hooks = AgentWiseGRPOHooks()

    assert hooks.batch_schema_fields("advantage", config={}) == ("agent_id", "traj_uid", "turn_id")
    assert hooks.tq_select_fields("advantage", config={"group_by_agent_id": True})[-2:] == ("agent_id", "traj_uid")
    assert "agent_id" not in hooks.tq_select_fields("advantage", config={"group_by_agent_id": False})


def test_agent_wise_hook_builds_agent_advantage_groups():
    groups = AgentWiseGRPOHooks().build_advantage_groups(_batch())

    assert groups.tolist() == [
        "prompt_Solver Agent",
        "prompt_Verifier Agent",
        "prompt_Solver Agent",
        "prompt_Verifier Agent",
    ]


def test_agent_wise_hook_computes_drmas_grpo_advantage():
    data = _batch()
    result = AgentWiseGRPOHooks().compute_advantage(
        data,
        adv_estimator=AdvantageEstimator.GRPO,
        config=OmegaConf.create({"group_by_agent_id": True}),
    )

    assert result.batch["advantages"][0].item() > 0
    assert result.batch["advantages"][1].item() < 0
    assert result.batch["advantages"][2].item() < 0
    assert result.batch["advantages"][3].item() > 0


def test_extension_hooks_for_config_selects_agent_wise_grpo():
    assert isinstance(
        extension_hooks_for_config({"trajweave": {"credit_allocator": "drmas_agent_wise_grpo"}}),
        AgentWiseGRPOHooks,
    )
    assert isinstance(
        extension_hooks_for_config({"trajweave": {"credit_allocator": "maporl_score_bonus"}}),
        AgentWiseGRPOHooks,
    )
    assert isinstance(extension_hooks_for_config({}), PPOExtensionHooks)


def test_extension_hooks_for_config_selects_maporl_full_ppo():
    hooks = extension_hooks_for_config({"trajweave": {"credit_allocator": "maporl_ppo_score_rule"}})

    assert isinstance(hooks, MAPoRLFullPPOHooks)
    fields = hooks.tq_select_fields("advantage", default_fields=("uid", "rm_scores"), config={})
    assert "traj_uid" in hooks.batch_schema_fields("advantage", config={})
    assert "round_id" in fields
    assert "raw_score" in fields
    assert "policy_group" in fields
    assert "worker_group" in fields


def test_maporl_hook_rule_horizon_changes_exact_training_rewards():
    common = {"rule_agent_share": "individual", "include_bonus": False}

    current = _run_maporl_gae({"maporl": {**common, "rule_horizon": "current"}})
    last = _run_maporl_gae({"maporl": {**common, "rule_horizon": "last"}})

    torch.testing.assert_close(current.batch["token_level_rewards"].sum(-1), torch.tensor([0.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(last.batch["token_level_rewards"].sum(-1), torch.tensor([1.0, 1.0, 1.0, 1.0]))
    assert "returns" in current.batch
    assert "advantages" in current.batch


def test_maporl_hook_rule_discount_changes_exact_training_rewards():
    common = {
        "rule_horizon": "discounted_sum",
        "rule_agent_share": "individual",
        "include_bonus": False,
    }

    discount_quarter = _run_maporl_gae({"algorithm": {"maporl": {**common, "rule_discount": 0.25}}})
    discount_one = _run_maporl_gae({"algorithm": {"maporl": {**common, "rule_discount": 1.0}}})

    torch.testing.assert_close(
        discount_quarter.batch["token_level_rewards"].sum(-1),
        torch.tensor([0.2, 0.2, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        discount_one.batch["token_level_rewards"].sum(-1),
        torch.tensor([0.5, 0.5, 1.0, 1.0]),
    )


def test_maporl_hook_alpha_changes_exact_training_rewards_from_trajweave_config():
    common = {"rule_horizon": "current", "rule_agent_share": "individual"}
    correctnesses = (0.0, 1.0, 1.0, 0.0)

    no_bonus = _run_maporl_gae(
        {"trajweave": {"maporl": {**common, "alpha": [0.0, 0.0, 0.0, 0.0]}}},
        raw_scores=(0.0, 0.0, 0.0, 0.0),
        correctnesses=correctnesses,
    )
    with_bonus = _run_maporl_gae(
        {"trajweave": {"maporl": {**common, "alpha": [0.2, 0.0, 0.0, 0.7]}}},
        raw_scores=(0.0, 0.0, 0.0, 0.0),
        correctnesses=correctnesses,
    )

    torch.testing.assert_close(no_bonus.batch["token_level_rewards"].sum(-1), torch.zeros(4))
    torch.testing.assert_close(
        with_bonus.batch["token_level_rewards"].sum(-1),
        torch.tensor([0.7, -0.7, 0.2, -0.2]),
    )


def test_maporl_hook_matches_shared_credit_assigner_and_exports_rewards():
    config = {
        "maporl": {
            "rule_horizon": "discounted_sum",
            "rule_agent_share": "individual",
            "rule_discount": 0.25,
            "alpha": [0.0, 0.0, 0.0, 0.0],
        }
    }
    result = _run_maporl_gae(config)
    expected = MAPoRLPPOScoreRuleCreditAssigner.from_config(config).score_turns(
        round_ids=[0, 0, 1, 1],
        agent_indices=[0, 1, 0, 1],
        raw_scores=[0.0, 0.0, 1.0, 1.0],
        correctnesses=[0.0, 0.0, 1.0, 1.0],
        finished_round=-1,
    )

    np.testing.assert_allclose(
        result.non_tensor_batch["maporl_reward"],
        [credit.reward for credit in expected],
    )
    assert MAPoRLFullPPOHooks().output_fields("advantage", ("advantages", "returns"), result) == (
        "advantages",
        "returns",
        "token_level_scores",
        "token_level_rewards",
    )


def test_maporl_hook_ignores_padding_rows_and_keeps_their_rewards_zero():
    source = _maporl_batch()
    tensors = {
        field: torch.cat([value, torch.ones(2, 3, dtype=value.dtype)], dim=0) for field, value in source.batch.items()
    }
    tensors["response_mask"][4:] = 0
    non_tensors = {
        field: np.concatenate([value, np.array(padding_values, dtype=object)])
        for field, value, padding_values in (
            ("uid", source.non_tensor_batch["uid"], ["pad-a", "pad-b"]),
            ("traj_uid", source.non_tensor_batch["traj_uid"], ["pad-a", "pad-b"]),
            ("round_id", source.non_tensor_batch["round_id"], [-1, -1]),
            ("agent_index", source.non_tensor_batch["agent_index"], [-1, -1]),
            ("raw_score", source.non_tensor_batch["raw_score"], [0.0, 0.0]),
            ("correctness", source.non_tensor_batch["correctness"], [0.0, 0.0]),
            ("finished_round", source.non_tensor_batch["finished_round"], [-1, -1]),
        )
    }
    data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    hooks = MAPoRLFullPPOHooks()
    result = hooks.prepare_dataproto(
        data,
        stage="advantage",
        config={"maporl": {"rule_horizon": "current", "rule_agent_share": "individual"}},
    )

    torch.testing.assert_close(result.batch["token_level_scores"][4:], torch.zeros(2, 3))
    torch.testing.assert_close(result.batch["token_level_rewards"][4:], torch.zeros(2, 3))
    np.testing.assert_allclose(result.non_tensor_batch["maporl_score"][4:], [0.0, 0.0])
    np.testing.assert_allclose(result.non_tensor_batch["maporl_bonus"][4:], [0.0, 0.0])
    np.testing.assert_allclose(result.non_tensor_batch["maporl_reward"][4:], [0.0, 0.0])

    metrics = hooks.compute_extra_metrics(result, {}, "advantage")
    assert metrics["trajweave/maporl/raw_score/mean"] == 0.5


def test_maporl_runtime_copies_orchestra_credit_config_into_algorithm():
    config = {
        "algorithm": {},
        "agent": {
            "orchestra": {
                "maporl": {
                    "rule_horizon": "last",
                    "rule_agent_share": "individual",
                    "rule_discount": 0.8,
                    "alpha": [0.1, 0.2, 0.3, 0.4],
                }
            }
        },
    }

    _install_maporl_algorithm_config(config)

    assert config["algorithm"]["maporl"]["rule_horizon"] == "last"
    assert config["algorithm"]["maporl"]["rule_discount"] == 0.8
    assert config["algorithm"]["maporl"]["alpha"] == (0.1, 0.2, 0.3, 0.4)


def test_extension_hooks_for_config_selects_agentflow_planner_grpo():
    hooks = extension_hooks_for_config({"trajweave": {"credit_allocator": "agentflow_planner_only_grpo"}})

    assert isinstance(hooks, AgentFlowPlannerGRPOHooks)
    fields = hooks.tq_select_fields("advantage", default_fields=("uid", "rm_scores"), config={})
    assert "agentflow_stage" in fields
    assert "tool_name" in fields
    assert "verifier_decision" in fields
    assert "step_id" in fields


def test_agentflow_hook_delegates_advantage_to_fallback():
    calls = []

    def fallback(data, **kwargs):
        calls.append(kwargs)
        return {"ok": True}

    result = AgentFlowPlannerGRPOHooks().compute_advantage(
        "batch",
        adv_estimator=AdvantageEstimator.GRPO,
        fallback=fallback,
        batch_keys=["responses"],
    )

    assert result == {"ok": True}
    assert calls[0]["batch_keys"] == ["responses"]
