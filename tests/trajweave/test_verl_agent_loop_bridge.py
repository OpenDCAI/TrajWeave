import pytest


def test_trajweave_agent_loop_runtime_config_reads_verl_overrides():
    pytest.importorskip("torch")
    pytest.importorskip("transfer_queue")
    from omegaconf import OmegaConf

    from trajweave.backends.verl.agent_loop import TrajWeaveAgentLoopRuntimeConfig

    config = OmegaConf.create(
        {
            "trajweave": {
                "recipe": "doctor_mas_math",
                "config": "configs/drmas/math_smoke.yaml",
                "coordination_protocol": "solver_verifier_loop",
                "trajectory_schema": "multi_agent_turn_v1",
                "credit_allocator": "doctor_mas_agent_wise",
            }
        }
    )

    runtime = TrajWeaveAgentLoopRuntimeConfig.from_verl_config(config)

    assert runtime.recipe == "doctor_mas_math"
    assert runtime.config_path == "configs/drmas/math_smoke.yaml"
    assert runtime.as_overrides()["trajweave.credit_allocator"] == "doctor_mas_agent_wise"


def test_verl_can_load_trajweave_agent_loop_manager_by_fqn():
    pytest.importorskip("torch")
    pytest.importorskip("transfer_queue")
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
    from verl.utils.import_utils import load_class_from_fqn

    from trajweave.backends.verl.agent_loop import TRAJWEAVE_AGENT_LOOP_MANAGER_FQN

    manager_cls = load_class_from_fqn(TRAJWEAVE_AGENT_LOOP_MANAGER_FQN, "AgentLoopManager")

    assert issubclass(manager_cls, AgentLoopManagerTQ)
    assert hasattr(manager_cls, "create")
    assert hasattr(manager_cls, "generate_sequences")


def test_trajweave_pads_rm_scores_to_response_mask_length():
    torch = pytest.importorskip("torch")

    from trajweave.backends.verl.agent_loop import _padded_rm_scores

    response_mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.int64)

    rm_scores = _padded_rm_scores(response_mask, reward_score=0.75, response_len=3)

    assert rm_scores.shape == response_mask.shape
    assert rm_scores.tolist() == [0.0, 0.0, 0.75, 0.0, 0.0]


def test_trajweave_agent_loop_accepts_agentflow_recipe_and_rejects_native_backend():
    pytest.importorskip("torch")
    pytest.importorskip("transfer_queue")

    from trajweave.backends.verl.agent_loop import _validate_agent_loop_backend

    _validate_agent_loop_backend("agentflow_planner_tool", "synthetic_tq")
    _validate_agent_loop_backend("agentflow_planner_tool", "hf_local_tq")
    with pytest.raises(ValueError, match="verl_tq"):
        _validate_agent_loop_backend("agentflow_planner_tool", "verl_tq")


def test_trajweave_agent_loop_to_python_supports_omegaconf_lists():
    pytest.importorskip("torch")
    pytest.importorskip("transfer_queue")
    from omegaconf import OmegaConf

    from trajweave.backends.verl.agent_loop import _to_python

    value = OmegaConf.create(
        [
            {
                "id": "qwen2.5/0.5b",
                "model_path": "/models/qwen2.5",
                "tokenizer_path": "/models/tokenizer",
            }
        ]
    )

    assert _to_python(value) == [
        {
            "id": "qwen2.5/0.5b",
            "model_path": "/models/qwen2.5",
            "tokenizer_path": "/models/tokenizer",
        }
    ]
