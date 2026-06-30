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
                "config": "examples/trajweave/configs/doctor_mas_math_smoke.yaml",
                "coordination_protocol": "solver_verifier_loop",
                "trajectory_schema": "multi_agent_turn_v1",
                "credit_allocator": "doctor_mas_agent_wise",
            }
        }
    )

    runtime = TrajWeaveAgentLoopRuntimeConfig.from_verl_config(config)

    assert runtime.recipe == "doctor_mas_math"
    assert runtime.config_path == "examples/trajweave/configs/doctor_mas_math_smoke.yaml"
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
