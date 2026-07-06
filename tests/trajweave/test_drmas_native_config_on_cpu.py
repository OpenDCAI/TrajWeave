from trajweave.runner import run_from_config


def test_drmas_native_math_config_prepares_agent_wise_verl_launch():
    result = run_from_config(
        {
            "recipe": "drmas_native_math",
            "mode": "verl_train",
            "prepare": {"tiny_verl_assets": {"enabled": False}},
            "drmas_native": {
                "agent_ids": ["Solver Agent", "Verifier Agent"],
                "model_ids": ["shared", "shared"],
                "agent_loop_backend": "synthetic_tq",
                "max_loop_num": 2,
            },
            "verl": {
                "enabled": True,
                "execute": False,
                "overrides": ["trainer.use_v1=true"],
            },
        },
        config_path="configs/drmas_native_math_verl_tiny.yaml",
    )

    command = result["verl_launch"]["command"]
    command_text = " ".join(command)
    assert result["drmas_native"]["task"] == "math"
    assert "trajweave.backends.verl.main_ppo" in command
    assert "++algorithm.group_by_agent_id=true" in command
    assert "+agent.agent_ids=[\"Solver Agent\",\"Verifier Agent\"]" in command
    assert "+trajweave.verl_extensions=[drmas_agent_wise_grpo]" in command
    assert "+agent.orchestra.math.max_loop_num=2" in command
    assert "+trajweave.agent_loop_backend=synthetic_tq" in command
    assert "trajweave.backends.verl.agent_loop.TrajWeaveAgentLoopManager" in command_text


def test_drmas_native_config_rejects_native_verl_tq_backend():
    try:
        run_from_config(
            {
                "recipe": "drmas_native_math",
                "mode": "verl_train",
                "prepare": {"tiny_verl_assets": {"enabled": False}},
                "drmas_native": {
                    "agent_ids": ["Solver Agent", "Verifier Agent"],
                    "model_ids": ["shared", "shared"],
                    "agent_loop_backend": "verl_tq",
                },
                "verl": {"enabled": True, "execute": False},
            }
        )
    except ValueError as exc:
        assert "not verl_tq" in str(exc)
    else:
        raise AssertionError("DrMAS native should reject native verl_tq.")
