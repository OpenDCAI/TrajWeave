from verl.workers.config import RolloutConfig


def test_rollout_config_accepts_vllm_seed():
    config = RolloutConfig(name="vllm", seed=42)

    assert config.seed == 42


def test_rollout_config_accepts_multi_actor_name_suffix():
    config = RolloutConfig(name="vllm", name_suffix="policy_a")

    assert config.name_suffix == "policy_a"
