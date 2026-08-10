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
    assert runtime.turn_padding_multiple == 2


def test_trajweave_agent_loop_runtime_config_validates_hf_local_dtype():
    from trajweave.backends.verl.runtime_config import TrajWeaveAgentLoopRuntimeConfig

    runtime = TrajWeaveAgentLoopRuntimeConfig.from_verl_config(
        {"trajweave": {"hf_local_dtype": "float16", "hf_local_model_cache_size": 1}}
    )

    assert runtime.hf_local_dtype == "fp16"
    assert runtime.hf_local_model_cache_size == 1
    assert runtime.as_overrides()["trajweave.hf_local_dtype"] == "fp16"
    assert runtime.as_overrides()["trajweave.hf_local_model_cache_size"] == "1"
    with pytest.raises(ValueError, match="hf_local_dtype"):
        TrajWeaveAgentLoopRuntimeConfig.from_verl_config({"trajweave": {"hf_local_dtype": "int8"}})
    with pytest.raises(ValueError, match="cache_size"):
        TrajWeaveAgentLoopRuntimeConfig.from_verl_config({"trajweave": {"hf_local_model_cache_size": -1}})


def test_hf_local_dtype_maps_to_torch_dtype():
    torch = pytest.importorskip("torch")

    from trajweave.backends.verl.local_generation import _torch_dtype

    assert _torch_dtype("fp32") is torch.float32
    assert _torch_dtype("fp16") is torch.float16
    assert _torch_dtype("bf16") is torch.bfloat16


def test_hf_local_model_cache_evicts_before_loading_next_group():
    from trajweave.backends.verl.local_generation import HFLocalGenerationMixin, _evict_local_model_cache

    first_model = object()
    cache = {"group_a": first_model}

    evicted = _evict_local_model_cache(cache, max_cached_models=1)

    assert evicted == ("group_a",)
    assert cache == {}
    assert _evict_local_model_cache({"group_a": first_model}, max_cached_models=0) == ()

    worker = type("Worker", (HFLocalGenerationMixin,), {})()
    worker._trajweave_local_models = {"group_b": object()}
    worker._trajweave_policy_version = 3
    result = worker.release_local_models()
    assert result == {"released_groups": ["group_b"], "policy_version": 3}
    assert worker._trajweave_local_models == {}


def test_verl_can_load_trajweave_agent_loop_manager_by_fqn():
    pytest.importorskip("torch")
    pytest.importorskip("transfer_queue")
    from trajweave.backends.verl.agent_loop import TRAJWEAVE_AGENT_LOOP_MANAGER_FQN
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
    from verl.utils.import_utils import load_class_from_fqn

    manager_cls = load_class_from_fqn(TRAJWEAVE_AGENT_LOOP_MANAGER_FQN, "AgentLoopManager")

    assert issubclass(manager_cls, AgentLoopManagerTQ)
    assert hasattr(manager_cls, "create")
    assert hasattr(manager_cls, "generate_sequences")


def test_trajweave_pads_rm_scores_to_response_mask_length():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transfer_queue")

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
    _validate_agent_loop_backend("marti_mars2_single_mcts", "vllm_marti_tq")
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


def test_native_vllm_sampling_params_only_include_policy_marker_for_grouped_client():
    pytest.importorskip("torch")
    pytest.importorskip("transfer_queue")

    from trajweave.backends.verl.agent_loop import _marti_vllm_sampling_params

    base = {"temperature": 0.7}
    single = _marti_vllm_sampling_params({}, base, policy_group="shared")
    grouped = _marti_vllm_sampling_params(
        {"trajweave": {"multi_actor": {"vllm": {"enabled": True}}}},
        base,
        policy_group="policy_b",
    )

    assert single == {"temperature": 0.7}
    assert grouped == {"temperature": 0.7, "trajweave_policy_group": "policy_b"}
    assert base == {"temperature": 0.7}
