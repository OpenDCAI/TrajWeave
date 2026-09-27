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


def test_atgrpo_emitter_route_builds_selected_spine_outputs():
    from trajweave.backends.verl.emitters.atgrpo import ATGRPOEmitterMixin
    from trajweave.backends.verl.emitters.registry import build_recipe_outputs, supported_emitter_recipes
    from trajweave.backends.verl.runtime_config import validate_agent_loop_backend

    class _Worker(ATGRPOEmitterMixin):
        config = {"agent": {"orchestra": {"atgrpo": {"max_turns": 3}}}}

        @staticmethod
        def _encode_prompt(value):
            return [11]

        @staticmethod
        def _encode_text(value):
            return [len(value)]

    assert "atgrpo_solver_verifier_math" in supported_emitter_recipes()
    validate_agent_loop_backend("atgrpo_solver_verifier_math", "synthetic_tq")
    validate_agent_loop_backend("atgrpo_solver_verifier_math", "hf_local_tq")
    outputs = build_recipe_outputs(
        _Worker(),
        recipe="atgrpo_solver_verifier_math",
        use_hf_local=False,
        prompt={
            "uid": "prompt",
            "raw_prompt": [{"content": "2 + 2"}],
            "reward_model": {"ground_truth": "4"},
            "__atgrpo_branch_factor__": 2,
        },
        session_id=0,
    )

    assert len(outputs) == 4
    assert {output.extra_fields["turn_id"] for output in outputs} == {0, 1}
    assert len({output.extra_fields["root_id"] for output in outputs}) == 1
    for turn_id in range(2):
        siblings = [output for output in outputs if output.extra_fields["turn_id"] == turn_id]
        assert len(siblings) == 2
        assert len({output.extra_fields["observation_group_id"] for output in siblings}) == 1
        assert sum(output.extra_fields["selected_for_expansion"] for output in siblings) == 1

    solver = [output for output in outputs if output.extra_fields["turn_id"] == 0]
    verifier = [output for output in outputs if output.extra_fields["turn_id"] == 1]
    assert [output.extra_fields["trajweave_role"] for output in outputs] == [
        "solver",
        "solver",
        "verifier",
        "verifier",
    ]
    assert [output.extra_fields["local_score"] for output in solver] == [1.0, 0.0]
    assert [output.extra_fields["local_score"] for output in verifier] == [1.0, -1.0]
    assert {output.reward_score for output in outputs} == {1.0}
    assert all(output.extra_fields["prompt_text"] for output in outputs)
    assert [output.extra_fields["response_text"] for output in outputs] == [
        "Final answer: 4",
        "Final answer: __wrong__",
        "APPROVED",
        "REVISE",
    ]
    assert {output.extra_fields["observation_text"] for output in outputs} == {"2 + 2"}
    assert {output.extra_fields["final_answer"] for output in outputs} == {"Final answer: 4"}
    assert all(output.extra_fields["workflow_success"] is True for output in outputs)


def test_matpo_synthetic_emitter_ids_are_unique_across_prompts():
    from trajweave.backends.verl.emitters.matpo import MATPOEmitterMixin
    from trajweave.backends.verl.emitters.registry import build_recipe_outputs

    class _Worker(MATPOEmitterMixin):
        config = {
            "agent": {
                "orchestra": {
                    "matpo": {
                        "planner_agent": "planner",
                        "worker_agent": "browsing_agent",
                        "tool_name": "search_and_browse",
                        "accuracy_reward_weight": 0.9,
                        "tool_format_reward_weight": 0.1,
                    }
                }
            }
        }

        @staticmethod
        def _encode_prompt(value):
            return [11]

        @staticmethod
        def _encode_text(value):
            return [len(value)]

    batches = []
    for uid in ("prompt-a", "prompt-b"):
        batches.extend(
            build_recipe_outputs(
                _Worker(),
                recipe="matpo_browse",
                use_hf_local=False,
                prompt={
                    "uid": uid,
                    "raw_prompt": [{"content": "capital France"}],
                    "reward_model": {"ground_truth": "Paris"},
                    "extra_info": {"search_query": "capital France"},
                },
                session_id=0,
            )
        )

    reqs_ids = [output.extra_fields["reqs_id"] for output in batches]
    assert len(reqs_ids) == len(set(reqs_ids))
    parents = {output.extra_fields["reqs_id"] for output in batches if not output.extra_fields["is_from_subagent_tool"]}
    for output in batches:
        if output.extra_fields["is_from_subagent_tool"]:
            assert output.extra_fields["parent_reqs_id"] in parents


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


def test_hf_marti_registry_preserves_stateful_tree_emitter():
    from types import SimpleNamespace

    from trajweave.backends.verl.emitters.registry import build_recipe_outputs

    calls = []
    expected = [object()]

    def emit(prompt, *, session_id):
        calls.append((prompt["uid"], session_id))
        return expected

    worker = SimpleNamespace(_build_hf_marti_mars2_outputs=emit)
    result = build_recipe_outputs(
        worker, recipe="marti_mars2_single_mcts", use_hf_local=True,
        prompt={"uid": "tree-1"}, session_id=2, validate=False,
    )
    assert result is expected
    assert calls == [("tree-1", 2)]
