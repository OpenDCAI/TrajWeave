"""从正式 recipe 模板生成独立、可直接运行的最小训练验收配置。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


# 每个方法使用真实 HF 模型；MARTI 另验 native vLLM，CoMLRL 按算法分别运行。
CASES = [
    ("drmas_math", "drmas/math_qwen05b_2gpu.yaml", 1, "math"),
    ("drmas_search", "drmas/search_qwen05b_2gpu.yaml", 1, "search"),
    ("maporl", "maporl/debate_math_multi_actor_qwen05b_2gpu.yaml", 2, "math"),
    ("agentflow", "agentflow/flow_grpo_qwen05b_2gpu.yaml", 1, "math"),
    ("gigpo", "gigpo/solver_verifier_math_qwen05b_2gpu.yaml", 1, "math"),
    ("comas", "comas/peer_review_math_qwen05b_2gpu.yaml", 2, "math"),
    ("atgrpo", "atgrpo/solver_verifier_math_qwen05b_2gpu.yaml", 1, "math"),
    ("matpo", "matpo/browse_qwen05b_1gpu.yaml", 1, "browse_qa"),
    ("mrlx", "mrlx/mgrpo_research_qa_2gpu.yaml", 2, "browse_qa"),
    ("wideseek_r1", "wideseek_r1/broad_search_qwen05b_1gpu.yaml", 1, "browse_qa"),
    ("marshal", "marshal/tictactoe_selfplay_qwen05b_1gpu.yaml", 1, "strategic_game"),
    ("marft", "marft/deepscaler_2agent_verl_tiny.yaml", 1, "math"),
    ("c3", "c3/reasoner_actor_math_verl_tiny.yaml", 3, "math"),
    ("marti_hf", "marti_mars2/stage1e_multi_agent_hf_smoke.yaml", 2, "controlled_code"),
    ("marti_vllm", "marti_mars2/stage1e_multi_agent_vllm_smoke.yaml", 2, "controlled_code"),
    *[(f"comlrl_{algorithm}", f"comlrl/{algorithm}_verl_tiny.yaml",
       4 if algorithm == "iac" else 3 if algorithm == "maac" else 2, "math")
      for algorithm in ("magrpo", "mareinforce", "marloo", "maremax", "iac", "maac",
                        "madpo", "marlhf", "madpo_iter", "marlhf_iter")],
]


def _replace_assets(value: Any, model: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"model_path", "tokenizer_path", "reward_model_name"}:
                value[key] = model
            elif key == "gpus":
                value[key] = 1
            else:
                _replace_assets(item, model)
    elif isinstance(value, list):
        for item in value:
            _replace_assets(item, model)


def _override(config: dict, key: str, value: Any, *, add: bool = False) -> None:
    items = config["verl"].setdefault("overrides", [])
    items[:] = [item for item in items if item.split("=", 1)[0].lstrip("+") != key]
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    items.append(f"{'+' if add else ''}{key}={encoded}")


def build_case(*, name: str, template: str, gpus: int, task: str, model: Path,
               output: Path, python: str, repo: Path) -> dict:
    config = yaml.safe_load((repo / "configs" / template).read_text())
    _replace_assets(config, str(model))
    case_dir = output / name
    dataset = case_dir / "dataset"
    config["mode"] = "verl_train"
    config["run"] = {"root_dir": str(case_dir / "runs"), "name": name, "tags": ["integration", "real-training"]}
    config.pop("export", None)
    config["prepare"] = {"verl_dataset": {
        "enabled": True, "output_dir": str(dataset), "task_family": task,
        "recipe_name": config["recipe"], "train_size": 4, "val_size": 2, "overwrite": True,
    }}
    for section in ("drmas_native", "maporl", "comas", "gigpo", "atgrpo", "matpo", "mrlx",
                    "wideseek_r1", "marshal", "marft", "c3", "comlrl", "marti_mars2"):
        if section not in config:
            continue
        settings = config[section]
        settings["agent_loop_backend"] = "vllm_marti_tq" if name == "marti_vllm" else "hf_local_tq"
        if "hf_local_dtype" in settings:
            settings["hf_local_dtype"] = "fp32"
    if "comas" in config:
        config["comas"].update(num_rounds=1, num_references=1)
    if "matpo" in config:
        config["matpo"]["max_turns"] = 2
    if "wideseek_r1" in config:
        config["wideseek_r1"]["max_parallel_subagents"] = 2
    if "comlrl" in config:
        section = config["comlrl"]
        section["max_turns"] = 2 if name.endswith(("_iac", "_maac")) else 1
        section["num_candidates"] = 1 if name.endswith(("_iac", "_maac")) else 4
        if "marlhf" in section:
            section["marlhf"].update(reward_num_train_epochs=1, reward_train_batch_size=1,
                                      preference_collection_batches=1, preference_num_candidates=4)
        if "iterative" in section:
            section["iterative"].update(
                num_iterations=1, num_train_epochs=1, num_target_candidates=4, pairs_per_sample=1,
            )
            section["iterative"]["comparator"]["num_candidates"] = 4
    if "marti_mars2" in config:
        config["marti_mars2"].update(max_num_nodes=4, initial_candidates=2)
        config["marti_mars2"]["worker_groups"] = {
            group: {"model_path": str(model), "tokenizer_path": str(model), "trainable": True, "gpus": 1}
            for group in dict.fromkeys(config["marti_mars2"]["model_ids"])
        }
    verl = config["verl"]
    verl.update(enabled=True, execute=True, python=python, cwd=str(repo))
    for key in ("command_file", "stdout_path", "stderr_path"):
        verl.pop(key, None)
    # 每个配置有独立 Ray 临时目录，NAS 保存模型、数据、日志和 checkpoint。
    # Ray 会附加 session 和 socket 名称，完整 Unix socket 路径不能超过 107 字节。
    ray_key = hashlib.sha256(str(case_dir.resolve()).encode()).hexdigest()[:12]
    ray_tmp = f"/tmp/tw-{ray_key}"
    verl["env"] = {
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo),
        "TOKENIZERS_PARALLELISM": "false", "RAY_DEDUP_LOGS": "0", "RAY_USAGE_STATS_ENABLED": "0",
        "RAY_ADDRESS": "local", "RAY_TMPDIR": ray_tmp, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1", "NCCL_IB_DISABLE": "1", "NCCL_P2P_DISABLE": "1",
        "NO_PROXY": "localhost,127.0.0.1,::1", "WANDB_MODE": "disabled",
    }
    common = {
        "trainer.use_v1": True, "trainer.nnodes": 1, "trainer.n_gpus_per_node": gpus,
        "trainer.total_epochs": 2, "trainer.total_training_steps": 2, "trainer.save_freq": 2,
        "trainer.test_freq": 2, "trainer.val_before_train": False, "trainer.logger": ["console"],
        "trainer.resume_mode": "disable", "trainer.experiment_name": name,
        "data.train_files": str(dataset / "train.jsonl"), "data.val_files": str(dataset / "val.jsonl"),
        "data.train_batch_size": 2, "data.val_batch_size": 1, "data.max_prompt_length": 512,
        "data.max_response_length": 96, "data.dataloader_num_workers": 0,
        "data.filter_overlong_prompts": False, "data.truncation": "right", "data.shuffle": False,
        "actor_rollout_ref.model.path": str(model), "actor_rollout_ref.model.tokenizer_path": str(model),
        "actor_rollout_ref.model.use_remove_padding": False,
        "actor_rollout_ref.model.enable_gradient_checkpointing": False,
        "actor_rollout_ref.rollout.name": "vllm" if name == "marti_vllm" else "hf",
        "actor_rollout_ref.rollout.prompt_length": 512, "actor_rollout_ref.rollout.response_length": 96,
        "actor_rollout_ref.rollout.n": 4 if name == "atgrpo" else 2,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 1024,
        "actor_rollout_ref.rollout.calculate_log_probs": name == "marti_vllm",
        "actor_rollout_ref.rollout.agent.num_workers": 1,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 2,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 1024,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.actor.optim.lr": 1e-5,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model"],
        "reward.num_workers": 1, "transfer_queue.enable": True,
        "transfer_queue.backend.SimpleStorage.num_data_storage_units": 2,
        "ray_kwargs.ray_init.num_cpus": 24,
    }
    for key, value in common.items():
        _override(config, key, value)
    _override(config, "actor_rollout_ref.model.override_config.attn_implementation", "eager", add=True)
    _override(config, "ray_kwargs.ray_init._temp_dir", ray_tmp, add=True)
    _override(config, "ray_kwargs.ray_init.include_dashboard", False, add=True)
    if name in {"maporl", "marft", "c3", "comlrl_iac", "comlrl_maac"}:
        for key, value in {
            "critic.model.path": str(model), "critic.model.tokenizer_path": str(model),
            "critic.model.use_remove_padding": False, "critic.model.enable_gradient_checkpointing": False,
            "critic.ppo_micro_batch_size_per_gpu": 1,
            "critic.ppo_max_token_len_per_gpu": 1024,
            "critic.forward_max_token_len_per_gpu": 1024,
        }.items():
            _override(config, key, value)
        _override(config, "critic.model.override_config.attn_implementation", "eager", add=True)
    if name == "marti_vllm":
        _override(config, "actor_rollout_ref.rollout.max_model_len", 1024)
        _override(config, "actor_rollout_ref.rollout.max_num_batched_tokens", 1024)
        _override(config, "actor_rollout_ref.rollout.data_parallel_size", 1)
        _override(config, "actor_rollout_ref.rollout.enforce_eager", True)
    if name.startswith("marti_"):
        # 代码函数需要完整生成；96 token 会把小模型的函数体截断。
        _override(config, "data.max_response_length", 256)
        _override(config, "actor_rollout_ref.rollout.response_length", 256)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--agentflow-model", type=Path, help="可选：为 Planner 使用更适合结构化输出的独立小模型")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    model, output = args.model.resolve(), args.output_dir.resolve()
    if not (model / "config.json").is_file():
        parser.error("--model 必须指向完整的本地 Hugging Face 模型目录")
    agentflow_model = args.agentflow_model.resolve() if args.agentflow_model else model
    if not (agentflow_model / "config.json").is_file():
        parser.error("--agentflow-model 必须指向完整的本地 Hugging Face 模型目录")
    unknown = set(args.only or ()) - {case[0] for case in CASES}
    if unknown:
        parser.error(f"未知验收项：{sorted(unknown)}")
    config_dir = output / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, template, gpus, task in CASES:
        if args.only and name not in args.only:
            continue
        case_model = agentflow_model if name == "agentflow" else model
        config = build_case(name=name, template=template, gpus=gpus, task=task,
                            model=case_model, output=output, python=args.python, repo=repo)
        model_type = json.loads((case_model / "config.json").read_text()).get("model_type", "")
        if model_type.startswith("qwen3"):
            _override(config, "data.apply_chat_template_kwargs.enable_thinking", False, add=True)
        path = config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        manifest.append({"name": name, "recipe": config["recipe"], "gpus": gpus,
                         "config": str(path), "source_template": template,
                         "command": [args.python, "-m", "trajweave.cli.run", "--config", str(path)]})
    (output / "suite.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(manifest), "manifest": str(output / "suite.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
