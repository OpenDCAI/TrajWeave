from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prepare_tiny_verl_assets(
    output_dir: str | Path = "outputs/doctor_mas_verl_tiny_assets",
    *,
    train_size: int = 2,
    val_size: int = 2,
    task_family: str = "math",
    recipe_name: str | None = None,
    overwrite: bool = True,
) -> dict[str, str]:
    output_path = Path(output_dir)
    model_path = output_path / "model"
    output_path.mkdir(parents=True, exist_ok=True)

    if overwrite or not (model_path / "config.json").exists():
        _write_tiny_model(model_path)
    dataset = prepare_verl_dataset(
        output_dir=output_path,
        train_size=train_size,
        val_size=val_size,
        task_family=task_family,
        recipe_name=recipe_name,
        overwrite=overwrite,
    )
    return {
        **dataset,
        "model_path": str(model_path),
    }


def prepare_verl_dataset(
    output_dir: str | Path,
    *,
    train_size: int = 2,
    val_size: int = 2,
    task_family: str = "math",
    recipe_name: str | None = None,
    overwrite: bool = True,
) -> dict[str, str]:
    """只生成 VERL 训练数据，不创建或修改模型资产。"""

    output_path = Path(output_dir)
    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"
    output_path.mkdir(parents=True, exist_ok=True)
    if overwrite or not train_path.exists():
        _write_jsonl(train_path, _rows(train_size, split="train", task_family=task_family, recipe_name=recipe_name))
    if overwrite or not val_path.exists():
        _write_jsonl(val_path, _rows(val_size, split="val", task_family=task_family, recipe_name=recipe_name))
    return {
        "output_dir": str(output_path),
        "train_file": str(train_path),
        "val_file": str(val_path),
        "task_family": task_family,
        "recipe_name": recipe_name or _default_recipe_name(task_family),
    }


def _write_tiny_model(model_path: Path) -> None:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        (
            "system user assistant Question Answer only What is the capital of France Japan Canada",
            "Who created the Python programming language Guido van Rossum Paris Tokyo Ottawa",
            "Final answer APPROVED SEARCH",
            "MrlX main explorer adapter query evidence",
            "CALL sub_adapter: capital France",
            "CALL search_and_browse: capital France",
            "Research result: France capital Paris",
            "Final answer: Paris",
            "Tic-Tac-Toe player_0 player_1 Board Legal actions Game history",
            "Return exactly <answer>0</answer> <answer>1</answer> <answer>2</answer>",
            "1 2 3 4 5 6 7 8 9 10 + - * ? . :",
        ),
        trainer=BpeTrainer(
            vocab_size=512,
            min_frequency=1,
            special_tokens=["<pad>", "<eos>", "<unk>"],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<eos>",
        model_max_length=256,
    )
    fast_tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"
    )

    config = Qwen2Config(
        vocab_size=len(fast_tokenizer),
        max_position_embeddings=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        bos_token_id=fast_tokenizer.bos_token_id,
        eos_token_id=fast_tokenizer.eos_token_id,
        pad_token_id=fast_tokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    config.attn_implementation = "eager"
    config._attn_implementation = "eager"
    # Keep generated tiny assets reproducible without changing the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(config)
    model_path.mkdir(parents=True, exist_ok=True)
    fast_tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)


def _math_rows(size: int, *, split: str, recipe_name: str | None = None) -> list[dict[str, Any]]:
    # 前两题刻意处于 0.5B 模型的能力边界，真实采样可产生组内 reward 差异。
    seeds = [
        (73, "*", 24),
        (43, "*", 27),
        (29, "*", 47),
        (48, "*", 32),
        (137, "+", 286),
        (1001, "-", 497),
    ]
    rows = []
    for index in range(size):
        left, operator, right = seeds[index % len(seeds)]
        if operator == "+":
            answer = str(left + right)
        elif operator == "-":
            answer = str(left - right)
        else:
            answer = str(left * right)
        rows.append(
            {
                "data_source": "trajweave_tiny_math",
                "prompt": [
                    {
                        "role": "user",
                        "content": f"What is {left} {operator} {right}? Answer only.",
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": index,
                    "split": split,
                    "trajweave_recipe": recipe_name or "doctor_mas_math",
                },
            }
        )
    return rows


def _search_rows(size: int, *, split: str, recipe_name: str | None = None) -> list[dict[str, Any]]:
    seeds = [
        (
            "Which city is the capital of France?",
            "Paris",
            "capital France",
            [
                {"title": "France", "text": "France is a country in Europe. Its capital city is Paris."},
                {"title": "Germany", "text": "Germany's capital city is Berlin."},
            ],
        ),
        (
            "Who created the Python programming language?",
            "Guido van Rossum",
            "Python programming language creator",
            [
                {"title": "Python", "text": "Python was created by Guido van Rossum and first released in 1991."},
                {"title": "Java", "text": "Java was originally developed by James Gosling."},
            ],
        ),
    ]
    rows = []
    for index in range(size):
        question, answer, query, documents = seeds[index % len(seeds)]
        rows.append(
            {
                "data_source": "trajweave_tiny_search",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "search",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": index,
                    "split": split,
                    "trajweave_recipe": recipe_name or "doctor_mas_search",
                    "search_query": query,
                    "documents": documents,
                },
            }
        )
    return rows


def _browse_qa_rows(size: int, *, split: str, recipe_name: str | None = None) -> list[dict[str, Any]]:
    seeds = [
        (
            "What is the capital of France?",
            "Paris",
            "capital France",
            [
                {"title": "France", "text": "France is a country in Europe. Its capital city is Paris."},
                {"title": "Germany", "text": "Germany's capital city is Berlin."},
            ],
        ),
        (
            "What is the capital of Japan?",
            "Tokyo",
            "capital Japan",
            [
                {"title": "Japan", "text": "Japan is an island country in East Asia. Its capital is Tokyo."},
                {"title": "South Korea", "text": "South Korea's capital is Seoul."},
            ],
        ),
        (
            "Who created the Python programming language?",
            "Guido van Rossum",
            "Python programming language creator",
            [
                {"title": "Python", "text": "Python was created by Guido van Rossum and first released in 1991."},
                {"title": "Java", "text": "Java was originally developed by James Gosling."},
            ],
        ),
        (
            "What is the capital of Canada?",
            "Ottawa",
            "capital Canada",
            [
                {"title": "Canada", "text": "Canada is a country in North America. Its capital city is Ottawa."},
                {"title": "Australia", "text": "Australia's capital city is Canberra."},
            ],
        ),
    ]
    rows = []
    for index in range(size):
        question, answer, query, documents = seeds[index % len(seeds)]
        rows.append(
            {
                "data_source": "trajweave_tiny_browse_qa",
                "prompt": [
                    {
                        "role": "system",
                        "content": _browse_system_prompt(recipe_name),
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "browse_qa",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": index,
                    "split": split,
                    "trajweave_recipe": recipe_name or "matpo_browse",
                    "search_query": query,
                    "documents": documents,
                },
            }
        )
    return rows


def _rows(size: int, *, split: str, task_family: str, recipe_name: str | None = None) -> list[dict[str, Any]]:
    if task_family == "math":
        return _math_rows(size, split=split, recipe_name=recipe_name)
    if task_family == "search":
        return _search_rows(size, split=split, recipe_name=recipe_name)
    if task_family == "browse_qa":
        return _browse_qa_rows(size, split=split, recipe_name=recipe_name)
    if task_family == "strategic_game":
        return _strategic_game_rows(size, split=split, recipe_name=recipe_name)
    raise ValueError(f"Unknown tiny VERL task_family: {task_family}")


def _default_recipe_name(task_family: str) -> str:
    if task_family == "math":
        return "doctor_mas_math"
    if task_family == "search":
        return "doctor_mas_search"
    if task_family == "browse_qa":
        return "matpo_browse"
    if task_family == "strategic_game":
        return "marshal_tictactoe_selfplay"
    return task_family


def _strategic_game_rows(size: int, *, split: str, recipe_name: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for index in range(size):
        rows.append(
            {
                "data_source": "trajweave_tiny_strategic_game",
                "prompt": [
                    {
                        "role": "system",
                        "content": "Play both sides of Tic-Tac-Toe through MARSHAL self-play.",
                    },
                    {
                        "role": "user",
                        "content": f"Start self-play game {index}. Choose only legal actions.",
                    },
                ],
                "ability": "strategic_game",
                "reward_model": {"style": "rule", "ground_truth": "self_play"},
                "extra_info": {
                    "index": index,
                    "split": split,
                    "trajweave_recipe": recipe_name or "marshal_tictactoe_selfplay",
                    "game": "tictactoe",
                    "strategy": "player0_win" if index % 2 == 0 else "player1_win",
                },
            }
        )
    return rows


def _browse_system_prompt(recipe_name: str | None) -> str:
    normalized_recipe = str(recipe_name or "").strip().lower()
    if normalized_recipe.startswith("mrlx") or normalized_recipe == "m-grpo":
        return (
            "You are the MrlX main explorer. Delegate research with "
            "'CALL sub_adapter: <query>' and finish with 'Final answer: <answer>'."
        )
    return "You are a MATPO planner. You may call a browsing agent for focused factual subtasks."


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
