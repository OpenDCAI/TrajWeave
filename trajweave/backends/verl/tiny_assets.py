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
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "system": 3,
        "user": 4,
        "assistant": 5,
        ":": 6,
        "Question": 7,
        "Answer": 8,
        "only": 9,
        ".": 10,
        "What": 11,
        "is": 12,
        "+": 13,
        "?": 14,
        "1": 15,
        "2": 16,
        "3": 17,
        "4": 18,
        "5": 19,
        "6": 20,
        "7": 21,
        "8": 22,
        "9": 23,
        "10": 24,
        "Final": 25,
        "answer": 26,
        "APPROVED": 27,
        "SEARCH": 28,
        "Paris": 29,
        "Guido": 30,
        "van": 31,
        "Rossum": 32,
        "Implement": 33,
        "add_one": 34,
        "square": 35,
        "x": 36,
        "return": 37,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<eos>",
        model_max_length=128,
    )
    fast_tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}assistant:"
    )

    config = Qwen2Config(
        vocab_size=len(vocab),
        max_position_embeddings=128,
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


def _code_rows(size: int, *, split: str, recipe_name: str | None = None) -> list[dict[str, Any]]:
    seeds = [
        (
            "Implement add_one(x).",
            "return x + 1",
            {"fn_name": "add_one", "inputs": ["1", "2", "5"], "outputs": ["2", "3", "6"]},
        ),
        (
            "Implement square(x).",
            "return x * x",
            {"fn_name": "square", "inputs": ["1", "2", "5"], "outputs": ["1", "4", "25"]},
        ),
    ]
    rows = []
    for index in range(size):
        prompt, answer, test_cases = seeds[index % len(seeds)]
        rows.append(
            {
                "data_source": "trajweave_tiny_code",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "code",
                "reward_model": {"style": "code", "ground_truth": answer, "test_cases": test_cases},
                "extra_info": {
                    "index": index,
                    "split": split,
                    "trajweave_recipe": recipe_name or "marti_mars2_single_mcts",
                },
            }
        )
    return rows


def _controlled_code_rows(size: int, *, split: str, recipe_name: str | None = None) -> list[dict[str, Any]]:
    prompt = (
        "Write a complete Python function named zigzag_code for an integer x. Return -1 if x is negative. "
        "Otherwise return 6 if x is divisible by 6, else 3 if divisible by 3, else 2 if divisible by 2, "
        "else 1. Return only executable Python code. Required signature: def zigzag_code(x):"
    )
    test_cases = {
        "fn_name": "zigzag_code",
        "inputs": ["-2", "0", "1", "2", "3", "4", "6", "9", "12"],
        "outputs": ["-1", "6", "1", "2", "3", "2", "6", "3", "6"],
    }
    return [
        {
            "data_source": "trajweave_controlled_code",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "code",
            "reward_model": {
                "style": "code",
                "ground_truth": "return min(10, max(0, x))",
                "test_cases": test_cases,
            },
            "extra_info": {
                "index": index,
                "split": split,
                "trajweave_recipe": recipe_name or "marti_mars2_single_mcts_fidelity",
                "acceptance_target": "mixed_verifier_rewards_within_tree",
            },
        }
        for index in range(size)
    ]


def _rows(size: int, *, split: str, task_family: str, recipe_name: str | None = None) -> list[dict[str, Any]]:
    if task_family == "math":
        return _math_rows(size, split=split, recipe_name=recipe_name)
    if task_family == "search":
        return _search_rows(size, split=split, recipe_name=recipe_name)
    if task_family == "code":
        return _code_rows(size, split=split, recipe_name=recipe_name)
    if task_family == "controlled_code":
        return _controlled_code_rows(size, split=split, recipe_name=recipe_name)
    raise ValueError(f"Unknown tiny VERL task_family: {task_family}")


def _default_recipe_name(task_family: str) -> str:
    if task_family == "math":
        return "doctor_mas_math"
    if task_family == "search":
        return "doctor_mas_search"
    if task_family == "code":
        return "marti_mars2_single_mcts"
    if task_family == "controlled_code":
        return "marti_mars2_single_mcts_fidelity"
    return task_family


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
