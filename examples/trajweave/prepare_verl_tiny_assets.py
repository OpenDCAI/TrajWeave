from __future__ import annotations

import argparse
import json

from trajweave.backends.verl.tiny_assets import prepare_tiny_verl_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local tiny VERL smoke-test assets.")
    parser.add_argument("--output-dir", default="outputs/doctor_mas_verl_tiny_assets")
    parser.add_argument("--train-size", type=int, default=2)
    parser.add_argument("--val-size", type=int, default=2)
    parser.add_argument("--task-family", choices=("math", "search"), default="math")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    result = prepare_tiny_verl_assets(
        args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        task_family=args.task_family,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
