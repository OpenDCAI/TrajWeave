from __future__ import annotations

import argparse

from trajweave.runner import dumps_result, result_exit_code, run_from_config_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a TrajWeave recipe from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a TrajWeave YAML config.")
    args = parser.parse_args(argv)
    result = run_from_config_path(args.config)
    print(dumps_result(result))
    return result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
