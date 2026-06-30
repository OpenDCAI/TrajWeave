from __future__ import annotations

import argparse

from trajweave.runner import dumps_result, run_from_config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TrajWeave recipe from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a TrajWeave YAML config.")
    args = parser.parse_args()
    print(dumps_result(run_from_config_path(args.config)))


if __name__ == "__main__":
    main()
