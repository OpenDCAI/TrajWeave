#!/usr/bin/env python3
"""检查当前发布文件的可移植性与泄露风险；不会输出疑似密钥原文。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PRIVATE_PATH = re.compile(
    r"/(?:mnt/(?:bn|data)/|data/workspace/|home/[\w.-]+/|Users/[\w.-]+/|root/|workspace/[\w.-]+/)"
)
SECRET = re.compile(
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    r"gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|"
    r"AKIA[A-Z0-9]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{35,}"
)
RUNTIME_SUFFIXES = {".pt", ".ckpt", ".safetensors", ".pyc", ".log", ".p12", ".pem", ".key"}
MAX_BYTES = 5 * 1024 * 1024


def inspect_file(path: Path, name: str) -> list[dict]:
    issues = []
    if path.is_symlink():
        return [{"path": name, "rule": "symlink_requires_review"}]
    if (
        path.suffix in RUNTIME_SUFFIXES
        or path.name == ".env"
        or path.name.startswith(".env.")
        and path.name != ".env.example"
    ):
        issues.append({"path": name, "rule": "private_or_runtime_artifact"})
    if path.stat().st_size > MAX_BYTES:
        issues.append({"path": name, "rule": "oversized_file", "bytes": path.stat().st_size})
        return issues
    data = path.read_bytes()
    if b"\x00" in data:
        return issues
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return issues
    for number, line in enumerate(content.splitlines(), 1):
        for rule, pattern in (("machine_specific_path", PRIVATE_PATH), ("credential_marker", SECRET)):
            if pattern.search(line):
                issues.append({"path": name, "line": number, "rule": rule})
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
    )
    names = sorted(set(item.decode() for item in result.stdout.split(b"\0") if item))
    issues = []
    checked = 0
    for name in names:
        path = repo / name
        if not path.exists() and not path.is_symlink():
            continue
        checked += 1
        issues.extend(inspect_file(path, name))
    for required in ("LICENSE", "Notice.txt", "README.md", "SECURITY.md", "CONTRIBUTING.md"):
        if not (repo / required).is_file():
            issues.append({"path": required, "rule": "missing_release_document"})
    print(
        json.dumps(
            {
                "scope": "current tracked and non-ignored files; Git history is separate",
                "files_checked": checked,
                "issues": issues,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
