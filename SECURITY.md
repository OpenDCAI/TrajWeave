# Security

TrajWeave is a research training framework. Run it with the privileges and network access appropriate to the datasets, model code, and generated programs you are evaluating.

## Trust boundaries

- Code evaluation executes model-generated Python in subprocesses. Timeouts and reliability guards are not a security sandbox. Use a disposable container or VM, restricted filesystem mounts, no credentials, and appropriate network restrictions for untrusted code.
- Some model/tokenizer loaders enable `trust_remote_code=True`, including the direct HF backend and tokenizer compatibility checks. Treat model loading as potential code execution: use only reviewed sources and isolate the process when necessary.
- PyTorch checkpoints may use pickle-based serialization. Do not load checkpoints from untrusted sources.
- Run artifacts may contain prompts, generated text, tool output, model paths, and dataset metadata. Review artifacts before sharing them; log redaction is not a complete data-loss-prevention system.
- Configure credentials through the environment or an external secret store. Do not commit environment files, tokens, model checkpoints, or private dataset records.

## Reporting a vulnerability

Do not post credentials, exploit details, or private training data in a public issue. Use the repository's GitHub private vulnerability reporting channel when available. Otherwise, contact an OpenDCAI repository maintainer privately through their published contact information before sharing sensitive details.

If a credential was committed, revoke or rotate it with its provider first. Removing it from the current tree does not remove it from earlier commits, other branches, forks, or caches. Coordinate any history rewrite with repository maintainers; do not force-push unrelated branches as part of a routine release.

## Release checks

Run `python scripts/check_repo_hygiene.py` before publishing changes. It checks current tracked and non-ignored files for machine-specific paths, selected credential markers, oversized files, and private/runtime artifacts. This is a lightweight guard, not a comprehensive secret scanner or an audit of Git history and dependencies.
