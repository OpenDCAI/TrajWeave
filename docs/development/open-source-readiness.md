# Open-source readiness review

Review dates: **2026-09-27–28**. Base: `lz-dev` at `56b562d`.

## Scope and verdict

The review inventoried all 1,028 tracked files at the starting revision, inspected configuration portability, packaging and attribution, publication documents, runtime artifacts, and credential markers, and separately scanned reachable Git history.

**The current release tree is ready for review with the changes below. This is not a security certification for every branch or historical Git object.** Repository history requires its own remediation process; publishing a cleaned tree does not remove data from prior commits.

## Findings addressed

| Finding                             | Effect                             | Change                                    |
| ----------------------------------- | ---------------------------------- | ----------------------------------------- |
| Personal model/runtime paths        | Presets failed on other machines   | Public defaults and environment overrides |
| Preset GPU masks                    | Overrode caller allocation         | Inherit device selection from the caller  |
| Unresolved role model expressions   | Tokenizer checks saw placeholders  | Resolve model fields before role checks   |
| README mixed with development diary | Unclear setup and validation scope | English overview and focused guides       |
| Wheel omitted third-party notices   | Package lost source attribution    | Include Notice.txt and license files      |
| Missing local-file ignore rules     | Accidental publication risk        | Ignore worktrees, private env/data/models |
| No publication hygiene guard        | Path/artifact regressions possible | Scanner and pre-commit hook               |
| Unclear code-execution boundary     | Timeout mistaken for sandbox       | Document execution and model trust        |

The previous README is retained in the [development archive](README-legacy.md), with private paths sanitized and relative links adjusted. Existing license and copyright notices remain intact. The three new README illustrations have [recorded generation prompts](../../assets/readme/generation.json) and are labeled conceptual artwork.

## Reproducibility and verification

**Release checks on 2026-09-28:** the joint CPU suite passed **962 tests** (70 warnings, 270.44 seconds); the current-tree guard checked **1,043 files** with no findings. Backend compilation, changed-file Ruff checks, the documented CPU smoke command, local documentation links, browser layout/images, and wheel/source-archive builds passed. Package contents include the main license, Notice.txt, and all four third-party license files.

Five targeted regression tests cover environment defaults, missing required variables, non-mutating configuration resolution, heterogeneous-role tokenizer/Hydra paths containing Unicode and punctuation, and preservation of caller GPU selection.

The CPU and GPU constraints record the core integration stacks. They are not complete dependency locks, and fresh installations on every supported platform have not been tested. Source-checkout installation remains the documented workflow.

The [historical GPU validation](../validation.md) remains separate from this publication review. No new GPU result is implied by a documentation or portability check.

Run the current-tree guard with:

```bash
python scripts/check_repo_hygiene.py
```

It checks tracked and non-ignored files for known credential markers, machine-specific path forms, large files, symlinks requiring review, runtime artifacts, and required release documents. It prints locations and rule names without printing secret values. Generic examples such as `/path/to/model` and system temporary paths are allowed.

## Remaining limits

- Pattern scanning is not exhaustive secret detection, and the current-tree guard does not rewrite or certify Git history.
- Generated code, remote model code, serialized checkpoints, and training configurations are trusted inputs unless isolated externally.
- Most smoke recipes validate HF generation plus GPU optimization; the native vLLM result covers the documented MARTI case.
- AgentFlow's tiny-batch run had zero gradients. Resume, asynchronous operation, large-scale performance, and benchmark reproduction remain subject to the limits in the validation record.
- Changes to historical branches, credential rotation, and any destructive history rewrite require separate coordination; none is performed by this release.
