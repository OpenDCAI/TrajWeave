# Getting started

TrajWeave currently supports source-checkout workflows on Linux. Use a dedicated environment for this project. The CPU and GPU constraints below describe the stacks used during integration; they are not a lockfile for every transitive dependency.

## CPU: inspect a protocol

From the repository root:

```bash
conda create -n trajweave-cpu python=3.11 -y
conda activate trajweave-cpu
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[math,test]' -c requirements/cpu-smoke-constraints.txt
python -m trajweave.cli.run --config configs/drmas/math_smoke.yaml
```

This deterministic smoke configuration checks agent turns, rewards, trajectory records, and run artifacts. It does **not** optimize a language model.

| Mode         | Meaning                                                   |
| ------------ | --------------------------------------------------------- |
| `smoke`      | Execute the recipe's lightweight protocol fixture         |
| `verl_plan`  | Prepare assets and inspect the generated training command |
| `verl_train` | Launch the configured VERL trainer                        |

A successful plan is not evidence of successful training. Some plan presets still prepare assets or load tokenizers.

## GPU: train a small real model

The verified integration environment used Python 3.11, CUDA 12.8 PyTorch wheels, vLLM 0.12.0, and H20 GPUs. Other CUDA, GPU, or package combinations need their own validation. Install into an idle project environment:

```bash
conda create -n trajweave-gpu python=3.11 -y
conda activate trajweave-gpu
python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[math,vllm]' -c requirements/gpu-smoke-constraints.txt
export PYTHONNOUSERSITE=1
```

Supply a local Hugging Face model directory containing model weights, configuration, and tokenizer files. The suite generator does not download models.

```bash
python -m trajweave.cli.prepare_smoke_suite \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --agentflow-model /path/to/Qwen3-0.6B \
  --output-dir outputs/smoke-suite

CUDA_VISIBLE_DEVICES=0 python -m trajweave.cli.run \
  --config outputs/smoke-suite/configs/atgrpo.yaml
```

The optional AgentFlow model override was used because the 0.5B planner frequently produced invalid formats. Qwen3 thinking is disabled by the generator for the short response budget. Valid output alone does not guarantee a nonzero learning signal.

Use `--only atgrpo matpo` to generate a subset. Review `outputs/smoke-suite/suite.json` before running: cases need **one to four GPUs**. Set `CUDA_VISIBLE_DEVICES` to the required number of available devices. Presets preserve the caller's GPU selection. Give concurrent runs separate suite directories and non-overlapping devices.

Most suite cases generate with HF on the CPU and optimize on GPUs. The `marti_vllm` case also tests native vLLM generation on GPUs. Do not treat these cases as throughput benchmarks.

## MARTI code scoring

The strict PRIME scorer needs the legacy `pyext` dependency. The helper downloads a hash-checked pyext 0.6 archive, extracts the module, adapts its Python 3.11 compatibility, and records provenance:

```bash
python -m trajweave.cli.prepare_prime_dependency
```

Install only while the environment is idle. Alternatively, pass `--target /path/to/dependencies` and include that directory in the MARTI configuration's `verl.env.PYTHONPATH`. Without the dependency, scoring reports a local fallback and strict PRIME acceptance fails.

Generated code executes in subprocesses. Use a disposable, isolated execution environment for untrusted programs; see [security boundaries](../SECURITY.md).

## Model paths in presets

Portable Qwen presets accept environment overrides and otherwise use public Hugging Face model IDs:

| Variable                            | Default                       |
| ----------------------------------- | ----------------------------- |
| `TRAJWEAVE_QWEN05B_INSTRUCT_PATH`   | `Qwen/Qwen2.5-0.5B-Instruct`  |
| `TRAJWEAVE_QWEN05B_BASE_PATH`       | `Qwen/Qwen2.5-0.5B`           |
| `TRAJWEAVE_QWEN15B_INSTRUCT_PATH`   | `Qwen/Qwen2.5-1.5B-Instruct`  |

For example:

```bash
export TRAJWEAVE_QWEN05B_INSTRUCT_PATH=/path/to/Qwen2.5-0.5B-Instruct
```

Role model references resolve before tokenizer compatibility checks. Hydra overrides resolve in the training process, after runtime directories are available. Heterogeneous model presets still require compatible tokenizers; setting a path does not bypass that check.

Use the suite generator for the maintained minimal training setup. Individual source presets may target planning, synthetic fixtures, or different resource layouts.

## Inspect a run

Read the run status, summary, metrics, trajectories, and checkpoints together. Verify that optimizer steps actually ran and that losses and gradients are finite. For methods expected to learn on a batch, check nonzero gradients and parameter updates. Exit code zero by itself is insufficient.

The [recipe catalog](recipes.md) lists case names and GPU counts. The [validation record](validation.md) explains what was measured and what remains unverified.
