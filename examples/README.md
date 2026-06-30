# TrajWeave Examples

This directory is reserved for runnable TrajWeave examples.

The upstream VERL example matrix was removed to keep this repository focused. Add examples back only when they are part of the TrajWeave MASRL path.

Current examples:

```text
trajweave/configs/doctor_mas_math_smoke.yaml
trajweave/configs/doctor_mas_search_smoke.yaml
trajweave/configs/doctor_mas_math_tiny_train.yaml
trajweave/configs/doctor_mas_math_verl_export.yaml
trajweave/configs/doctor_mas_math_verl_agent_loop_dryrun.yaml
trajweave/configs/doctor_mas_math_hf_gpu_smoke.yaml
```

Run a config with:

```bash
PYTHONPATH=. python3 -m trajweave.cli.run --config examples/trajweave/configs/doctor_mas_math_smoke.yaml
```
