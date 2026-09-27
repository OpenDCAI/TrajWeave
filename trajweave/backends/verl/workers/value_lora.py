from __future__ import annotations

from types import MethodType
from typing import Any

from verl.workers.engine_workers import TrainingWorker


class MARFTValueCriticTrainingWorker(TrainingWorker):
    """TrainingWorker whose value-model LoRA uses PEFT's token-classification contract."""

    def __init__(self, config: Any):
        super().__init__(config)
        _install_value_model_lora_builder(self.engine, model_config=self.model_config)


def _install_value_model_lora_builder(engine: Any, *, model_config: Any) -> None:
    lora_rank = int(getattr(model_config, "lora_rank", 0) or 0)
    if lora_rank <= 0:
        return
    if not hasattr(engine, "_build_lora_module"):
        raise ValueError(
            "MARFT independent_critic=lora requires an engine with value-model LoRA support; "
            f"got {type(engine).__name__}."
        )
    engine._build_lora_module = MethodType(_build_value_model_lora_module, engine)


def _build_value_model_lora_module(engine: Any, module: Any) -> Any:
    """Wrap a VERL value model without treating it as a causal language model."""

    from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model

    module.enable_input_require_grads()
    model_config = engine.model_config
    lora_adapter_path = getattr(model_config, "lora_adapter_path", None)
    if lora_adapter_path is not None:
        from verl.utils.fs import copy_to_local

        local_adapter_path = copy_to_local(lora_adapter_path, use_shm=model_config.use_shm)
        _validate_value_adapter_config(PeftConfig.from_pretrained(local_adapter_path))
        wrapped = PeftModel.from_pretrained(module, local_adapter_path, is_trainable=True)
        _validate_value_adapter_config(wrapped.peft_config["default"])
        return wrapped

    from verl.utils.py_functional import convert_to_regular_types

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=model_config.lora_rank,
        lora_alpha=model_config.lora_alpha,
        target_modules=convert_to_regular_types(model_config.target_modules),
        target_parameters=convert_to_regular_types(getattr(model_config, "target_parameters", None)),
        exclude_modules=convert_to_regular_types(getattr(model_config, "exclude_modules", None)),
        modules_to_save=["score"],
        bias="none",
    )
    return get_peft_model(module, lora_config)


def _validate_value_adapter_config(peft_config: Any) -> None:
    task_type = getattr(peft_config, "task_type", None)
    normalized_task = str(getattr(task_type, "value", task_type)).upper()
    if normalized_task not in {"TOKEN_CLS", "TOKEN_CLASSIFICATION"}:
        raise ValueError(
            f"MARFT critic LoRA adapters must use PEFT TaskType.TOKEN_CLS; adapter declares {task_type!r}."
        )


__all__ = ["MARFTValueCriticTrainingWorker"]
