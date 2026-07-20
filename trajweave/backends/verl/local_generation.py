from __future__ import annotations

import gc
import logging
from contextlib import nullcontext
from typing import Any

import torch

from trajweave.backends.verl.runtime_config import TrajWeaveAgentLoopRuntimeConfig, config_get
from trajweave.backends.verl.schema import flatten_token_ids, to_python
from verl.utils.chat_template import apply_chat_template

logger = logging.getLogger(__name__)


class HFLocalGenerationMixin:
    def _encode_prompt(self, raw_prompt: Any) -> list[int]:
        try:
            token_ids = apply_chat_template(
                self.tokenizer,
                raw_prompt,
                add_generation_prompt=True,
                tokenize=True,
                **self.config.data.get("apply_chat_template_kwargs", {}),
            )
        except Exception:
            token_ids = self._encode_text(str(raw_prompt))
        token_ids = flatten_token_ids(token_ids)
        return token_ids[-self.rollout_config.prompt_length :]

    def _encode_text(self, text: str) -> list[int]:
        try:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            token_ids = self.tokenizer.encode(text)
        token_ids = flatten_token_ids(token_ids)
        if not token_ids:
            token_ids = [self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0]
        return token_ids[: self.rollout_config.response_length]

    def _encode_prompt_text(self, text: str) -> list[int]:
        return self._encode_chat_prompt(self.tokenizer, text)

    def _encode_chat_prompt(self, tokenizer: Any, text: str) -> list[int]:
        messages = [{"role": "user", "content": text}]
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
        except Exception:
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                token_ids = tokenizer.encode(text)
        token_ids = flatten_token_ids(token_ids)
        if not token_ids:
            token_ids = [tokenizer.eos_token_id or tokenizer.pad_token_id or 0]
        return token_ids[-self.rollout_config.prompt_length :]

    def _decode_response_ids(self, response_ids: list[int]) -> str:
        try:
            return str(self.tokenizer.decode(response_ids, skip_special_tokens=True))
        except TypeError:
            return str(self.tokenizer.decode(response_ids))

    def _generate_local_response_ids(
        self,
        prompt_ids: list[int],
        *,
        policy_group: str = "shared",
        sample_seed: int | None = None,
        validate: bool = False,
        prompt_text: str | None = None,
    ) -> list[int]:
        model = self._local_model(policy_group=policy_group)
        tokenizer = self._local_tokenizer(policy_group=policy_group)
        device = next(model.parameters()).device
        if prompt_text is not None and tokenizer is not self.tokenizer:
            input_ids = self._encode_chat_prompt(tokenizer, prompt_text)
        else:
            input_ids = self._local_prompt_ids(prompt_ids, policy_group=policy_group, tokenizer=tokenizer)
        input_ids = torch.tensor([input_ids[-self.rollout_config.prompt_length :]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        temperature = float(config_get(self.rollout_config, "temperature", 1.0))
        top_p = float(config_get(self.rollout_config, "top_p", 1.0))
        top_k = int(config_get(self.rollout_config, "top_k", -1))
        if validate:
            val_kwargs = config_get(self.rollout_config, "val_kwargs", {}) or {}
            temperature = float(config_get(val_kwargs, "temperature", 0.0))
            top_p = float(config_get(val_kwargs, "top_p", 1.0))
            top_k = int(config_get(val_kwargs, "top_k", -1))
        do_sample = temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.rollout_config.response_length,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p})
            if top_k > 0:
                generation_kwargs["top_k"] = top_k

        devices = []
        if device.type == "cuda" and device.index is not None:
            devices = [device.index]
        seed_context = torch.random.fork_rng(devices=devices) if sample_seed is not None else nullcontext()
        with torch.no_grad(), seed_context:
            if sample_seed is not None:
                torch.manual_seed(int(sample_seed))
            sequences = model.generate(**generation_kwargs)
        group_response_ids = sequences[0, input_ids.shape[-1] :].detach().cpu().tolist()
        if not group_response_ids:
            return [self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0]
        if tokenizer is self.tokenizer:
            response_ids = group_response_ids
        else:
            response_text = tokenizer.decode(group_response_ids, skip_special_tokens=True)
            response_ids = self._encode_text(response_text)
        return [int(token_id) for token_id in response_ids[: self.rollout_config.response_length]]

    def _local_model(self, *, policy_group: str = "shared"):
        if not hasattr(self, "_trajweave_local_models"):
            self._trajweave_local_models = {}
        cache: dict[str, Any] = self._trajweave_local_models
        if policy_group in cache:
            model = cache.pop(policy_group)
            cache[policy_group] = model
            return model
        from transformers import AutoModelForCausalLM

        model_path = self._worker_group_model_path(policy_group) or self.model_config.local_path
        runtime = TrajWeaveAgentLoopRuntimeConfig.from_verl_config(self.config)
        evicted_groups = _evict_local_model_cache(cache, runtime.hf_local_model_cache_size)
        if evicted_groups:
            logger.info(
                "Evicted TrajWeave HF rollout models before loading group=%s: %s",
                policy_group,
                evicted_groups,
            )
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.model_config.trust_remote_code,
            # HF local rollout 固定在 CPU；默认 fp32，资源受限时可由 runtime 显式选择 fp16/bf16。
            "dtype": _torch_dtype(runtime.hf_local_dtype),
        }
        if str(model_path) == str(self.model_config.local_path):
            kwargs["config"] = self.model_config.hf_config
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs,
        )
        model.eval()
        cache[policy_group] = model
        return cache[policy_group]

    def reload_local_models(self, model_paths: dict[str, str], *, policy_version: int) -> dict[str, Any]:
        """原子切换本地 rollout 使用的 HF 权重快照。"""

        normalized_paths = {str(group_id): str(path) for group_id, path in model_paths.items() if path}
        if not normalized_paths:
            raise ValueError("HF local rollout weight reload requires at least one model path.")

        old_models = getattr(self, "_trajweave_local_models", {})
        old_tokenizers = getattr(self, "_trajweave_local_tokenizers", {})
        self._trajweave_local_models = {}
        self._trajweave_local_tokenizers = {}
        self._trajweave_model_path_overrides = normalized_paths
        self._trajweave_policy_version = int(policy_version)
        del old_models
        del old_tokenizers
        gc.collect()

        # 在返回 trainer 前完成加载，避免快照轮转后才首次读取文件。
        loaded_groups = sorted(group_id for group_id in normalized_paths if group_id != "__default__")
        if not loaded_groups:
            loaded_groups = ["shared"]
        for group_id in loaded_groups:
            self._local_model(policy_group=group_id)

        logger.info(
            "Reloaded TrajWeave HF rollout snapshots at policy_version=%s for groups=%s",
            policy_version,
            loaded_groups,
        )
        return {
            "policy_version": self._trajweave_policy_version,
            "loaded_groups": loaded_groups,
            "model_paths": normalized_paths,
        }

    def release_local_models(self) -> dict[str, Any]:
        """在训练权重导出前释放 rollout 模型，避免两套完整权重形成内存峰值。"""

        old_models = getattr(self, "_trajweave_local_models", {})
        released_groups = tuple(str(group_id) for group_id in old_models)
        self._trajweave_local_models = {}
        del old_models
        gc.collect()
        logger.info("Released TrajWeave HF rollout models: %s", released_groups)
        return {
            "released_groups": list(released_groups),
            "policy_version": self._local_policy_version(),
        }

    def _local_policy_version(self) -> int:
        return int(getattr(self, "_trajweave_policy_version", 0))

    def _local_tokenizer(self, *, policy_group: str = "shared"):
        tokenizer_path = self._worker_group_tokenizer_path(policy_group)
        model_path = self._worker_group_model_path(policy_group)
        if not tokenizer_path and not model_path:
            return self.tokenizer
        tokenizer_source = tokenizer_path or model_path
        if str(tokenizer_source) == str(self.model_config.local_path):
            return self.tokenizer
        if not hasattr(self, "_trajweave_local_tokenizers"):
            self._trajweave_local_tokenizers = {}
        cache: dict[str, Any] = self._trajweave_local_tokenizers
        cache_key = str(tokenizer_source)
        if cache_key in cache:
            return cache[cache_key]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        cache[cache_key] = tokenizer
        return tokenizer

    def _local_prompt_ids(self, prompt_ids: list[int], *, policy_group: str, tokenizer: Any) -> list[int]:
        if tokenizer is self.tokenizer:
            return prompt_ids
        try:
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        except TypeError:
            prompt_text = self.tokenizer.decode(prompt_ids)
        try:
            ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(prompt_text)
        ids = flatten_token_ids(ids)
        if not ids:
            ids = [tokenizer.eos_token_id or tokenizer.pad_token_id or 0]
        return ids

    def _worker_group_model_path(self, group_id: str) -> str | None:
        overrides = getattr(self, "_trajweave_model_path_overrides", {})
        if group_id in overrides:
            return str(overrides[group_id])
        if "__default__" in overrides:
            return str(overrides["__default__"])
        group_cfg = self._worker_group_config(group_id)
        model_path = config_get(group_cfg, "model_path", default=None)
        if model_path is None:
            return None
        return str(to_python(model_path))

    def _worker_group_tokenizer_path(self, group_id: str) -> str | None:
        group_cfg = self._worker_group_config(group_id)
        tokenizer_path = config_get(group_cfg, "tokenizer_path", default=None)
        if tokenizer_path is None:
            return None
        return str(to_python(tokenizer_path))

    def _worker_group_config(self, group_id: str) -> Any:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        worker_groups = config_get(agent_cfg, "worker_groups", default={}) or {}
        worker_groups = to_python(worker_groups)
        if isinstance(worker_groups, dict):
            return worker_groups.get(group_id, {}) or {}
        if isinstance(worker_groups, list):
            for group in worker_groups:
                if isinstance(group, dict) and str(group.get("id")) == str(group_id):
                    return group
        return {}

    def _maporl_worker_group_model_path(self, group_id: str) -> str | None:
        """MAPoRL 旧调用点的兼容别名。"""

        return self._worker_group_model_path(group_id)

    def _maporl_worker_group_tokenizer_path(self, group_id: str) -> str | None:
        """MAPoRL 旧调用点的兼容别名。"""

        return self._worker_group_tokenizer_path(group_id)

    def _maporl_worker_group_config(self, group_id: str) -> Any:
        """MAPoRL 旧调用点的兼容别名。"""

        return self._worker_group_config(group_id)


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _evict_local_model_cache(cache: dict[str, Any], max_cached_models: int) -> tuple[str, ...]:
    if max_cached_models <= 0:
        return ()
    evicted: list[str] = []
    while len(cache) >= max_cached_models:
        group_id = next(iter(cache))
        model = cache.pop(group_id)
        del model
        evicted.append(group_id)
    if evicted:
        gc.collect()
    return tuple(evicted)
