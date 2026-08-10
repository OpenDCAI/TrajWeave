from __future__ import annotations

from typing import Any

import torch
from verl.utils.chat_template import apply_chat_template

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import flatten_token_ids, to_python


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

    def _generate_local_response_ids(self, prompt_ids: list[int], *, policy_group: str = "shared") -> list[int]:
        model = self._local_model(policy_group=policy_group)
        tokenizer = self._local_tokenizer(policy_group=policy_group)
        device = next(model.parameters()).device
        input_ids = self._local_prompt_ids(prompt_ids, policy_group=policy_group, tokenizer=tokenizer)
        input_ids = torch.tensor([input_ids[-self.rollout_config.prompt_length :]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        sampling = self._local_sampling_kwargs()
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.rollout_config.response_length,
                **sampling,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
                eos_token_id=tokenizer.eos_token_id,
            )
        group_response_ids = sequences[0, input_ids.shape[-1] :].detach().cpu().tolist()
        if not group_response_ids:
            return [self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0]
        if tokenizer is self.tokenizer:
            response_ids = group_response_ids
        else:
            response_text = tokenizer.decode(group_response_ids, skip_special_tokens=True)
            response_ids = self._encode_text(response_text)
        return [int(token_id) for token_id in response_ids[: self.rollout_config.response_length]]

    def _generate_local_response_with_logprobs(
        self,
        prompt_ids: list[int],
        *,
        policy_group: str = "shared",
    ) -> tuple[list[int], list[float]]:
        model = self._local_model(policy_group=policy_group)
        tokenizer = self._local_tokenizer(policy_group=policy_group)
        if tokenizer is not self.tokenizer:
            raise ValueError("Rollout logprob capture requires the rollout and policy tokenizers to match.")
        device = next(model.parameters()).device
        input_ids = self._local_prompt_ids(prompt_ids, policy_group=policy_group, tokenizer=tokenizer)
        input_ids = torch.tensor([input_ids[-self.rollout_config.prompt_length :]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        sampling = self._local_sampling_kwargs()
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.rollout_config.response_length,
                **sampling,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        response_ids = output.sequences[0, input_ids.shape[-1] :].detach().cpu().tolist()
        if not response_ids:
            response_ids = [tokenizer.eos_token_id or tokenizer.pad_token_id or 0]
        logprobs = []
        for token_id, scores in zip(response_ids, output.scores, strict=False):
            token_logprobs = torch.log_softmax(scores[0].float(), dim=-1)
            logprobs.append(float(token_logprobs[int(token_id)].detach().cpu().item()))
        if len(logprobs) < len(response_ids):
            logprobs.extend([0.0] * (len(response_ids) - len(logprobs)))
        return [int(token_id) for token_id in response_ids], logprobs

    def _local_sampling_kwargs(self) -> dict[str, Any]:
        rollout = getattr(self, "rollout_config", None)
        do_sample = bool(getattr(rollout, "do_sample", False))
        if not do_sample:
            return {"do_sample": False}
        sampling = {
            "do_sample": True,
            "temperature": float(getattr(rollout, "temperature", 1.0)),
            "top_p": float(getattr(rollout, "top_p", 1.0)),
        }
        top_k = int(getattr(rollout, "top_k", -1))
        if top_k > 0:
            sampling["top_k"] = top_k
        return sampling

    def _local_model(self, *, policy_group: str = "shared"):
        if not hasattr(self, "_trajweave_local_models"):
            self._trajweave_local_models = {}
        cache: dict[str, Any] = self._trajweave_local_models
        if policy_group in cache:
            return cache[policy_group]
        from transformers import AutoModelForCausalLM

        model_path = self._maporl_worker_group_model_path(policy_group) or self.model_config.local_path
        kwargs: dict[str, Any] = {"trust_remote_code": self.model_config.trust_remote_code}
        if str(model_path) == str(self.model_config.local_path):
            kwargs["config"] = self.model_config.hf_config
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **kwargs,
        )
        model.eval()
        cache[policy_group] = model
        return cache[policy_group]

    def _local_tokenizer(self, *, policy_group: str = "shared"):
        tokenizer_path = self._maporl_worker_group_tokenizer_path(policy_group)
        model_path = self._maporl_worker_group_model_path(policy_group)
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

    def _maporl_worker_group_model_path(self, group_id: str) -> str | None:
        group_cfg = self._maporl_worker_group_config(group_id)
        model_path = config_get(group_cfg, "model_path", default=None)
        if model_path is None:
            return None
        return str(to_python(model_path))

    def _maporl_worker_group_tokenizer_path(self, group_id: str) -> str | None:
        group_cfg = self._maporl_worker_group_config(group_id)
        tokenizer_path = config_get(group_cfg, "tokenizer_path", default=None)
        if tokenizer_path is None:
            return None
        return str(to_python(tokenizer_path))

    def _maporl_worker_group_config(self, group_id: str) -> Any:
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
