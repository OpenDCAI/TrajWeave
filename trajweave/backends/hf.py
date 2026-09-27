from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer


@dataclass
class HFTransformersPolicyBackend:
    model_path: str
    device: str | None = None
    device_map: str | None = None
    torch_dtype: str | None = None
    max_new_tokens: int = 128
    generation_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "HFTransformersPolicyBackend requires torch and transformers. "
                "Install the project LLM dependencies before using backend.type=hf-transformers."
            ) from exc

        dtype = None
        if self.torch_dtype:
            dtype = getattr(torch, self.torch_dtype)
        self._uses_stable_tokenizer = self.model_path == "__tiny_random_gpt2__"
        if self._uses_stable_tokenizer:
            self.tokenizer = StableByteTokenizer()
            config = GPT2Config(
                vocab_size=258,
                n_positions=256,
                n_embd=64,
                n_layer=2,
                n_head=2,
                bos_token_id=1,
                eos_token_id=1,
                pad_token_id=0,
            )
            self.model = GPT2LMHeadModel(config)
            if dtype is not None:
                self.model.to(dtype=dtype)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": True}
            if self.device_map is not None:
                load_kwargs["device_map"] = self.device_map
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        if self.device is not None and self.device_map is None:
            self.model.to(self.device)
        if (
            not self._uses_stable_tokenizer
            and self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        import torch

        if self._uses_stable_tokenizer:
            token_ids = self.tokenizer.encode(request.prompt)[-192:]
            inputs = {
                "input_ids": torch.tensor([token_ids], dtype=torch.long, device=self.model.device),
                "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long, device=self.model.device),
            }
        else:
            inputs = self.tokenizer(request.prompt, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                **self.generation_config,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        response_ids = outputs.sequences[0, prompt_len:].detach().cpu().tolist()
        if self._uses_stable_tokenizer:
            text = self.tokenizer.decode(response_ids)
        else:
            text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        logprobs: list[float] = []
        if outputs.scores:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            )
            logprobs = transition_scores[0, -len(response_ids) :].detach().cpu().tolist()
        return PolicyResponse(
            text=text,
            token_ids=[int(token_id) for token_id in response_ids],
            logprobs=[float(value) for value in logprobs],
            metadata={"policy_backend": "hf-transformers", "model_path": self.model_path},
        )
