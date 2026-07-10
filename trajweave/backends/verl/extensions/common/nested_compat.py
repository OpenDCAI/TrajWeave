from __future__ import annotations

from typing import Any

import torch


def apply_tq_nested_compat_patch(config: Any = None) -> None:
    """Install small TensorDict/NestedTensor shims needed by TransferQueue smoke runs."""

    install_worker_nested_tensor_compat()
    _patch_trainer_base_padding_helpers()


def install_worker_nested_tensor_compat() -> None:
    _patch_nested_sum()
    _patch_nested_to_padded_tensor()
    _patch_worker_losses()
    _patch_fsdp_response_outputs()


def _patch_nested_sum() -> None:
    if getattr(torch.Tensor, "_trajweave_nested_sum_patch", False):
        return
    original_sum = torch.Tensor.sum

    def patched_sum(self, *args, **kwargs):
        if getattr(self, "is_nested", False) and not args and not kwargs:
            return self.values().sum()
        return original_sum(self, *args, **kwargs)

    patched_sum._trajweave_original = original_sum  # type: ignore[attr-defined]
    torch.Tensor.sum = patched_sum
    torch.Tensor._trajweave_nested_sum_patch = True


def _patch_nested_to_padded_tensor() -> None:
    if getattr(torch.nested.to_padded_tensor, "_trajweave_patch", False):
        return

    original_to_padded_tensor = torch.nested.to_padded_tensor

    def patched_to_padded_tensor(input, padding, output_size=None, *args, **kwargs):
        try:
            return original_to_padded_tensor(input, padding, output_size, *args, **kwargs)
        except NotImplementedError:
            if not getattr(input, "is_nested", False):
                raise
            if output_size is None:
                offsets = input.offsets()
                batch_size = len(offsets) - 1
                max_len = int(offsets.diff().max().item()) if batch_size else 0
                output_size = (batch_size, max_len)
            return _nested_to_padded_tensor_with_size(input, padding=padding, output_size=tuple(output_size))

    patched_to_padded_tensor._trajweave_patch = True  # type: ignore[attr-defined]
    torch.nested.to_padded_tensor = patched_to_padded_tensor


def _patch_worker_losses() -> None:
    from verl.workers.utils import losses

    if getattr(losses.ppo_loss, "_trajweave_nested_compat_patch", False):
        return

    def patched_ppo_loss(config, model_output, data, dp_group=None):
        from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
        from verl.utils import tensordict_utils as tu
        from verl.utils.dataset.dataset_utils import DatasetPadMode
        from verl.utils.metric import AggregationType, Metric
        from verl.workers.utils.padding import no_padding_2_padding, response_from_nested

        pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        response_mask_source = data["response_mask"]
        use_nested_response_loss = pad_mode == DatasetPadMode.NO_PADDING and getattr(
            model_output["log_probs"], "is_nested", False
        )
        use_flat_response_loss = (
            pad_mode == DatasetPadMode.NO_PADDING
            and not getattr(model_output["log_probs"], "is_nested", False)
            and model_output["log_probs"].dim() == 2
            and model_output["log_probs"].shape[0] == 1
            and getattr(response_mask_source, "is_nested", False)
        )

        if use_nested_response_loss:
            log_prob = response_from_nested(model_output["log_probs"], response_mask_source).values().unsqueeze(0)
            entropy = model_output.get("entropy", None)
            if entropy is not None:
                entropy = response_from_nested(entropy, response_mask_source).values().unsqueeze(0)
        elif use_flat_response_loss:
            log_prob = model_output["log_probs"]
            entropy = model_output.get("entropy", None)
        else:
            log_prob = no_padding_2_padding(model_output["log_probs"], data)
            entropy = model_output.get("entropy", None)
            if entropy is not None:
                entropy = no_padding_2_padding(entropy, data)

        config.global_batch_info["dp_size"] = data["dp_size"]
        config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
        config.global_batch_info["global_batch_size"] = data["global_batch_size"]
        config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

        if (
            data["dp_size"] > 1
            or data["batch_num_tokens"] is not None
            or data["global_batch_size"] is not None
            or config.loss_scale_factor is not None
        ):
            metric_aggregation = AggregationType.SUM
        else:
            metric_aggregation = AggregationType.MEAN

        fields = ["response_mask", "old_log_probs", "advantages"]
        if "rollout_is_weights" in data:
            fields.append("rollout_is_weights")
        if "ref_log_prob" in data:
            fields.append("ref_log_prob")
        selected = data.select(*fields)

        if use_nested_response_loss or use_flat_response_loss:
            response_mask = selected["response_mask"].values().to(bool).unsqueeze(0)
            old_log_prob = selected["old_log_probs"].values().unsqueeze(0)
            advantages = selected["advantages"].values().unsqueeze(0)
            rollout_is_weights = (
                selected["rollout_is_weights"].values().unsqueeze(0) if "rollout_is_weights" in selected else None
            )
            ref_log_prob = selected["ref_log_prob"].values().unsqueeze(0) if "ref_log_prob" in selected else None
        else:
            selected = _tensordict_to_padded_tensor_compat(selected)
            response_mask = selected["response_mask"].to(bool)
            old_log_prob = selected["old_log_probs"]
            advantages = selected["advantages"]
            rollout_is_weights = selected.get("rollout_is_weights", None)
            ref_log_prob = selected.get("ref_log_prob", None)

        policy_loss_fn = get_policy_loss_fn(config.policy_loss.get("loss_mode", "vanilla"))
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )
        metrics = dict(Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN))
        metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
        policy_loss = pg_loss

        if entropy is not None:
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=response_mask,
                loss_agg_mode=config.loss_agg_mode,
                **config.global_batch_info,
            )
            policy_loss -= config.entropy_coeff * entropy_loss
            metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

        if config.use_kl_loss:
            assert ref_log_prob is not None, "ref_log_prob is required when use_kl_loss is enabled"
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=response_mask,
                loss_agg_mode=config.loss_agg_mode,
                **config.global_batch_info,
            )
            policy_loss += kl_loss * config.kl_loss_coef
            metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
            metrics["kl_coef"] = config.kl_loss_coef

        return policy_loss, metrics

    patched_ppo_loss._trajweave_nested_compat_patch = True  # type: ignore[attr-defined]
    losses.ppo_loss = patched_ppo_loss
    try:
        import verl.workers.engine_workers as engine_workers

        engine_workers.ppo_loss = patched_ppo_loss
    except Exception:
        return


def _patch_fsdp_response_outputs() -> None:
    from verl.workers.engine.fsdp import transformer_impl as ti

    cls = ti.FSDPEngineWithLMHead
    if getattr(cls.prepare_model_outputs, "_trajweave_response_patch", False):
        return

    original_prepare_model_outputs = cls.prepare_model_outputs

    def patched_prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
        from verl.utils import tensordict_utils as tu
        from verl.utils.dataset.dataset_utils import DatasetPadMode

        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        calculate_sum_pi_squared = tu.get_non_tensor_data(
            data=micro_batch,
            key="calculate_sum_pi_squared",
            default=False,
        )
        distillation_use_topk = tu.get_non_tensor_data(data=micro_batch, key="distillation_use_topk", default=False)

        force_flat_response = logits_processor_func is not None
        supported_flat_path = (
            force_flat_response
            and not use_remove_padding
            and not use_fused_kernels
            and pad_mode == DatasetPadMode.NO_PADDING
            and not calculate_sum_pi_squared
            and not distillation_use_topk
        )
        if not supported_flat_path:
            return original_prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func)

        input_ids = micro_batch["input_ids"]
        response_mask = micro_batch["response_mask"]
        logits = output.logits
        temperature = output_args["temperature"].unsqueeze(-1).unsqueeze(-1)
        logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))

        input_ids_padded = torch.nested.to_padded_tensor(
            input_ids,
            padding=0,
            output_size=(micro_batch.batch_size[0], logits.shape[1]),
        )
        seq_lengths = input_ids.offsets().diff()
        response_lens = response_mask.offsets().diff()
        log_prob_pieces = []
        entropy_pieces = []
        entropy = None
        if calculate_entropy:
            if not self.engine_config.entropy_checkpointing:
                entropy = ti.verl_F.entropy_from_logits(logits)
            else:
                entropy = torch.utils.checkpoint.checkpoint(ti.verl_F.entropy_from_logits, logits)

        for row, (seq_len, resp_len) in enumerate(zip(seq_lengths.tolist(), response_lens.tolist(), strict=False)):
            start = seq_len - resp_len - 1
            end = seq_len - 1
            log_prob_pieces.append(
                ti.logprobs_from_logits(
                    logits=logits[row, start:end],
                    labels=input_ids_padded[row, start + 1 : end + 1],
                )
            )
            if calculate_entropy and entropy is not None:
                entropy_pieces.append(entropy[row, start:end])

        model_output = {"log_probs": torch.cat(log_prob_pieces, dim=0).unsqueeze(0)}
        if calculate_entropy:
            model_output["entropy"] = torch.cat(entropy_pieces, dim=0).unsqueeze(0)
        return model_output

    patched_prepare_model_outputs._trajweave_response_patch = True  # type: ignore[attr-defined]
    cls.prepare_model_outputs = patched_prepare_model_outputs


def _patch_trainer_base_padding_helpers() -> None:
    from verl.trainer.ppo.v1 import trainer_base as tb

    if getattr(tb.PPOTrainer, "_trajweave_nested_compat_patch", False):
        return

    original_compute_old_log_prob = tb.PPOTrainer._compute_old_log_prob
    original_compute_metrics = tb.PPOTrainer._compute_metrics

    def patched_compute_old_log_prob(self, batch, metrics: dict):
        try:
            return original_compute_old_log_prob(self, batch, metrics)
        except NotImplementedError as exc:
            if "NestedTensor" not in str(exc) and "to_padded_tensor" not in str(exc):
                raise
            return _compute_old_log_prob_with_padding_compat(self, batch, metrics)

    def patched_compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        try:
            return original_compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch)
        except NotImplementedError as exc:
            if "NestedTensor" not in str(exc) and "to_padded_tensor" not in str(exc):
                raise
            raise RuntimeError(
                "TrajWeave nested TensorDict compatibility reached _compute_metrics. "
                "Please disable no-padding smoke or add a dedicated metrics shim."
            ) from exc

    tb.PPOTrainer._compute_old_log_prob = patched_compute_old_log_prob
    tb.PPOTrainer._compute_metrics = patched_compute_metrics
    tb.PPOTrainer._trajweave_nested_compat_patch = True


def _compute_old_log_prob_with_padding_compat(self, batch, metrics: dict):
    from verl.trainer.ppo import core_algos
    from verl.utils.debug.metrics import calculate_debug_metrics

    tb = _trainer_base_module()
    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
    if bypass_recomputing_logprobs:
        data = tb.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["rollout_log_probs"],
        )
        data["old_log_probs"] = data.pop("rollout_log_probs")
        tb.tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)
        return batch

    batch.extra_info.update(
        {
            "calculate_entropy": True,
            "compute_loss": False,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        }
    )
    output = self.actor_rollout_wg.compute_log_prob(batch)
    assert len(output) == len(batch)

    fields = ["entropy", "log_probs", "response_mask"]
    if self.config.actor_rollout_ref.rollout.calculate_log_probs:
        fields.extend(["responses", "rollout_log_probs"])
    data = tb.tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

    data["old_log_probs"] = tb.response_from_nested(data.pop("log_probs"), data["response_mask"])
    data["entropy"] = tb.response_from_nested(data.pop("entropy"), data["response_mask"])
    batch = tb.tq.kv_batch_put(
        keys=batch.keys,
        partition_id=batch.partition_id,
        fields=data.select("old_log_probs", "entropy"),
    )

    data_proto = tb.DataProto(batch=_tensordict_to_padded_tensor_compat(data))
    actor_config = self.config.actor_rollout_ref.actor
    entropy_agg = core_algos.agg_loss(
        loss_mat=data_proto.batch["entropy"],
        loss_mask=data_proto.batch["response_mask"],
        loss_agg_mode=actor_config.loss_agg_mode,
        loss_scale_factor=actor_config.loss_scale_factor,
    )
    metrics.update({"actor/entropy": entropy_agg.detach().item()})
    if self.config.actor_rollout_ref.rollout.calculate_log_probs:
        metrics.update(calculate_debug_metrics(data_proto))

    return batch


def _trainer_base_module():
    from verl.trainer.ppo.v1 import trainer_base

    return trainer_base


def _nested_to_padded_tensor_compat(nested_tensor: torch.Tensor, padding: int | float = 0) -> torch.Tensor:
    try:
        return nested_tensor.to_padded_tensor(padding)
    except NotImplementedError:
        offsets = nested_tensor.offsets()
        batch_size = len(offsets) - 1
        max_len = int(offsets.diff().max().item()) if batch_size else 0
        return _nested_to_padded_tensor_with_size(nested_tensor, padding=padding, output_size=(batch_size, max_len))


def _nested_to_padded_tensor_with_size(
    nested_tensor: torch.Tensor,
    *,
    padding: int | float,
    output_size: tuple[int, ...],
) -> torch.Tensor:
    values = nested_tensor.values()
    offsets = nested_tensor.offsets()
    if len(output_size) != 2 or values.dim() != 1:
        raise NotImplementedError("TrajWeave nested padding shim only supports 1-D jagged values.")
    padded = torch.full(output_size, padding, dtype=values.dtype, device=values.device)
    starts = offsets[:-1].tolist()
    lengths = offsets.diff().tolist()
    for row, (start, length) in enumerate(zip(starts, lengths, strict=False)):
        padded[row, :length] = values[start : start + length]
    return padded


def _tensordict_to_padded_tensor_compat(data):
    try:
        return data.to_padded_tensor()
    except NotImplementedError:
        from tensordict import TensorDict

        padded = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and getattr(value, "is_nested", False):
                padded[key] = _nested_to_padded_tensor_compat(value, padding=0)
            else:
                padded[key] = value
        return TensorDict(padded, batch_size=data.batch_size)
