from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from trajweave.backends.verl.extensions.common.hooks import AgentWiseGRPOHooks, extension_hooks_for_config
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker


class TrajWeaveActorRolloutRefWorker(ActorRolloutRefWorker):
    """Actor worker that installs TrajWeave import shims before VERL engine init."""

    def __init__(self, *args, **kwargs):
        from trajweave.backends.verl import _ensure_torch_dtensor_import_compat
        from trajweave.backends.verl.extensions.common.nested_compat import install_worker_nested_tensor_compat

        _ensure_torch_dtensor_import_compat()
        install_worker_nested_tensor_compat()
        super().__init__(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_hf_rollout_snapshot(
        self,
        local_path: str,
        global_step: int,
        max_ckpt_to_keep: int = 2,
    ) -> None:
        """导出仅供本地 rollout 热加载的 HF 权重，不写 optimizer 分片。"""

        if self.actor is None:
            raise RuntimeError("Actor model must be initialized before exporting a rollout snapshot.")
        manager = self.actor.engine.checkpoint_manager
        previous_contents = manager.checkpoint_save_contents
        previous_paths = manager.previous_saved_paths
        previous_step = manager.previous_global_step
        try:
            manager.checkpoint_save_contents = ["hf_model"]
            manager.previous_saved_paths = []
            with _merged_lora_hf_state_dict(manager.model):
                self.actor.engine.save_checkpoint(
                    local_path=local_path,
                    global_step=global_step,
                    max_ckpt_to_keep=None,
                )
        finally:
            manager.checkpoint_save_contents = previous_contents
            manager.previous_saved_paths = previous_paths
            manager.previous_global_step = previous_step


@contextmanager
def _merged_lora_hf_state_dict(model: Any):
    """Make the FSDP HF exporter see merged base-model keys for PEFT actors."""

    unwrapped = getattr(model, "_fsdp_wrapped_module", model)
    if not getattr(unwrapped, "peft_config", None):
        yield
        return

    from verl.utils.checkpoint import fsdp_checkpoint_manager

    original_get_state_dict = fsdp_checkpoint_manager.get_fsdp_full_state_dict

    def get_merged_state_dict(candidate: Any, *args: Any, **kwargs: Any):
        if candidate is model:
            return _collect_merged_lora_snapshot_state_dict(candidate)
        return original_get_state_dict(candidate, *args, **kwargs)

    # Actor workers are isolated Ray processes. Keep the override scoped to this
    # snapshot so regular resumable checkpoints retain their PEFT state dicts.
    fsdp_checkpoint_manager.get_fsdp_full_state_dict = get_merged_state_dict
    try:
        yield
    finally:
        fsdp_checkpoint_manager.get_fsdp_full_state_dict = original_get_state_dict


def _collect_merged_lora_snapshot_state_dict(model: Any) -> dict[str, torch.Tensor]:
    from verl.utils.fsdp_utils import collect_merged_lora_params

    if not _has_cpu_parameters(model):
        return collect_merged_lora_params(model)

    # collect_merged_lora_params restores the live base weights before return.
    # Its CPU tensors can alias those weights, which also erases the collected
    # LoRA delta. Copy the full merged state while the merge context is active.
    from verl.utils.fsdp_utils import get_fsdp_full_state_dict, merged_lora_context, normalize_peft_param_name

    with merged_lora_context(model, backup_adapters=True):
        merged = normalize_peft_param_name(get_fsdp_full_state_dict(model, offload_to_cpu=True, rank0_only=True))
        return {name: _copied_cpu_tensor(value) for name, value in merged.items()}


def _has_cpu_parameters(model: Any) -> bool:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return False
    return any(
        (device := getattr(parameter, "device", None)) is not None and device.type == "cpu"
        for parameter in parameters()
    )


def _copied_cpu_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "full_tensor"):
        value = value.full_tensor()
    return value.detach().to(device="cpu", copy=True)


def apply_drmas_agent_wise_grpo_patch(config: Any = None) -> None:
    """Install Dr.MAS agent-wise GRPO behavior around VERL at runtime.

    The patch is intentionally kept outside ``verl/``. It only changes behavior
    when ``algorithm.group_by_agent_id`` is enabled by TrajWeave launch config.
    """

    from trajweave.backends.verl.extensions.common import nested_compat

    _patch_core_grpo()
    _patch_ray_compute_advantage()
    _patch_v1_multi_trajectory_advantage()
    _patch_v1_trainer_transfer_queue_fields()
    nested_compat.apply_tq_nested_compat_patch(config)


def _patch_core_grpo() -> None:
    from verl.trainer.ppo import core_algos

    if getattr(core_algos.compute_grpo_outcome_advantage, "_trajweave_drmas_patch", False):
        return

    hooks = AgentWiseGRPOHooks()

    def compute_grpo_outcome_advantage(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        index: np.ndarray,
        traj_index: np.ndarray | None = None,
        epsilon: float = 1e-6,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
        group_by_agent_id: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hooks.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            group_by_agent_id=group_by_agent_id,
        )

    compute_grpo_outcome_advantage._trajweave_drmas_patch = True  # type: ignore[attr-defined]
    core_algos.compute_grpo_outcome_advantage = compute_grpo_outcome_advantage
    core_algos.ADV_ESTIMATOR_REGISTRY[core_algos.AdvantageEstimator.GRPO.value] = compute_grpo_outcome_advantage


def _patch_ray_compute_advantage() -> None:
    from verl.trainer.ppo import core_algos, ray_trainer

    if getattr(ray_trainer.compute_advantage, "_trajweave_drmas_patch", False):
        return

    original_compute_advantage = ray_trainer.compute_advantage
    hooks = AgentWiseGRPOHooks()

    def compute_advantage(
        data,
        adv_estimator,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
    ):
        if adv_estimator != core_algos.AdvantageEstimator.GRPO:
            return original_compute_advantage(
                data,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

        return hooks.compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
            fallback=original_compute_advantage,
        )

    compute_advantage._trajweave_drmas_patch = True  # type: ignore[attr-defined]
    ray_trainer.compute_advantage = compute_advantage


def _patch_v1_multi_trajectory_advantage() -> None:
    from verl.trainer.ppo import core_algos, ray_trainer
    from verl.trainer.ppo.v1 import utils as v1_utils

    if getattr(v1_utils.compute_advantage_for_multi_trajectories, "_trajweave_drmas_patch", False):
        return

    original_compute_multi = v1_utils.compute_advantage_for_multi_trajectories
    v1_utils.compute_advantage = ray_trainer.compute_advantage

    def compute_advantage_for_multi_trajectories(
        data,
        batch_keys: list[str],
        adv_estimator,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
    ):
        if adv_estimator == core_algos.AdvantageEstimator.GRPO and _config_get(config, "group_by_agent_id", False):
            return ray_trainer.compute_advantage(
                data,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

        return original_compute_multi(
            data=data,
            batch_keys=batch_keys,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    compute_advantage_for_multi_trajectories._trajweave_drmas_patch = True  # type: ignore[attr-defined]
    v1_utils.compute_advantage_for_multi_trajectories = compute_advantage_for_multi_trajectories

    try:
        from verl.trainer.ppo.v1 import trainer_base

        trainer_base.compute_advantage_for_multi_trajectories = compute_advantage_for_multi_trajectories
    except Exception:
        return


def _patch_v1_trainer_transfer_queue_fields() -> None:
    from verl.trainer.ppo.v1 import trainer_base as tb

    if getattr(tb.PPOTrainer, "_trajweave_drmas_patch", False):
        return

    original_init_resource_pool_mgr = tb.PPOTrainer._init_resource_pool_mgr

    class _NullLLMServerManager:
        def get_client(self):
            return None

        def get_replicas(self):
            return []

    class _NullCheckpointEngineManager:
        def sleep_replicas(self):
            return None

        def update_weights(self, global_steps: int | None = None):
            return None

    def patched_setup(self):
        self._init_tokenizer()
        self._init_dataloader()
        self._init_dump_executor()
        self._init_resource_pool_mgr()
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if tb.Role.ActorRolloutRef in self.role_worker_mapping:
            actor_role = tb.Role.ActorRolloutRef
        elif tb.Role.ActorRollout in self.role_worker_mapping:
            actor_role = tb.Role.ActorRollout
        else:
            actor_role = tb.Role.Actor
        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        actor_rollout_cls = tb.RayClassWithInitArgs(
            cls=self.role_worker_mapping[actor_role],
            config=self.config.actor_rollout_ref,
            distillation_config=self.config.get("distillation"),
            role=str(actor_role),
        )
        self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        if self.use_critic:
            critic_cfg = tb.omega_conf_to_dataclass(self.config.critic)
            critic_cfg.engine.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            critic_cfg.engine.max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            worker_cfg = tb.TrainingWorkerConfig(
                model_type="value_model",
                model_config=getattr(critic_cfg, "model_config", None) or critic_cfg.model,
                engine_config=critic_cfg.engine,
                optimizer_config=critic_cfg.optim,
                checkpoint_config=critic_cfg.checkpoint,
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(tb.Role.Critic)
            critic_cls = tb.RayClassWithInitArgs(cls=self.role_worker_mapping[tb.Role.Critic], config=worker_cfg)
            self.resource_pool_to_cls[resource_pool][str(tb.Role.Critic)] = critic_cls

        all_wg = {}
        wg_kwargs = {}
        if tb.OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = tb.OmegaConf.select(self.config.global_profiler, "steps")
            if tb.OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    tb.OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                )
                wg_kwargs["worker_nsight_options"] = tb.OmegaConf.to_container(
                    tb.OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.config.trainer.device

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = tb.create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = tb.RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            tb.logger.info(f"create worker group {spawn_wg.keys()}")

        if self.use_critic:
            self.critic_wg = all_wg[str(tb.Role.Critic)]
            self.critic_wg.reset()
            value_loss_ = tb.partial(tb.value_loss, config=critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        lora_rank = self.config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or self.config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if self.use_reference_policy:
            self.ref_policy_wg = all_wg.get(str(tb.Role.ActorRolloutRef))
            if not self.ref_in_actor and self.ref_policy_wg is None:
                raise ValueError(
                    "TrajWeave self-managed TQ reference policy requires actor LoRA or an ActorRolloutRef worker."
                )

        resource_pool = (
            self.resource_pool_manager.get_resource_pool(tb.Role.RewardModel)
            if self.config.reward.reward_model.enable
            else None
        )
        self.reward_loop_manager = tb.RewardLoopManager(config=self.config, rm_resource_pool=resource_pool)

        if self.use_teacher_policy:
            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(tb.Role.TeacherModel)
            self.teacher_model_manager = tb.MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config = tb.omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        if _is_trajweave_self_managed_tq(self.config):
            self.llm_server_manager = _NullLLMServerManager()
            self.checkpoint_manager = _NullCheckpointEngineManager()
        else:
            self.llm_server_manager = tb.LLMServerManager.create(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rollout_resource_pool=actor_rollout_resource_pool,
            )
            checkpoint_engine_config = tb.omega_conf_to_dataclass(
                self.config.actor_rollout_ref.rollout.checkpoint_engine
            )
            checkpoint_engine_config.backend = "naive"
            self.checkpoint_manager = tb.CheckpointEngineManager(
                config=checkpoint_engine_config,
                actor_wg=self.actor_rollout_wg,
                replicas=self.llm_server_manager.get_replicas(),
            )

        self.checkpoint_manager.sleep_replicas()
        self._load_checkpoint()
        tb.logger.info("all initialize finished, ready to fit")

    def patched_init_resource_pool_mgr(self):
        if not _is_trajweave_self_managed_tq(self.config):
            return original_init_resource_pool_mgr(self)

        config = self.config
        self.role_worker_mapping = {}
        self.mapping = {}
        self.role_worker_mapping[tb.Role.Actor] = tb.ray.remote(TrajWeaveActorRolloutRefWorker)
        self.mapping[tb.Role.Actor] = "global_pool"

        if tb.need_critic(config):
            worker_factory = getattr(self, "_critic_training_worker_cls", None)
            critic_worker_cls = worker_factory() if callable(worker_factory) else tb.TrainingWorker
            self.role_worker_mapping[tb.Role.Critic] = tb.ray.remote(critic_worker_cls)
            self.mapping[tb.Role.Critic] = "global_pool"

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}

        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")
            resource_pool_spec["reward_pool"] = [
                config.reward.reward_model.n_gpus_per_node
            ] * config.reward.reward_model.nnodes
            self.mapping[tb.Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[tb.Role.RewardModel] = "global_pool"

        distillation_config = config.get("distillation")
        if tb.is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")
            resource_pool_spec["teacher_pool"] = [distillation_config.n_gpus_per_node] * distillation_config.nnodes
            self.mapping[tb.Role.TeacherModel] = "teacher_pool"

        self.resource_pool_manager = tb.ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def patched_compute_advantage(self, batch, metrics: dict):
        hooks = extension_hooks_for_config(self.config)
        fields = list(hooks.tq_select_fields("advantage", config=self.config.algorithm))
        data = tb.tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        response_mask = data["response_mask"]
        data = tb.DataProto(batch=_tensordict_to_padded_tensor_compat(data))
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = _pop_tq_field_as_object_array(data, "uid")
        for field in hooks.batch_schema_fields("advantage"):
            if field in data.batch:
                data.non_tensor_batch[field] = _pop_tq_field_as_object_array(data, field)

        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = tb.apply_kl_penalty(
                data,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = tb.compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        data = hooks.compute_advantage(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
            fallback=tb.compute_advantage_for_multi_trajectories,
        )

        output_fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            output_fields.append("token_level_rewards")
        if rollout_correction:
            output_fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                output_fields.append("rollout_is_weights")
        output_fields = list(hooks.output_fields("advantage", tuple(output_fields), data, config=self.config.algorithm))

        output = {}
        for field in output_fields:
            output[field] = tb.response_to_nested(data.batch[field], response_mask)
        output = tb.TensorDict(output, batch_size=len(batch))
        return tb.tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

    tb.PPOTrainer._setup = patched_setup
    tb.PPOTrainer._init_resource_pool_mgr = patched_init_resource_pool_mgr
    tb.PPOTrainer._compute_advantage = patched_compute_advantage
    tb.PPOTrainer._trajweave_drmas_patch = True

    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

    if not getattr(PPOTrainerSync.on_step_end, "_trajweave_hf_local_sync", False):
        original_sync_on_step_end = PPOTrainerSync.on_step_end

        def patched_sync_on_step_end(self):
            if _config_get(_config_get(self.config, "trajweave", {}), "agent_loop_backend") == "hf_local_tq":
                from trajweave.backends.verl.weight_sync import sync_hf_local_rollout_weights
                from verl.utils.debug import marked_timer

                with marked_timer("update_weights", self.timing_raw, color="red"):
                    sync_hf_local_rollout_weights(self)
                return None
            return original_sync_on_step_end(self)

        patched_sync_on_step_end._trajweave_hf_local_sync = True  # type: ignore[attr-defined]
        PPOTrainerSync.on_step_end = patched_sync_on_step_end


def _is_trajweave_self_managed_tq(config: Any) -> bool:
    return _config_get(_config_get(config, "trajweave", {}), "agent_loop_backend") in {"synthetic_tq", "hf_local_tq"}


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        return config.get(key, default)
    except (AttributeError, TypeError):
        return getattr(config, key, default)


def _pop_tq_field_as_object_array(data, key: str) -> np.ndarray:
    value = data.batch.pop(key)
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "data"):
        value = value.data
    return np.array(value, dtype=object)


def _nested_to_padded_tensor_compat(nested_tensor: torch.Tensor, padding: int | float = 0) -> torch.Tensor:
    try:
        return nested_tensor.to_padded_tensor(padding)
    except NotImplementedError:
        values = nested_tensor.values()
        offsets = nested_tensor.offsets()
        if values.dim() != 1:
            raise
        batch_size = len(offsets) - 1
        max_len = int(offsets.diff().max().item()) if batch_size else 0
        padded = torch.full((batch_size, max_len), padding, dtype=values.dtype, device=values.device)
        lengths = offsets.diff().tolist()
        starts = offsets[:-1].tolist()
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
