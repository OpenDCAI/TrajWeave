from __future__ import annotations

import logging
import os
from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from trajweave.backends.verl.extensions import apply_verl_runtime_extensions
from trajweave.backends.verl.multi_actor import reconcile_multi_actor_global_assets
from trajweave.backends.verl.trainers import register_trajweave_trainers
from trajweave.backends.verl.trainers.comlrl_iterative_runtime import (
    apply_iterative_resume_model_paths,
    iterative_settings,
    run_comlrl_iterative_training,
)
from trajweave.backends.verl.trainers.comlrl_staged import (
    build_marlhf_preference_config,
    collect_marlhf_preferences,
    prepare_marlhf_online_config,
    train_marlhf_reward_model,
)
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _bind_agent_loop_manager(trainer, manager) -> None:
    trainer.agent_loop_manager = manager
    from trajweave.backends.verl.weight_sync import sync_hf_local_rollout_weights

    sync_hf_local_rollout_weights(trainer)


@ray.remote
class TrajWeaveTaskRunnerV1:
    """VERL V1 TaskRunner with TrajWeave runtime extensions loaded in Ray."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ

        self.agent_loop_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )
        _bind_agent_loop_manager(self.trainer, self.agent_loop_manager)

    def _shutdown_agent_loop_manager(self) -> None:
        if self.agent_loop_manager is None:
            return
        for worker in getattr(self.agent_loop_manager, "agent_loop_workers", ()):
            ray.kill(worker, no_restart=True)
        self.agent_loop_manager = None
        if self.trainer is not None:
            self.trainer.agent_loop_manager = None

    def run(self, config: DictConfig):
        import transfer_queue as tq

        from verl.trainer.ppo.v1 import get_trainer_cls

        iterative = iterative_settings(config)
        apply_iterative_resume_model_paths(config)
        marlhf_dispatch = prepare_marlhf_online_config(config)
        reconcile_multi_actor_global_assets(config)
        applied = apply_verl_runtime_extensions(config)
        if applied:
            logger.info("Applied TrajWeave VERL runtime extensions: %s", ", ".join(applied))

        register_trajweave_trainers()
        trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        tq.init(config.transfer_queue)
        try:
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            if iterative:
                self.init_agent_loop_manager()
                run_comlrl_iterative_training(
                    trainer=self.trainer,
                    agent_loop_manager=self.agent_loop_manager,
                    config=config,
                )
                return
            if marlhf_dispatch is not None:
                online_config = config
                self.config = build_marlhf_preference_config(online_config)
                self.init_agent_loop_manager()
                preference_pairs = collect_marlhf_preferences(
                    self.trainer,
                    self.agent_loop_manager,
                    self.config,
                )
                self._shutdown_agent_loop_manager()
                train_marlhf_reward_model(online_config, preference_pairs)
                self.config = online_config
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
        finally:
            self._shutdown_agent_loop_manager()
            tq.close()


@hydra.main(config_path="../../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    apply_iterative_resume_model_paths(config)
    prepare_marlhf_online_config(config)
    reconcile_multi_actor_global_assets(config)
    applied = apply_verl_runtime_extensions(config)
    if applied:
        logger.info("Applied TrajWeave VERL runtime extensions in driver: %s", ", ".join(applied))

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    if config.trainer.use_v1:
        run_ppo(config, task_runner_class=TrajWeaveTaskRunnerV1)
    else:
        from verl.trainer.main_ppo_v0 import TaskRunner

        logger.warning("Legacy trainer `main_ppo_v0.py` is deprecated; TrajWeave extensions are only tested on V1.")
        run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
