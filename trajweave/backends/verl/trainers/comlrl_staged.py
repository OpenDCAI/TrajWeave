from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict

from trajweave.backends.verl.extensions.comlrl.preference import (
    PREFERENCE_TQ_FIELDS,
    validate_preference_pairs,
)
from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import batch_item, to_python
from trajweave.backends.verl.workers.scalar_head import RewardModelWorker
from trajweave.core.preference import JointPreferencePair
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.core.trajectory import MultiAgentTrajectory
from trajweave.credit.comlrl.marlhf import JointRewardModelScorer
from trajweave.credit.comlrl.preference import build_joint_preference_pairs

COMLRL_REINFORCE_HOOK_FQN = "trajweave.backends.verl.extensions.comlrl.reinforce.CoMLRLReinforceHooks"
COMLRL_REINFORCE_EXTENSION = "trajweave_comlrl_reinforce"


class MARLHFStageState(str, Enum):
    INITIAL = "initial"
    PREFERENCES_COLLECTED = "preferences_collected"
    REWARD_MODEL_FROZEN = "reward_model_frozen"
    ONLINE_RL_COMPLETED = "online_rl_completed"


@dataclass(frozen=True)
class RLDispatchSpec:
    algorithm: str
    trainer: str
    advantage_mode: str | None
    critic_topology: str | None = None


_RL_DISPATCH = {
    "magrpo": RLDispatchSpec("magrpo", "trajweave_multi_actor_sync", "mean"),
    "mareinforce": RLDispatchSpec("mareinforce", "trajweave_multi_actor_sync", "raw"),
    "marloo": RLDispatchSpec("marloo", "trajweave_multi_actor_sync", "rloo"),
    "maremax": RLDispatchSpec("maremax", "trajweave_multi_actor_sync", "max"),
    "iac": RLDispatchSpec("iac", "trajweave_multi_actor_critic_sync", None, "independent"),
    "maac": RLDispatchSpec("maac", "trajweave_multi_actor_critic_sync", None, "centralized"),
}


def resolve_marlhf_rl_dispatch(algorithm: str) -> RLDispatchSpec:
    normalized = str(algorithm).strip().lower()
    try:
        return _RL_DISPATCH[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unsupported MARLHF RL algorithm {algorithm!r}; expected one of {sorted(_RL_DISPATCH)}."
        ) from error


def is_marlhf_config(config: Any) -> bool:
    comlrl = _select(config, "trajweave.comlrl", {}) or {}
    marlhf = config_get(comlrl, "marlhf", {}) or {}
    algorithm = str(config_get(comlrl, "algorithm", "")).strip().lower()
    return algorithm == "marlhf" or bool(config_get(marlhf, "enabled", False))


def prepare_marlhf_online_config(config: Any) -> RLDispatchSpec | None:
    if not is_marlhf_config(config):
        return None
    if _select(config, "trainer.use_v1", None) is False:
        raise ValueError("MARLHF requires trainer.use_v1=true.")
    comlrl = _select(config, "trajweave.comlrl", {}) or {}
    marlhf = config_get(comlrl, "marlhf", {}) or {}
    algorithm = str(config_get(marlhf, "rl_algorithm", "magrpo")).strip().lower()
    dispatch = resolve_marlhf_rl_dispatch(algorithm)
    max_turns = int(config_get(comlrl, "max_turns", config_get(comlrl, "num_turns", 1)))
    joint_mode = str(config_get(comlrl, "joint_mode", "aligned")).strip().lower()
    if max_turns != 1:
        raise ValueError("MARLHF requires trajweave.comlrl.max_turns=1.")
    if joint_mode not in {"align", "aligned"}:
        raise ValueError("MARLHF requires trajweave.comlrl.joint_mode=aligned.")
    rollout_n = int(_select(config, "actor_rollout_ref.rollout.n", 1))
    if dispatch.critic_topology is None and rollout_n < 2:
        raise ValueError("MAGRPO-family MARLHF requires actor_rollout_ref.rollout.n>=2.")
    if dispatch.critic_topology is not None and rollout_n != 1:
        raise ValueError("TrajWeave IAC/MAAC MARLHF requires actor_rollout_ref.rollout.n=1.")

    _update(config, "trajweave.comlrl.algorithm", dispatch.algorithm)
    _update(config, "trajweave.comlrl.max_turns", 1)
    _update(config, "trajweave.comlrl.joint_mode", "aligned")
    _update(config, "trajweave.comlrl.marlhf.enabled", True)
    _update(config, "trajweave.comlrl.marlhf.rl_algorithm", dispatch.algorithm)
    if dispatch.critic_topology is None:
        _update(config, "trajweave.credit_allocator", f"comlrl_{dispatch.algorithm}")
        _ensure_verl_extension(config, COMLRL_REINFORCE_EXTENSION)
        _update(config, "critic.enable", False)
        _update(config, "algorithm.adv_estimator", "reinforce_plus_plus")
        _update(config, "algorithm.extension_hooks_class", COMLRL_REINFORCE_HOOK_FQN)
        _update(config, "actor_rollout_ref.actor.policy_loss.loss_mode", "gpg")
        _update(config, "actor_rollout_ref.actor.loss_agg_mode", "seq-mean-token-sum")
        _update(config, "actor_rollout_ref.actor.ppo_epochs", 1)
        _update(config, "algorithm.use_kl_in_reward", False)
    else:
        _prepare_marlhf_actor_critic_config(config, marlhf=marlhf, algorithm=dispatch.algorithm)
    _update(config, "trainer.v1.trainer_mode", dispatch.trainer)
    return dispatch


def _prepare_marlhf_actor_critic_config(config: Any, *, marlhf: Any, algorithm: str) -> None:
    actor_groups = [str(group) for group in to_python(_select(config, "agent.model_ids", []))]
    actor_groups = list(dict.fromkeys(actor_groups))
    if not actor_groups:
        raise ValueError("MARLHF IAC/MAAC requires non-empty agent.model_ids.")
    critic_model = str(config_get(marlhf, "critic_model_name", "")).strip()
    critic_tokenizer = str(config_get(marlhf, "critic_tokenizer_name", critic_model)).strip()
    if not critic_model or not critic_tokenizer:
        raise ValueError("MARLHF IAC/MAAC requires marlhf.critic_model_name and a critic tokenizer.")
    critic_type = str(config_get(marlhf, "critic_type", "v")).strip().lower()
    if critic_type not in {"v", "q"}:
        raise ValueError("MARLHF critic_type must be one of: v, q.")
    critic_gpus = int(config_get(marlhf, "critic_gpus", 1))
    if critic_gpus != 1:
        raise ValueError("Current TrajWeave MARLHF IAC/MAAC requires critic_gpus=1 per critic route.")
    max_length = int(config_get(marlhf, "critic_max_length", 2048))
    if max_length < 1:
        raise ValueError("MARLHF critic_max_length must be positive.")
    if algorithm == "iac":
        routes = [
            {
                "critic_group": f"critic-{group}",
                "actor_groups": [group],
                "model_path": critic_model,
                "tokenizer_path": critic_tokenizer,
                "gpus": critic_gpus,
            }
            for group in actor_groups
        ]
        topology = "independent"
    else:
        routes = [
            {
                "critic_group": "centralized",
                "actor_groups": actor_groups,
                "model_path": critic_model,
                "tokenizer_path": critic_tokenizer,
                "gpus": critic_gpus,
            }
        ]
        topology = "centralized"
    _update(config, "critic.enable", True)
    _update(config, "algorithm.adv_estimator", "gae")
    _update(config, "algorithm.extension_hooks_class", None)
    _update(config, "actor_rollout_ref.actor.policy_loss.loss_mode", "gpg")
    _update(config, "actor_rollout_ref.actor.loss_agg_mode", "seq-mean-token-sum")
    _update(config, "actor_rollout_ref.actor.ppo_epochs", 1)
    _update(config, "critic.optim.lr", float(config_get(marlhf, "critic_learning_rate", 5.0e-6)))
    _update(
        config,
        "trajweave.comlrl.actor_critic",
        {
            "topology": topology,
            "critic_type": critic_type,
            "max_length": max_length,
            "value_loss_coef": float(config_get(marlhf, "critic_value_loss_coef", 0.6)),
            "critic_routes": routes,
        },
    )


def build_marlhf_preference_config(config: Any) -> DictConfig:
    if not is_marlhf_config(config):
        raise ValueError("MARLHF preference collection requires an enabled MARLHF config.")
    copied = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    marlhf = _select(copied, "trajweave.comlrl.marlhf", {}) or {}
    preference_candidates = int(
        config_get(
            marlhf,
            "preference_num_candidates",
            max(2, int(_select(copied, "actor_rollout_ref.rollout.n", 2))),
        )
    )
    if preference_candidates < 2:
        raise ValueError("MARLHF preference_num_candidates must be at least 2.")
    _update(copied, "trajweave.comlrl.algorithm", "madpo")
    _update(copied, "trajweave.comlrl.max_turns", 1)
    _update(copied, "trajweave.comlrl.joint_mode", "aligned")
    _update(copied, "trajweave.comlrl.marlhf.reward_model_active", False)
    _update(copied, "actor_rollout_ref.rollout.n", preference_candidates)
    _update(copied, "actor_rollout_ref.rollout.val_kwargs.n", preference_candidates)
    return copied


def marlhf_team_from_config(config: Any) -> TeamSpec:
    agent_config = _select(config, "agent", {}) or {}
    agent_ids = [str(value) for value in to_python(config_get(agent_config, "agent_ids", []))]
    model_ids = [str(value) for value in to_python(config_get(agent_config, "model_ids", []))]
    if not agent_ids or len(agent_ids) != len(model_ids):
        raise ValueError("MARLHF requires equally sized non-empty agent.agent_ids and agent.model_ids.")
    if len(set(agent_ids)) != len(agent_ids) or len(set(model_ids)) != len(model_ids):
        raise ValueError("MARLHF agent IDs and model IDs must each be unique.")
    return TeamSpec(
        name="comlrl_marlhf",
        agents=tuple(
            AgentSpec(agent_id, "solver", model_id, trainable=True)
            for agent_id, model_id in zip(agent_ids, model_ids, strict=True)
        ),
        policy_groups=tuple(PolicyGroupSpec(model_id, trainable=True) for model_id in model_ids),
        orchestra="comlrl_full_joint_tree",
        reward="marlhf_joint_reward_model",
        credit="marlhf",
        max_turns=1,
        metadata={"algorithm": "marlhf"},
    )


def preference_pairs_from_tq_batch(
    batch: Any,
    *,
    expected_worker_groups: Sequence[str],
) -> tuple[JointPreferencePair, ...]:
    extra_fields = (
        "agent_name",
        "traj_uid",
        "prompt_text",
        "response_text",
        "tree_node_id",
        "joint_action_ids",
        "candidate_mean",
    )
    iterative_fields = (
        "raw_policy_reward",
        "raw_comparator_reward",
        "raw_candidate_rewards",
        "policy_provenance",
        "comparator_provenance",
        "winner_source",
        "loser_source",
    )
    try:
        fields = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=(*PREFERENCE_TQ_FIELDS, *extra_fields, *iterative_fields),
        )
        has_iterative_fields = all(field in fields for field in iterative_fields)
    except KeyError:
        fields = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=(*PREFERENCE_TQ_FIELDS, *extra_fields),
        )
        has_iterative_fields = False
    active_masks = [float(to_python(batch_item(fields["preference_loss_mask"], row))) for row in range(len(batch.keys))]
    if active_masks and all(value == 0.0 for value in active_masks):
        return ()
    paired_rows = validate_preference_pairs(fields, expected_worker_groups=expected_worker_groups)
    rows_by_pair: dict[str, list[Any]] = {}
    for pair_rows in paired_rows:
        rows_by_pair.setdefault(pair_rows.preference_pair_id, []).append(pair_rows)

    output: list[JointPreferencePair] = []
    for pair_id in sorted(rows_by_pair):
        prompts: dict[str, str] = {}
        chosen: dict[str, str] = {}
        rejected: dict[str, str] = {}
        episode_ids: set[str] = set()
        tree_node_ids: set[str] = set()
        chosen_action_ids: set[str] = set()
        rejected_action_ids: set[str] = set()
        candidate_means: set[float] = set()
        chosen_rewards: set[float] = set()
        rejected_rewards: set[float] = set()
        raw_policy_rewards: set[float] = set()
        raw_comparator_rewards: set[float] = set()
        raw_candidate_rewards: set[str] = set()
        policy_provenance: set[str] = set()
        comparator_provenance: set[str] = set()
        winner_sources: set[str] = set()
        loser_sources: set[str] = set()
        for pair_rows in rows_by_pair[pair_id]:
            chosen_row, rejected_row = pair_rows.chosen_row, pair_rows.rejected_row
            chosen_agent = str(to_python(batch_item(fields["agent_name"], chosen_row)))
            rejected_agent = str(to_python(batch_item(fields["agent_name"], rejected_row)))
            if chosen_agent != rejected_agent or not chosen_agent:
                raise ValueError(f"MARLHF pair {pair_id!r} has inconsistent agent rows.")
            chosen_prompt = str(to_python(batch_item(fields["prompt_text"], chosen_row)))
            rejected_prompt = str(to_python(batch_item(fields["prompt_text"], rejected_row)))
            if chosen_prompt != rejected_prompt:
                raise ValueError(f"MARLHF pair {pair_id!r} chosen/rejected prompts must match.")
            prompts[chosen_agent] = chosen_prompt
            chosen[chosen_agent] = str(to_python(batch_item(fields["response_text"], chosen_row)))
            rejected[chosen_agent] = str(to_python(batch_item(fields["response_text"], rejected_row)))
            episode_ids.update(
                str(to_python(batch_item(fields["traj_uid"], row))) for row in (chosen_row, rejected_row)
            )
            tree_node_ids.update(
                str(to_python(batch_item(fields["tree_node_id"], row))) for row in (chosen_row, rejected_row)
            )
            chosen_action_ids.add(_single_action_id(fields["joint_action_ids"], chosen_row))
            rejected_action_ids.add(_single_action_id(fields["joint_action_ids"], rejected_row))
            candidate_means.update(
                float(to_python(batch_item(fields["candidate_mean"], row))) for row in (chosen_row, rejected_row)
            )
            chosen_rewards.add(pair_rows.chosen_reward)
            rejected_rewards.add(pair_rows.rejected_reward)
            if has_iterative_fields:
                for row in (chosen_row, rejected_row):
                    raw_policy_rewards.add(float(to_python(batch_item(fields["raw_policy_reward"], row))))
                    raw_comparator_rewards.add(float(to_python(batch_item(fields["raw_comparator_reward"], row))))
                    raw_candidate_rewards.add(_canonical_json(batch_item(fields["raw_candidate_rewards"], row)))
                    policy_provenance.add(_canonical_json(batch_item(fields["policy_provenance"], row)))
                    comparator_provenance.add(_canonical_json(batch_item(fields["comparator_provenance"], row)))
                    winner_sources.add(str(to_python(batch_item(fields["winner_source"], row))))
                    loser_sources.add(str(to_python(batch_item(fields["loser_source"], row))))
        if any(len(values) != 1 for values in (episode_ids, tree_node_ids, chosen_action_ids, rejected_action_ids)):
            raise ValueError(f"MARLHF pair {pair_id!r} contains inconsistent trajectory identifiers.")
        required_reward_values = (candidate_means, chosen_rewards, rejected_rewards)
        if any(len(values) != 1 for values in required_reward_values):
            raise ValueError(f"MARLHF pair {pair_id!r} contains inconsistent reward metadata.")
        if has_iterative_fields:
            iterative_values = (
                raw_policy_rewards,
                raw_comparator_rewards,
                raw_candidate_rewards,
                policy_provenance,
                comparator_provenance,
                winner_sources,
                loser_sources,
            )
            if any(len(values) != 1 for values in iterative_values):
                raise ValueError(f"MARLHF pair {pair_id!r} contains inconsistent iterative metadata.")
        output.append(
            JointPreferencePair(
                preference_pair_id=pair_id,
                episode_id=next(iter(episode_ids)),
                tree_node_id=next(iter(tree_node_ids)),
                chosen_joint_action_id=next(iter(chosen_action_ids)),
                rejected_joint_action_id=next(iter(rejected_action_ids)),
                prompts_by_agent=prompts,
                chosen_by_agent=chosen,
                rejected_by_agent=rejected,
                chosen_reward=next(iter(chosen_rewards)),
                rejected_reward=next(iter(rejected_rewards)),
                candidate_mean=next(iter(candidate_means)),
                metadata={
                    "source": "trajweave_transfer_queue",
                    **(
                        {
                            "raw_policy_reward": next(iter(raw_policy_rewards)),
                            "raw_comparator_reward": next(iter(raw_comparator_rewards)),
                            "raw_candidate_rewards": json.loads(next(iter(raw_candidate_rewards))),
                            "policy_provenance": json.loads(next(iter(policy_provenance))),
                            "comparator_provenance": json.loads(next(iter(comparator_provenance))),
                            "winner_source": next(iter(winner_sources)),
                            "loser_source": next(iter(loser_sources)),
                        }
                        if has_iterative_fields
                        else {}
                    ),
                },
            )
        )
    return tuple(output)


class PreferenceCollector(Protocol):
    def __call__(
        self,
        task_reward: Callable[..., float],
    ) -> Sequence[MultiAgentTrajectory]: ...


class OnlineRLRunner(Protocol):
    def __call__(
        self,
        *,
        dispatch: RLDispatchSpec,
        reward_router: MARLHFRewardRouter,
    ) -> Any: ...


class MARLHFRewardRouter:
    """Use the frozen joint RM online while preserving task reward for eval."""

    def __init__(self, scorer: JointRewardModelScorer, task_reward: Callable[..., float]) -> None:
        if not scorer.is_frozen:
            raise ValueError("Online MARLHF requires a frozen reward model scorer.")
        if not callable(task_reward):
            raise TypeError("task_reward must be callable.")
        self.scorer = scorer
        self.task_reward = task_reward

    def training_reward(
        self,
        prompts_by_agent: Mapping[str, str],
        responses_by_agent: Mapping[str, str],
    ) -> float:
        return self.scorer.score(prompts_by_agent, responses_by_agent)

    def evaluation_reward(self, *args: Any, **kwargs: Any) -> float:
        value = float(self.task_reward(*args, **kwargs))
        if value != value or value in {float("inf"), float("-inf")}:
            raise FloatingPointError("Task evaluation reward must be finite.")
        return value


class PreferenceCollectionStage:
    """Collect task-rewarded rollouts, then derive joint preference pairs."""

    def __init__(
        self,
        *,
        pair_builder: Callable[..., list[JointPreferencePair]] = build_joint_preference_pairs,
        pair_selection: str = "reward_gap",
        pairs_per_sample: int = 16,
    ) -> None:
        self.pair_builder = pair_builder
        self.pair_selection = pair_selection
        self.pairs_per_sample = pairs_per_sample

    def run(
        self,
        workflow: MARLHFStagedWorkflow,
        collector: PreferenceCollector,
    ) -> tuple[JointPreferencePair, ...]:
        workflow._require_state(MARLHFStageState.INITIAL, "preference collection")
        if not callable(collector):
            raise TypeError("Preference collector must be callable.")
        trajectories = tuple(collector(workflow.task_reward))
        if not trajectories:
            raise ValueError("Preference collection produced no task-rewarded trajectories.")
        pairs: list[JointPreferencePair] = []
        for trajectory in trajectories:
            pairs.extend(
                self.pair_builder(
                    trajectory,
                    pair_selection=self.pair_selection,
                    pairs_per_sample=self.pairs_per_sample,
                )
            )
        if not pairs:
            raise ValueError("Task rewards produced no non-tied joint preference pairs.")
        workflow.preference_pairs = tuple(pairs)
        workflow.state = MARLHFStageState.PREFERENCES_COLLECTED
        return workflow.preference_pairs


class RewardModelTrainingStage:
    """Train the pairwise joint RM and return a frozen serving scorer."""

    def __init__(self, *, epochs: int = 1, batch_size: int = 1) -> None:
        if epochs < 1 or batch_size < 1:
            raise ValueError("Reward model epochs and batch_size must be positive.")
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self, workflow: MARLHFStagedWorkflow) -> JointRewardModelScorer:
        workflow._require_state(MARLHFStageState.PREFERENCES_COLLECTED, "reward model training")
        worker = workflow.reward_worker
        if worker.is_frozen_for_evaluation:
            raise RuntimeError("Reward model worker was frozen before its training stage.")
        if worker.optimizer is None:
            raise RuntimeError("Reward model training stage requires a worker optimizer.")
        if not workflow.preference_pairs:
            raise RuntimeError("Reward model training stage requires collected preference pairs.")

        losses: list[float] = []
        pairs = workflow.preference_pairs
        for _epoch in range(self.epochs):
            for start in range(0, len(pairs), self.batch_size):
                loss = worker.train_preference_batch(pairs[start : start + self.batch_size])
                losses.append(float(loss.item()))
        worker.freeze_for_evaluation()
        scorer = JointRewardModelScorer(worker)
        if not scorer.is_frozen:
            raise RuntimeError("Reward model training did not produce a frozen evaluation scorer.")
        workflow.reward_model_losses = tuple(losses)
        workflow.reward_scorer = scorer
        workflow.state = MARLHFStageState.REWARD_MODEL_FROZEN
        return scorer


class OnlineRLStage:
    """Dispatch online MARL with RM training reward and task evaluation reward."""

    def run(
        self,
        workflow: MARLHFStagedWorkflow,
        *,
        algorithm: str,
        runner: OnlineRLRunner,
    ) -> Any:
        workflow._require_state(MARLHFStageState.REWARD_MODEL_FROZEN, "online RL")
        if workflow.reward_scorer is None or not workflow.reward_scorer.is_frozen:
            raise RuntimeError("Online RL cannot start without a frozen reward model scorer.")
        if not callable(runner):
            raise TypeError("Online RL runner must be callable.")
        dispatch = resolve_marlhf_rl_dispatch(algorithm)
        router = MARLHFRewardRouter(workflow.reward_scorer, workflow.task_reward)
        result = runner(dispatch=dispatch, reward_router=router)
        workflow.rl_dispatch = dispatch
        workflow.online_rl_result = result
        workflow.state = MARLHFStageState.ONLINE_RL_COMPLETED
        return result


class MARLHFStagedWorkflow:
    """Strict preference collection -> RM training -> online RL state machine."""

    def __init__(
        self,
        *,
        task_reward: Callable[..., float],
        reward_worker: RewardModelWorker,
        preference_stage: PreferenceCollectionStage | None = None,
        reward_model_stage: RewardModelTrainingStage | None = None,
        online_rl_stage: OnlineRLStage | None = None,
    ) -> None:
        if not callable(task_reward):
            raise TypeError("task_reward must be callable.")
        self.task_reward = task_reward
        self.reward_worker = reward_worker
        self.preference_stage = preference_stage or PreferenceCollectionStage()
        self.reward_model_stage = reward_model_stage or RewardModelTrainingStage()
        self.online_rl_stage = online_rl_stage or OnlineRLStage()
        self.state = MARLHFStageState.INITIAL
        self.preference_pairs: tuple[JointPreferencePair, ...] = ()
        self.reward_model_losses: tuple[float, ...] = ()
        self.reward_scorer: JointRewardModelScorer | None = None
        self.rl_dispatch: RLDispatchSpec | None = None
        self.online_rl_result: Any = None

    def collect_preferences(self, collector: PreferenceCollector) -> tuple[JointPreferencePair, ...]:
        return self.preference_stage.run(self, collector)

    def load_preference_pairs(self, pairs: Sequence[JointPreferencePair]) -> tuple[JointPreferencePair, ...]:
        self._require_state(MARLHFStageState.INITIAL, "preference loading")
        stable_pairs = tuple(pairs)
        if not stable_pairs:
            raise ValueError("MARLHF preference loading requires at least one pair.")
        for pair in stable_pairs:
            pair.validate()
        self.preference_pairs = stable_pairs
        self.state = MARLHFStageState.PREFERENCES_COLLECTED
        return stable_pairs

    def train_reward_model(self) -> JointRewardModelScorer:
        return self.reward_model_stage.run(self)

    def run_online_rl(self, *, algorithm: str, runner: OnlineRLRunner) -> Any:
        return self.online_rl_stage.run(self, algorithm=algorithm, runner=runner)

    def _require_state(self, expected: MARLHFStageState, operation: str) -> None:
        if self.state is not expected:
            raise RuntimeError(
                f"Cannot run {operation} while MARLHF workflow state is {self.state.value!r}; "
                f"expected {expected.value!r}."
            )


def collect_marlhf_preferences(trainer: Any, agent_loop_manager: Any, config: Any) -> tuple[JointPreferencePair, ...]:
    marlhf = _select(config, "trajweave.comlrl.marlhf", {}) or {}
    collection_batches = int(config_get(marlhf, "preference_collection_batches", 1))
    if collection_batches < 1:
        raise ValueError("MARLHF preference_collection_batches must be positive.")
    expected_groups = tuple(str(group) for group in trainer.multi_actor_trainable_group_ids)
    trainer.agent_loop_manager = agent_loop_manager
    output: list[JointPreferencePair] = []
    for _ in range(collection_batches):
        trainer._add_batch_to_generate()
        batch = trainer.replay_buffer.sample(
            partition_id="train",
            batch_size=int(config.data.train_batch_size),
        )
        try:
            output.extend(preference_pairs_from_tq_batch(batch, expected_worker_groups=expected_groups))
        finally:
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
    if not output:
        raise ValueError("MARLHF preference collection produced no non-tied pairs.")
    return tuple(output)


def train_marlhf_reward_model(config: Any, pairs: Sequence[JointPreferencePair]) -> MARLHFStagedWorkflow:
    team = marlhf_team_from_config(config)
    marlhf = _select(config, "trajweave.comlrl.marlhf", {}) or {}
    model_name = str(config_get(marlhf, "reward_model_name", "")).strip()
    if not model_name:
        raise ValueError("MARLHF requires trajweave.comlrl.marlhf.reward_model_name.")
    worker = RewardModelWorker.from_pretrained(
        model_name,
        team,
        freeze_backbone=bool(config_get(marlhf, "reward_freeze_backbone", False)),
        learning_rate=float(config_get(marlhf, "reward_learning_rate", 1.0e-5)),
        max_length=_optional_positive_int(config_get(marlhf, "reward_max_length", None), "reward_max_length"),
        device=str(config_get(marlhf, "reward_model_device", "cpu")),
        torch_dtype=config_get(marlhf, "reward_torch_dtype", "fp32"),
    )
    workflow = MARLHFStagedWorkflow(
        task_reward=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Task reward is only used during MARLHF preference collection.")
        ),
        reward_worker=worker,
        reward_model_stage=RewardModelTrainingStage(
            epochs=int(config_get(marlhf, "reward_num_train_epochs", 1)),
            batch_size=int(config_get(marlhf, "reward_train_batch_size", 1)),
        ),
    )
    workflow.load_preference_pairs(pairs)
    workflow.train_reward_model()
    checkpoint_value = config_get(marlhf, "reward_model_checkpoint", None)
    if checkpoint_value:
        checkpoint_path = Path(str(checkpoint_value))
    else:
        checkpoint_path = Path(str(_select(config, "trainer.default_local_dir", "checkpoints"))) / (
            "marlhf/reward_model.pt"
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    workflow.reward_worker.save_checkpoint(checkpoint_path)
    _update(config, "trajweave.comlrl.marlhf.reward_model_checkpoint", str(checkpoint_path))
    _update(config, "trajweave.comlrl.marlhf.reward_model_active", True)
    return workflow


def _canonical_json(value: Any) -> str:
    return json.dumps(to_python(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _single_action_id(values: Any, row: int) -> str:
    normalized = to_python(batch_item(values, row))
    if not isinstance(normalized, list | tuple) or len(normalized) != 1 or not str(normalized[0]):
        raise ValueError("MARLHF preference rows must reference exactly one joint action.")
    return str(normalized[0])


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"MARLHF {field} must be positive when configured.")
    return parsed


def _select(config: Any, path: str, default: Any = None) -> Any:
    try:
        value = OmegaConf.select(config, path)
    except (AttributeError, TypeError, ValueError):
        value = None
    if value is not None:
        return value
    current = config
    for part in path.split("."):
        current = config_get(current, part, None)
        if current is None:
            return default
    return current


def _update(config: Any, path: str, value: Any) -> None:
    if isinstance(config, DictConfig):
        with open_dict(config):
            OmegaConf.update(config, path, value, merge=False, force_add=True)
        return
    current = config
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise TypeError(f"Cannot update config path {path!r} on {type(current)!r}.")
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _ensure_verl_extension(config: Any, extension: str) -> None:
    raw = to_python(_select(config, "trajweave.verl_extensions", []))
    if raw is None:
        names: list[str] = []
    elif isinstance(raw, str):
        names = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, Sequence):
        names = [str(item).strip() for item in raw if str(item).strip()]
    else:
        raise TypeError(f"trajweave.verl_extensions must be a string or sequence, got {type(raw)!r}.")
    _update(config, "trajweave.verl_extensions", list(dict.fromkeys([*names, extension])))


__all__ = [
    "MARLHFRewardRouter",
    "MARLHFStageState",
    "MARLHFStagedWorkflow",
    "OnlineRLRunner",
    "build_marlhf_preference_config",
    "collect_marlhf_preferences",
    "is_marlhf_config",
    "marlhf_team_from_config",
    "preference_pairs_from_tq_batch",
    "prepare_marlhf_online_config",
    "OnlineRLStage",
    "PreferenceCollectionStage",
    "PreferenceCollector",
    "RLDispatchSpec",
    "RewardModelTrainingStage",
    "resolve_marlhf_rl_dispatch",
    "train_marlhf_reward_model",
]
