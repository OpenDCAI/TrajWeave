from trajweave.credit.comlrl.actor_critic import (
    OneStepTD,
    build_iac_critic_text,
    build_iac_q_critic_text,
    build_iac_v_critic_text,
    build_maac_critic_text,
    build_maac_q_critic_text,
    build_maac_v_critic_text,
    deduplicate_maac_critic_indices,
    deduplicate_maac_critic_samples,
    iac_shared_head_clipped_value_loss,
    normalize_agent_turn_advantages,
    one_step_td_targets,
    ratio_free_sequence_actor_loss,
    unclipped_mse_critic_loss,
)
from trajweave.credit.comlrl.iterative import (
    IndexedPolicyComparison,
    compare_policy_candidates_by_index,
    select_policy_comparisons,
)
from trajweave.credit.comlrl.marlhf import JointPreferenceBatch, JointRewardModelScorer
from trajweave.credit.comlrl.preference import (
    build_joint_preference_pairs,
    preference_pairs_to_training_samples,
)
from trajweave.credit.comlrl.reinforce import (
    CoMLRLReinforceCreditAssigner,
    CompletionReturnProjection,
    apply_sequence_kl,
    compute_group_advantages,
    compute_joint_tree_returns,
    project_joint_returns_to_completions,
)

__all__ = [
    "CoMLRLReinforceCreditAssigner",
    "CompletionReturnProjection",
    "IndexedPolicyComparison",
    "JointPreferenceBatch",
    "JointRewardModelScorer",
    "OneStepTD",
    "apply_sequence_kl",
    "build_iac_critic_text",
    "build_iac_q_critic_text",
    "build_iac_v_critic_text",
    "build_maac_critic_text",
    "build_maac_q_critic_text",
    "build_maac_v_critic_text",
    "build_joint_preference_pairs",
    "compare_policy_candidates_by_index",
    "compute_group_advantages",
    "compute_joint_tree_returns",
    "deduplicate_maac_critic_indices",
    "deduplicate_maac_critic_samples",
    "iac_shared_head_clipped_value_loss",
    "normalize_agent_turn_advantages",
    "one_step_td_targets",
    "preference_pairs_to_training_samples",
    "project_joint_returns_to_completions",
    "ratio_free_sequence_actor_loss",
    "select_policy_comparisons",
    "unclipped_mse_critic_loss",
]
