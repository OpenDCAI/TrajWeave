from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.extensions.common.hooks import _DEFAULT_ADVANTAGE_TQ_FIELDS, PPOExtensionHooks


class MATPOParentChildIntegrityError(ValueError):
    """Raised when MATPO parent/child request linkage is malformed before credit assignment."""


def _validate_matpo_parent_child_links(
    *,
    reqs_ids: Any,
    parent_reqs_ids: Any,
    is_child: Any,
    traj_uids: Any = None,
) -> None:
    """Validate MATPO parent/child request linkage before advantage broadcast.

    Checks:
      * every non-empty ``reqs_id`` is unique;
      * every child's ``parent_reqs_id`` resolves to a real main-agent row;
      * there is no self-reference or parent/child cycle;
      * the current single-level planner/worker protocol holds (a child cannot
        itself be referenced as somebody else's parent);
      * there are no orphan children (children whose parent id does not exist).

    Raises:
        MATPOParentChildIntegrityError: with ``reqs_id``, ``parent_reqs_id`` and
            ``traj_uid`` (when available) identifying the offending row(s).
    """
    row_count = len(reqs_ids)

    def traj_uid_for(row: int) -> str:
        if traj_uids is None or row >= len(traj_uids):
            return "<unknown>"
        return str(traj_uids[row])

    seen_reqs_ids: dict[str, int] = {}
    for row in range(row_count):
        reqs_id = str(reqs_ids[row])
        if not reqs_id:
            continue
        if reqs_id in seen_reqs_ids:
            other_row = seen_reqs_ids[reqs_id]
            raise MATPOParentChildIntegrityError(
                f"Duplicate reqs_id={reqs_id!r} found at rows {other_row} and {row} "
                f"(traj_uid={traj_uid_for(row)!r}). reqs_id must be unique within a batch."
            )
        seen_reqs_ids[reqs_id] = row

    reqs_id_to_row = {str(reqs_ids[row]): row for row in range(row_count) if str(reqs_ids[row])}
    main_rows = {row for row in range(row_count) if not bool(is_child[row])}

    for row in range(row_count):
        if not bool(is_child[row]):
            continue
        reqs_id = str(reqs_ids[row])
        parent_reqs_id = str(parent_reqs_ids[row]) if row < len(parent_reqs_ids) else ""
        if not parent_reqs_id:
            raise MATPOParentChildIntegrityError(
                f"Orphan child at row {row} (reqs_id={reqs_id!r}, traj_uid={traj_uid_for(row)!r}): "
                "is_from_subagent_tool=True but parent_reqs_id is empty."
            )
        if parent_reqs_id == reqs_id:
            raise MATPOParentChildIntegrityError(
                f"Self-referencing MATPO row at row {row} (reqs_id={reqs_id!r}, "
                f"traj_uid={traj_uid_for(row)!r}): parent_reqs_id equals its own reqs_id."
            )
        parent_row = reqs_id_to_row.get(parent_reqs_id)
        if parent_row is None:
            raise MATPOParentChildIntegrityError(
                f"Orphan child at row {row} (reqs_id={reqs_id!r}, parent_reqs_id={parent_reqs_id!r}, "
                f"traj_uid={traj_uid_for(row)!r}): no row in this batch has that reqs_id."
            )
        if parent_row not in main_rows:
            raise MATPOParentChildIntegrityError(
                f"Illegal MATPO linkage at row {row} (reqs_id={reqs_id!r}, "
                f"parent_reqs_id={parent_reqs_id!r}, traj_uid={traj_uid_for(row)!r}): the referenced "
                f"parent row {parent_row} is itself a child. MATPO's planner/worker protocol only "
                "supports a single parent -> child level; chained delegation is not allowed."
            )

    # Cycle detection: with the single-level constraint above already enforced,
    # a cycle can only occur between two "main" rows referencing each other via
    # parent_reqs_id, which would otherwise slip past the child-only checks.
    visited: set[str] = set()
    for start_reqs_id in list(reqs_id_to_row):
        if start_reqs_id in visited:
            continue
        chain: list[str] = []
        current = start_reqs_id
        while current:
            if current in chain:
                cycle_row = reqs_id_to_row[current]
                raise MATPOParentChildIntegrityError(
                    f"Cyclic MATPO parent/child reference detected involving reqs_id={current!r} "
                    f"(row {cycle_row}, traj_uid={traj_uid_for(cycle_row)!r}): chain={chain + [current]}."
                )
            chain.append(current)
            visited.add(current)
            current_row = reqs_id_to_row[current]
            current = str(parent_reqs_ids[current_row]) if current_row < len(parent_reqs_ids) else ""
            if current and current not in reqs_id_to_row:
                break


@dataclass(frozen=True)
class MATPOParentBroadcastHooks(PPOExtensionHooks):
    name: str = "matpo_parent_broadcast"

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        if stage != "advantage":
            return ()
        return (
            "reqs_id",
            "parent_reqs_id",
            "is_from_subagent_tool",
            "turn_count",
            "agent_type",
            "role_id",
            "shared_model_id",
            "matpo_tool_format_valid",
            "matpo_tool_call_count",
        )

    def tq_select_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...] | None = None,
        config: Any = None,
    ) -> tuple[str, ...]:
        if default_fields is None:
            default_fields = _DEFAULT_ADVANTAGE_TQ_FIELDS if stage == "advantage" else ()
        fields = list(super().tq_select_fields(stage, default_fields=default_fields, config=config))
        if stage == "advantage":
            fields.extend(self.batch_schema_fields(stage, config=config))
        return tuple(dict.fromkeys(fields))

    def compute_advantage(
        self,
        data: Any,
        *,
        batch_keys: list[str] | None = None,
        adv_estimator: Any,
        gamma: float = 1.0,
        lam: float = 1.0,
        num_repeat: int = 1,
        norm_adv_by_std_in_grpo: bool = True,
        config: Any = None,
        fallback: Any = None,
    ) -> Any:
        import numpy as np
        import torch

        if fallback is None:
            raise ValueError("MATPOParentBroadcastHooks requires a fallback advantage implementation.")
        is_child_values = data.non_tensor_batch.get("is_from_subagent_tool")
        if is_child_values is None and "is_from_subagent_tool" in data.batch.keys():
            is_child_values = data.batch["is_from_subagent_tool"].detach().cpu().tolist()
        if is_child_values is None:
            return fallback(
                data,
                batch_keys=batch_keys,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

        is_child = np.array([bool(value) for value in is_child_values])
        if not is_child.any():
            return fallback(
                data,
                batch_keys=batch_keys,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

        main_mask = ~is_child
        if not main_mask.any():
            data.batch["advantages"] = torch.zeros_like(data.batch["token_level_rewards"])
            data.batch["returns"] = torch.zeros_like(data.batch["token_level_rewards"])
            return data

        reqs_ids = np.array([str(value) for value in data.non_tensor_batch.get("reqs_id", [])], dtype=object)
        parent_reqs_ids = np.array(
            [str(value) for value in data.non_tensor_batch.get("parent_reqs_id", [])], dtype=object
        )
        traj_uids = data.non_tensor_batch.get("traj_uid")
        _validate_matpo_parent_child_links(
            reqs_ids=reqs_ids,
            parent_reqs_ids=parent_reqs_ids,
            is_child=is_child,
            traj_uids=traj_uids,
        )

        main_indices = np.where(main_mask)[0]
        main_batch_keys = [batch_keys[int(index)] for index in main_indices] if batch_keys is not None else None
        main_data = data[main_mask]
        main_data = fallback(
            main_data,
            batch_keys=main_batch_keys,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

        advantages = torch.zeros_like(data.batch["token_level_rewards"])
        returns = torch.zeros_like(data.batch["token_level_rewards"])
        for output_row, source_row in enumerate(main_indices):
            advantages[source_row] = main_data.batch["advantages"][output_row]
            returns[source_row] = main_data.batch["returns"][output_row]

        req_to_row = {reqs_ids[row]: row for row in main_indices if row < len(reqs_ids)}
        for row in np.where(is_child)[0]:
            # Validation above already guarantees every child resolves to a real main row.
            parent_row = req_to_row[parent_reqs_ids[row]]
            parent_response_mask = data.batch["response_mask"][parent_row].bool()
            if parent_response_mask.any():
                parent_advantage_scalar = advantages[parent_row][parent_response_mask][0]
                parent_return_scalar = returns[parent_row][parent_response_mask][0]
            else:
                parent_advantage_scalar = advantages.new_zeros(())
                parent_return_scalar = returns.new_zeros(())
            child_response_mask = data.batch["response_mask"][row]
            advantages[row] = parent_advantage_scalar * child_response_mask
            returns[row] = parent_return_scalar * child_response_mask

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data


def apply_matpo_parent_broadcast_patch(config: Any = None) -> None:
    """启用 MATPO parent-broadcast 所需的 TQ 字段与权重同步，不改写 VERL 默认算法。"""

    from trajweave.backends.verl.extensions.common.runtime import apply_hook_aware_tq_runtime

    apply_hook_aware_tq_runtime(config)
