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
    if len(parent_reqs_ids) != row_count or len(is_child) != row_count:
        raise MATPOParentChildIntegrityError(
            "MATPO reqs_id, parent_reqs_id, and is_from_subagent_tool arrays must all match the batch size."
        )

    def traj_uid_for(row: int) -> str:
        if traj_uids is None or row >= len(traj_uids):
            return "<unknown>"
        return str(traj_uids[row])

    seen_reqs_ids: dict[str, int] = {}
    for row in range(row_count):
        reqs_id = str(reqs_ids[row])
        if not reqs_id:
            raise MATPOParentChildIntegrityError(
                f"Missing reqs_id at row {row} (traj_uid={traj_uid_for(row)!r}); every MATPO row needs an ID."
            )
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


def _child_marker(value: Any, *, row: int) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise MATPOParentChildIntegrityError(
        f"Invalid is_from_subagent_tool marker at row {row}: expected bool or 0/1, got {value!r}."
    )


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
        row_count = int(data.batch.batch_size[0])
        is_child_values = data.non_tensor_batch.get("is_from_subagent_tool")
        if is_child_values is None and "is_from_subagent_tool" in data.batch.keys():
            is_child_values = data.batch["is_from_subagent_tool"].detach().cpu().tolist()
        if is_child_values is None:
            raise MATPOParentChildIntegrityError(
                "MATPO batch is missing the is_from_subagent_tool marker array; refusing unsafe fallback."
            )
        if len(is_child_values) != row_count:
            raise MATPOParentChildIntegrityError("MATPO is_from_subagent_tool marker array must match the batch size.")
        is_child = np.array([_child_marker(value, row=row) for row, value in enumerate(is_child_values)])

        raw_reqs_ids = data.non_tensor_batch.get("reqs_id")
        raw_parent_reqs_ids = data.non_tensor_batch.get("parent_reqs_id")
        if raw_reqs_ids is None or raw_parent_reqs_ids is None:
            raise MATPOParentChildIntegrityError(
                "MATPO batch is missing reqs_id or parent_reqs_id arrays; refusing unsafe fallback."
            )
        if len(raw_reqs_ids) != row_count or len(raw_parent_reqs_ids) != row_count:
            raise MATPOParentChildIntegrityError("MATPO reqs_id and parent_reqs_id arrays must match the batch size.")
        reqs_ids = np.array([str(value) for value in raw_reqs_ids], dtype=object)
        parent_reqs_ids = np.array([str(value) for value in raw_parent_reqs_ids], dtype=object)
        if is_child.all():
            raise MATPOParentChildIntegrityError(
                "MATPO batch contains only child rows and has no main planner row for advantage assignment."
            )
        raw_traj_uids = data.non_tensor_batch.get("traj_uid")
        raw_turn_counts = data.non_tensor_batch.get("turn_count")
        if raw_turn_counts is None and "turn_count" in data.batch.keys():
            raw_turn_counts = data.batch["turn_count"].detach().cpu().tolist()
        if raw_traj_uids is None or raw_turn_counts is None:
            raise MATPOParentChildIntegrityError(
                "MATPO batch is missing traj_uid or turn_count arrays required for "
                "trajectory-level advantage assignment."
            )
        if len(raw_traj_uids) != row_count or len(raw_turn_counts) != row_count:
            raise MATPOParentChildIntegrityError("MATPO traj_uid and turn_count arrays must match the batch size.")
        traj_uids = np.array([str(value) for value in raw_traj_uids], dtype=object)
        turn_counts = np.array(
            [int(value.item() if hasattr(value, "item") else value) for value in raw_turn_counts], dtype=np.int64
        )
        _validate_matpo_parent_child_links(
            reqs_ids=reqs_ids,
            parent_reqs_ids=parent_reqs_ids,
            is_child=is_child,
            traj_uids=traj_uids,
        )

        main_indices = np.where(~is_child)[0]
        representative_by_traj: dict[str, int] = {}
        for row in main_indices:
            traj_uid = traj_uids[row]
            previous = representative_by_traj.get(traj_uid)
            if previous is None or turn_counts[row] > turn_counts[previous]:
                representative_by_traj[traj_uid] = int(row)
        representative_indices = np.array(list(representative_by_traj.values()), dtype=np.int64)
        representative_mask = np.zeros(row_count, dtype=bool)
        representative_mask[representative_indices] = True
        representative_batch_keys = (
            [batch_keys[int(index)] for index in representative_indices] if batch_keys is not None else None
        )
        representative_data = fallback(
            data[representative_mask],
            batch_keys=representative_batch_keys,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

        advantages = torch.zeros_like(data.batch["token_level_rewards"])
        returns = torch.zeros_like(data.batch["token_level_rewards"])
        trajectory_scalars: dict[str, tuple[Any, Any]] = {}
        for output_row, source_row in enumerate(representative_indices):
            response_mask = representative_data.batch["response_mask"][output_row].bool()
            if response_mask.any():
                advantage_scalar = representative_data.batch["advantages"][output_row][response_mask][0]
                return_scalar = representative_data.batch["returns"][output_row][response_mask][0]
            else:
                advantage_scalar = advantages.new_zeros(())
                return_scalar = returns.new_zeros(())
            trajectory_scalars[traj_uids[source_row]] = (advantage_scalar, return_scalar)

        for row in main_indices:
            advantage_scalar, return_scalar = trajectory_scalars[traj_uids[row]]
            response_mask = data.batch["response_mask"][row]
            advantages[row] = advantage_scalar * response_mask
            returns[row] = return_scalar * response_mask

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
