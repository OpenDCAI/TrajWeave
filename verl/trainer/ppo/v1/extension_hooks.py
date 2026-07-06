# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional PPO v1 extension hooks.

The default trainer path must stay identical for existing VERL algorithms. A
caller can opt in by setting ``algorithm.extension_hooks_class`` to a fully
qualified class name. The hook object is intentionally small: it can request
extra TransferQueue fields, move metadata into ``DataProto.non_tensor_batch``,
override advantage calculation, and add metrics.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from verl.protocol import DataProto
from verl.utils.import_utils import load_class_from_fqn


class PPOV1ExtensionHooks:
    """No-op extension contract for PPO v1 trainer integrations."""

    name = "default"

    def tq_select_fields(self, stage: str, default_fields: tuple[str, ...], config: Any = None) -> tuple[str, ...]:
        return default_fields

    def batch_schema_fields(self, stage: str, config: Any = None) -> tuple[str, ...]:
        return ()

    def prepare_dataproto(self, data: DataProto, *, stage: str, config: Any = None) -> DataProto:
        return data

    def compute_advantage(
        self,
        data: DataProto,
        *,
        batch_keys: list[str],
        adv_estimator: Any,
        gamma: float,
        lam: float,
        num_repeat: int,
        norm_adv_by_std_in_grpo: bool,
        config: Any = None,
        fallback: Callable[..., DataProto],
    ) -> DataProto:
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

    def output_fields(
        self,
        stage: str,
        default_fields: tuple[str, ...],
        data: DataProto,
        config: Any = None,
    ) -> tuple[str, ...]:
        return default_fields

    def update_metrics(self, data: DataProto, metrics: dict[str, Any], *, stage: str, config: Any = None) -> None:
        return None


def get_ppo_v1_extension_hooks(config: Any) -> PPOV1ExtensionHooks:
    hook_class_fqn = _select(config, "algorithm.extension_hooks_class")
    if hook_class_fqn is None:
        hook_class_fqn = _select(config, "trainer.v1.extension_hooks_class")
    if not hook_class_fqn:
        return PPOV1ExtensionHooks()
    hook_cls = load_class_from_fqn(str(hook_class_fqn), "PPOV1ExtensionHooks")
    return hook_cls()


def pop_tq_field_as_object_array(data: DataProto, key: str) -> np.ndarray:
    value = data.batch.pop(key)
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "data"):
        value = value.data
    return np.array(value, dtype=object)


def _select(config: Any, dotted_key: str, default: Any = None) -> Any:
    current = config
    for key in dotted_key.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
            continue
        try:
            current = current.get(key, default)
        except (AttributeError, TypeError):
            current = getattr(current, key, default)
    return current
