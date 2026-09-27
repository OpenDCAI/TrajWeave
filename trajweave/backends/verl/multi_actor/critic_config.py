from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trajweave.backends.verl.schema import to_python

_ALLOWED_TOPOLOGIES = {"independent", "centralized"}
_ALLOWED_CRITIC_TYPES = {"v", "q"}


@dataclass(frozen=True)
class CriticRouteSpec:
    critic_group: str
    actor_groups: tuple[str, ...]
    topology: str
    critic_type: str
    shared_with_actor: bool
    model_path: str | None
    tokenizer_path: str | None
    gpus: int
    max_length: int

    @property
    def group_id(self) -> str:
        return self.critic_group


def normalize_critic_route_specs(
    raw_routes: Any,
    *,
    trainable_actor_groups: Sequence[str],
    topology: str | None = None,
    critic_type: str | None = None,
    max_length: int = 2048,
) -> tuple[CriticRouteSpec, ...]:
    """Normalize list/mapping critic config into a stable routing contract."""

    actor_groups = tuple(dict.fromkeys(str(group) for group in trainable_actor_groups))
    if not actor_groups:
        raise ValueError("Critic routing requires at least one trainable actor group.")

    raw = to_python(raw_routes)
    common: dict[str, Any] = {}
    if isinstance(raw, Mapping) and ("groups" in raw or "routes" in raw):
        common = {key: value for key, value in raw.items() if key not in {"groups", "routes"}}
        raw = raw.get("groups", raw.get("routes"))
    if raw is None:
        raw = []
    if isinstance(raw, Mapping):
        raw = [{"critic_group": group_id, **dict(value or {})} for group_id, value in raw.items()]
    if not isinstance(raw, list | tuple):
        raise ValueError("critic routes must be a list or a mapping keyed by critic group.")

    default_topology = _normalize_choice(
        topology or common.get("topology", "independent"), "topology", _ALLOWED_TOPOLOGIES
    )
    default_critic_type = _normalize_choice(
        critic_type or common.get("critic_type", "v"), "critic_type", _ALLOWED_CRITIC_TYPES
    )
    routes: list[CriticRouteSpec] = []
    for index, item in enumerate(raw):
        item = to_python(item)
        if not isinstance(item, Mapping):
            raise ValueError(f"critic route {index} must be a mapping.")
        route = {**common, **dict(item)}
        route_topology = _normalize_choice(route.get("topology", default_topology), "topology", _ALLOWED_TOPOLOGIES)
        route_critic_type = _normalize_choice(
            route.get("critic_type", default_critic_type), "critic_type", _ALLOWED_CRITIC_TYPES
        )
        group_id = route.get("critic_group", route.get("id", route.get("group_id")))
        if group_id is None and route_topology == "independent":
            group_id = route.get("actor_group")
        if group_id is None and route_topology == "centralized":
            group_id = "centralized"
        if not str(group_id or "").strip():
            raise ValueError(f"critic route {index} requires a non-empty critic_group/id.")

        covered = route.get("actor_groups")
        if covered is None:
            actor_group = route.get("actor_group")
            if actor_group is not None:
                covered = [actor_group]
            elif route_topology == "independent" and str(group_id) in actor_groups:
                covered = [str(group_id)]
            elif route_topology == "centralized":
                covered = actor_groups
        if isinstance(covered, str):
            covered = [covered]
        if not isinstance(covered, list | tuple) or not covered:
            raise ValueError(f"critic route {group_id!r} requires non-empty actor_groups.")

        routes.append(
            CriticRouteSpec(
                critic_group=str(group_id),
                actor_groups=tuple(dict.fromkeys(str(group) for group in covered)),
                topology=route_topology,
                critic_type=route_critic_type,
                shared_with_actor=bool(route.get("shared_with_actor", False)),
                model_path=_optional_path(route.get("model_path")),
                tokenizer_path=_optional_path(route.get("tokenizer_path")),
                gpus=_positive_gpu_count(route.get("gpus", 1), group_id=str(group_id)),
                max_length=_positive_integer(
                    route.get("max_length", common.get("max_length", max_length)),
                    field=f"Critic group {group_id!r} max_length",
                ),
            )
        )

    return validate_critic_route_specs(routes, trainable_actor_groups=actor_groups)


def validate_critic_route_specs(
    routes: Sequence[CriticRouteSpec],
    *,
    trainable_actor_groups: Sequence[str],
    allow_shared_with_actor: bool = False,
) -> tuple[CriticRouteSpec, ...]:
    """Validate IAC/MAAC topology, exact actor coverage, and worker resources."""

    normalized = tuple(routes)
    actors = tuple(dict.fromkeys(str(group) for group in trainable_actor_groups))
    if not normalized:
        raise ValueError("At least one critic route is required.")
    group_ids = [route.critic_group for route in normalized]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError(f"critic_group values must be unique; got {group_ids}.")
    topologies = {route.topology for route in normalized}
    if len(topologies) != 1:
        raise ValueError(f"All critic routes must use one topology; got {sorted(topologies)}.")
    critic_types = {route.critic_type for route in normalized}
    if len(critic_types) != 1:
        raise ValueError(f"All critic routes must use one critic_type; got {sorted(critic_types)}.")

    for route in normalized:
        _normalize_choice(route.topology, "topology", _ALLOWED_TOPOLOGIES)
        _normalize_choice(route.critic_type, "critic_type", _ALLOWED_CRITIC_TYPES)
        _positive_gpu_count(route.gpus, group_id=route.critic_group)
        _positive_integer(route.max_length, field=f"Critic group {route.critic_group!r} max_length")
        unknown = sorted(set(route.actor_groups) - set(actors))
        if unknown:
            raise ValueError(f"Critic group {route.critic_group!r} references unknown actor groups: {unknown}.")
        if route.shared_with_actor and not allow_shared_with_actor:
            raise ValueError(
                "shared_with_actor=true is not supported by TrajWeave's current VERL worker: "
                "a real actor value-head worker is required. Configure a separate critic."
            )
        if not route.shared_with_actor:
            missing = []
            if not route.model_path:
                missing.append("model_path")
            if not route.tokenizer_path:
                missing.append("tokenizer_path")
            if missing:
                raise ValueError(f"Separate critic group {route.critic_group!r} requires {', '.join(missing)}.")

    topology_name = next(iter(topologies))
    if topology_name == "independent":
        invalid = [route.critic_group for route in normalized if len(route.actor_groups) != 1]
        if invalid:
            raise ValueError(f"Independent critics must cover exactly one actor group each: {invalid}.")
        coverage = [route.actor_groups[0] for route in normalized]
        if sorted(coverage) != sorted(actors) or len(coverage) != len(actors):
            raise ValueError(
                "IAC independent topology requires exactly one critic for every trainable actor group; "
                f"actors={list(actors)}, coverage={coverage}."
            )
    else:
        if len(normalized) != 1:
            raise ValueError("MAAC centralized topology requires exactly one critic group.")
        coverage = normalized[0].actor_groups
        if set(coverage) != set(actors) or len(coverage) != len(actors):
            raise ValueError(
                "MAAC centralized critic must cover all trainable actor groups exactly once; "
                f"actors={list(actors)}, coverage={list(coverage)}."
            )
    return normalized


def resolve_actor_critic_settings(config: Any, *, required: bool = False) -> dict[str, Any] | None:
    """Resolve the two supported runtime config paths without recipe-specific imports."""

    root = to_python(config)
    if not isinstance(root, Mapping):
        if required:
            raise ValueError("Actor-critic runtime requires a mapping configuration.")
        return None
    trajweave = root.get("trajweave")
    if not isinstance(trajweave, Mapping):
        if required:
            raise ValueError(
                "Actor-critic runtime requires trajweave.actor_critic or trajweave.comlrl.actor_critic; "
                "algorithm.actor_critic and trajweave.multi_actor.critic_routes are unsupported."
            )
        return None
    direct = trajweave.get("actor_critic")
    comlrl = trajweave.get("comlrl")
    nested = comlrl.get("actor_critic") if isinstance(comlrl, Mapping) else None
    if direct is not None and nested is not None:
        raise ValueError(
            "Configure actor-critic runtime at exactly one path: "
            "trajweave.actor_critic or trajweave.comlrl.actor_critic."
        )
    settings = direct if direct is not None else nested
    if settings is None:
        if required:
            raise ValueError(
                "Actor-critic runtime requires trajweave.actor_critic or trajweave.comlrl.actor_critic; "
                "algorithm.actor_critic and trajweave.multi_actor.critic_routes are unsupported."
            )
        return None
    settings = to_python(settings)
    if not isinstance(settings, Mapping):
        raise TypeError("Actor-critic runtime settings must be a mapping.")
    return dict(settings)


# Short aliases used by launch/config callers.
normalize_critic_routes = normalize_critic_route_specs
validate_critic_routes = validate_critic_route_specs


def _normalize_choice(value: Any, field: str, allowed: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}; got {value!r}.")
    return normalized


def _optional_path(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _positive_gpu_count(value: Any, *, group_id: str) -> int:
    return _positive_integer(value, field=f"Critic group {group_id!r} gpus")


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer.")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer.") from exc
    if count <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{field} must be a positive integer.")
    return count


__all__ = [
    "CriticRouteSpec",
    "normalize_critic_route_specs",
    "normalize_critic_routes",
    "resolve_actor_critic_settings",
    "validate_critic_route_specs",
    "validate_critic_routes",
]
