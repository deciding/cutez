from dataclasses import dataclass, field
import inspect
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class Config:
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None
    pre_hook: Callable[..., Any] | None = None

    def __post_init__(self):
        object.__setattr__(self, "kwargs", _normalize_config_value(self.kwargs))


@dataclass(frozen=True)
class AutotuneSpec:
    configs: tuple[Config, ...]
    key: tuple[str, ...]
    warmup: int = 0
    rep: int = 0
    cache_results: bool = True
    force_retune: bool = False
    cache_path: Path | None = None
    do_bench: Callable[..., Any] | None = None


class AutotuneError(RuntimeError):
    pass


def autotune(
    *,
    configs: list[Config],
    key: list[str],
    warmup: int = 0,
    rep: int = 0,
    cache_results: bool = True,
    force_retune: bool = False,
    cache_path: str | PathLike[str] | None = None,
    do_bench: Callable[..., Any] | None = None,
):
    if not configs:
        raise ValueError("autotune requires at least one config")

    normalized_cache_path = None if cache_path is None else Path(cache_path)

    spec = AutotuneSpec(
        configs=tuple(configs),
        key=tuple(key),
        warmup=warmup,
        rep=rep,
        cache_results=cache_results,
        force_retune=force_retune,
        cache_path=normalized_cache_path,
        do_bench=do_bench,
    )

    def decorator(fn):
        setattr(fn, "__cutez_autotune__", spec)
        return fn

    return decorator


def read_autotune_spec(kernel) -> AutotuneSpec | None:
    spec = getattr(kernel, "__cutez_autotune__", None)
    if spec is not None:
        return spec

    call = getattr(kernel, "__call__", None)
    return getattr(call, "__cutez_autotune__", None)


def autotune_spec_applies_to_call(kernel, spec: AutotuneSpec | None) -> bool:
    if spec is None:
        return False
    return not inspect.isfunction(kernel)


def get_autotune_spec(kernel) -> AutotuneSpec | None:
    return read_autotune_spec(kernel)


def config_identity(config: Config) -> tuple[tuple[str, Any], ...]:
    return (
        ("kwargs", freeze_for_cache(config.kwargs)),
        ("name", freeze_for_cache(config.name)),
        ("pre_hook", freeze_for_cache(config.pre_hook)),
    )


def _normalize_config_value(value):
    return _normalize_config_value_impl(value, seen=set())


def _normalize_config_value_impl(value, seen):
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in seen:
            return value
        seen.add(value_id)
        try:
            return {k: _normalize_config_value_impl(v, seen) for k, v in value.items()}
        finally:
            seen.remove(value_id)
    if isinstance(value, list):
        value_id = id(value)
        if value_id in seen:
            return value
        seen.add(value_id)
        try:
            return tuple(_normalize_config_value_impl(item, seen) for item in value)
        finally:
            seen.remove(value_id)
    if isinstance(value, tuple):
        value_id = id(value)
        if value_id in seen:
            return value
        seen.add(value_id)
        try:
            return tuple(_normalize_config_value_impl(item, seen) for item in value)
        finally:
            seen.remove(value_id)
    return value


def freeze_for_cache(value):
    if isinstance(value, dict):
        return tuple(
            sorted((freeze_for_cache(k), freeze_for_cache(v)) for k, v in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_for_cache(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(freeze_for_cache(item) for item in value))
    try:
        hash(value)
    except TypeError:
        attrs = getattr(value, "__dict__", None)
        if attrs is not None:
            return (
                type(value),
                tuple(
                    sorted(
                        (freeze_for_cache(k), freeze_for_cache(v))
                        for k, v in attrs.items()
                    )
                ),
            )
        return (type(value), repr(value))
    return value
