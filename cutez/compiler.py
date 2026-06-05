import inspect
import json

import cutlass.cute as cute

from .autotune import (
    AutotuneError,
    Config,
    autotune_spec_applies_to_call,
    config_identity,
    freeze_for_cache,
    read_autotune_spec,
)
from .benchmark import benchmark


_COMPILE_CACHE = {}
_BEST_CONFIG_CACHE = {}


def _kernel_identity(kernel):
    if inspect.isfunction(kernel):
        return kernel
    return type(kernel)


def _config_label(config):
    return config.name or repr(dict(config.kwargs))


def _log_verbose(verbose, message):
    if verbose:
        print(message)


def _format_candidate_failures(failures):
    lines = ["all autotune candidates failed"]
    for label, exc in failures:
        lines.append(f"- {label}: {type(exc).__name__}: {exc}")
    return "\n".join(lines)


def _tuning_key(kernel, spec, runtime_key_values):
    missing_name = next(
        (name for name in spec.key if name not in runtime_key_values), None
    )
    if missing_name is not None:
        available_keys = ", ".join(sorted(runtime_key_values)) or "none"
        raise AutotuneError(
            f"missing autotune key field '{missing_name}' in resolved key values; "
            f"available keys: {available_keys}"
        )
    return (
        _kernel_identity(kernel),
        tuple(runtime_key_values[name] for name in spec.key),
    )


def _resolve_key_values(kernel, spec, *args, **kwargs):
    sig = inspect.signature(kernel)
    param_names = list(sig.parameters.keys())
    nargs = dict(zip(param_names, args))
    all_args = {**nargs, **kwargs}
    result = {}
    for name in spec.key:
        if name not in all_args:
            available = ", ".join(sorted(all_args)) or "none"
            raise AutotuneError(
                f"missing autotune key field '{name}' in resolved key values; "
                f"available keys: {available}"
            )
        result[name] = all_args[name]
    return result


def _compile_signature(args, kwargs):
    return (freeze_for_cache(args), freeze_for_cache(kwargs))


def _compile_cache_key(kernel, candidate_kwargs, config, args, kwargs):
    return (
        _kernel_identity(kernel),
        config_identity(config),
        freeze_for_cache(candidate_kwargs),
        _compile_signature(args, kwargs),
    )


def _get_cached_compiled_candidate(kernel, candidate_kwargs, config, args, kwargs):
    cache_key = _compile_cache_key(kernel, candidate_kwargs, config, args, kwargs)
    return _COMPILE_CACHE.get(cache_key), cache_key


def _compile_candidate(candidate_kernel, cache_key, args, kwargs):
    compiled = cute.compile(candidate_kernel, *args, **kwargs)
    _COMPILE_CACHE[cache_key] = compiled
    return compiled


def _reconstruct_candidate(kernel, candidate_kwargs):
    if inspect.isfunction(kernel):
        return kernel
    return type(kernel)(**candidate_kwargs)


def _compile_target_and_kwargs(kernel, candidate_kwargs, kwargs):
    if inspect.isfunction(kernel):
        return kernel, {**kwargs, **candidate_kwargs}
    return _reconstruct_candidate(kernel, candidate_kwargs), kwargs


def _pre_hook_kwargs(kernel, candidate_kwargs, kwargs):
    if inspect.isfunction(kernel):
        return {**kwargs, **candidate_kwargs}
    return kwargs


def _benchmark_args_and_kwargs(kernel, compiled, args, kwargs):
    if not inspect.isfunction(kernel) or not callable(compiled):
        return args, kwargs

    kernel_signature = inspect.signature(kernel)
    runtime_kwargs = {}
    runtime_args = []
    arg_index = 0
    for param in kernel_signature.parameters.values():
        annotation_name = getattr(param.annotation, "__name__", None)
        is_compile_time = annotation_name == "Constexpr"

        if param.kind == param.KEYWORD_ONLY:
            if param.name in kwargs and not is_compile_time:
                runtime_kwargs[param.name] = kwargs[param.name]
            continue

        if param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            continue

        if param.name in kwargs:
            if not is_compile_time:
                runtime_kwargs[param.name] = kwargs[param.name]
            continue

        if arg_index >= len(args):
            continue

        if is_compile_time:
            arg_index += 1
            continue

        runtime_args.append(args[arg_index])
        arg_index += 1
    return tuple(runtime_args), runtime_kwargs


def _persistent_cache_enabled(spec):
    return (
        spec.cache_results
        and spec.cache_path is not None
        and all(config.pre_hook is None for config in spec.configs)
    )


def _stable_kernel_identifier(kernel):
    target = (
        kernel
        if hasattr(kernel, "__module__") and hasattr(kernel, "__qualname__")
        else None
    )
    if target is None:
        target = getattr(type(kernel), "__call__", None)
    if target is None:
        return None

    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not module or not qualname:
        return None
    return f"{module}.{qualname}"


def _normalize_persisted_value(value):
    if isinstance(value, list):
        return tuple(_normalize_persisted_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_persisted_value(item) for key, item in value.items()}
    return value


def _persisted_entry_matches(entry, kernel_id, tuning_values):
    return (
        isinstance(entry, dict)
        and entry.get("kernel") == kernel_id
        and _normalize_persisted_value(entry.get("key")) == tuning_values
    )


def _load_persistent_entries(cache_path):
    try:
        payload = json.loads(cache_path.read_text())
    except FileNotFoundError:
        return [], False
    except (OSError, json.JSONDecodeError):
        return [], True

    if not isinstance(payload, dict):
        return [], True

    entries = payload.get("entries")
    if not isinstance(entries, list):
        return [], True
    return entries, False


def _persisted_config_from_entry(entry):
    config = entry.get("config")
    if not isinstance(config, dict):
        return None

    kwargs = config.get("kwargs")
    if not isinstance(kwargs, dict):
        return None

    name = config.get("name")
    if name is not None and not isinstance(name, str):
        return None

    return Config(kwargs=_normalize_persisted_value(kwargs), name=name, pre_hook=None)


def _config_from_spec(config, spec):
    for current in spec.configs:
        if config.kwargs == current.kwargs and config.name == current.name:
            return current
    return None


def _is_json_serializable(value):
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _load_persisted_best_config(kernel, spec, runtime_key_values, tuning_key):
    if not _persistent_cache_enabled(spec):
        return None, False

    kernel_id = _stable_kernel_identifier(kernel)
    if kernel_id is None:
        return None, False

    tuning_values = tuple(runtime_key_values[name] for name in spec.key)
    entries, had_read_error = _load_persistent_entries(spec.cache_path)
    for entry in entries:
        if not _persisted_entry_matches(entry, kernel_id, tuning_values):
            continue
        config = _persisted_config_from_entry(entry)
        if config is None:
            return None, had_read_error
        config = _config_from_spec(config, spec)
        if config is None:
            return None, had_read_error
        candidate_kwargs = dict(config.kwargs)
        cached_best = (config, candidate_kwargs)
        _BEST_CONFIG_CACHE[tuning_key] = cached_best
        return cached_best, had_read_error
    return None, had_read_error


def _persist_best_config(kernel, spec, runtime_key_values, best_config):
    if not _persistent_cache_enabled(spec):
        return

    kernel_id = _stable_kernel_identifier(kernel)
    if kernel_id is None:
        return

    persisted_config = {"kwargs": dict(best_config.kwargs), "name": best_config.name}
    entry = {
        "kernel": kernel_id,
        "key": [runtime_key_values[name] for name in spec.key],
        "config": persisted_config,
    }
    if not _is_json_serializable(entry):
        return

    persisted_entries, had_read_error = _load_persistent_entries(spec.cache_path)
    if had_read_error:
        persisted_entries = []

    entries = [
        existing
        for existing in persisted_entries
        if not _persisted_entry_matches(
            existing, kernel_id, tuple(runtime_key_values[name] for name in spec.key)
        )
    ]
    entries.append(entry)

    try:
        spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
        spec.cache_path.write_text(json.dumps({"entries": entries}, indent=2))
    except (OSError, TypeError, ValueError):
        return


def compile(kernel, *args, **kwargs):
    verbose = kwargs.pop("verbose", False)
    spec = read_autotune_spec(kernel)
    if spec is not None and (
        inspect.isfunction(kernel) or autotune_spec_applies_to_call(kernel, spec)
    ):
        runtime_key_values = _resolve_key_values(kernel, spec, *args, **kwargs)
        do_bench = spec.do_bench or benchmark
        tuning_key = _tuning_key(kernel, spec, runtime_key_values)

        use_cache = spec.cache_results and not spec.force_retune
        cached_best = _BEST_CONFIG_CACHE.get(tuning_key) if use_cache else None
        loaded_from_disk = False
        if cached_best is None and use_cache:
            cached_best, _ = _load_persisted_best_config(
                kernel, spec, runtime_key_values, tuning_key
            )
            loaded_from_disk = cached_best is not None
        if cached_best is not None:
            best_config, best_candidate_kwargs = cached_best
            cached_compiled, cache_key = _get_cached_compiled_candidate(
                kernel, best_candidate_kwargs, best_config, args, kwargs
            )
            if cached_compiled is not None:
                if loaded_from_disk:
                    _log_verbose(
                        verbose,
                        "disk-cache-hit: loaded best config from persistent cache",
                    )
                else:
                    _log_verbose(verbose, "cache-hit")
                return cached_compiled

            candidate_kernel, compile_kwargs = _compile_target_and_kwargs(
                kernel, best_candidate_kwargs, kwargs
            )
            if best_config.pre_hook is not None:
                best_config.pre_hook(
                    candidate_kernel,
                    *args,
                    **_pre_hook_kwargs(kernel, best_candidate_kwargs, kwargs),
                )
            if loaded_from_disk:
                _log_verbose(
                    verbose,
                    "disk-cache-hit: loaded best config from persistent cache",
                )
            return _compile_candidate(candidate_kernel, cache_key, args, compile_kwargs)

        best_compiled = None
        best_time = None
        best_config = None
        best_candidate_kwargs = None
        failures = []
        for config in spec.configs:
            candidate_kwargs = dict(config.kwargs)
            cached_compiled, cache_key = _get_cached_compiled_candidate(
                kernel, candidate_kwargs, config, args, kwargs
            )
            if cached_compiled is not None:
                compiled = cached_compiled
            else:
                try:
                    candidate_kernel, compile_kwargs = _compile_target_and_kwargs(
                        kernel, candidate_kwargs, kwargs
                    )
                except Exception as exc:
                    failures.append((_config_label(config), exc))
                    continue

                if config.pre_hook is not None:
                    config.pre_hook(
                        candidate_kernel,
                        *args,
                        **_pre_hook_kwargs(kernel, candidate_kwargs, kwargs),
                    )

                try:
                    compiled = _compile_candidate(
                        candidate_kernel, cache_key, args, compile_kwargs
                    )
                except Exception as exc:
                    failures.append((_config_label(config), exc))
                    continue
            try:
                bench_args, bench_kwargs = _benchmark_args_and_kwargs(
                    kernel, compiled, args, kwargs
                )
                timed = do_bench(
                    compiled,
                    *bench_args,
                    warmup=spec.warmup,
                    rep=spec.rep,
                    **bench_kwargs,
                )
                if verbose:
                    print(f"[cutez.autotune] config={config.kwargs} time={timed}")
            except Exception as exc:
                failures.append((_config_label(config), exc))
                continue
            if best_time is None or timed < best_time:
                best_time = timed
                best_compiled = compiled
                best_config = config
                best_candidate_kwargs = candidate_kwargs

        if best_config is None:
            raise AutotuneError(_format_candidate_failures(failures))

        if verbose:
            print(
                f"[cutez.autotune] best_config={best_config.kwargs} best_time={best_time}"
            )

        if spec.cache_results and best_config is not None:
            _BEST_CONFIG_CACHE[tuning_key] = (best_config, best_candidate_kwargs)
            _persist_best_config(kernel, spec, runtime_key_values, best_config)

        return best_compiled
    return cute.compile(kernel, *args, **kwargs)
