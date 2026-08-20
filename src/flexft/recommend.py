"""Advisory method selection for flexible-grid transforms."""

from __future__ import annotations

import math
import statistics
import threading
import time
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .core import (
    CenteredDirectFRDFT,
    CenteredFRDFT,
    _complex_dtype,
    _validate_positive_int,
)


_METHODS = ("direct", "bluestein")
_DEFAULT_MAX_DIRECT_BYTES = 512 * 1024**2
_RECOMMENDATION_CACHE: dict[tuple[Any, ...], "MethodRecommendation"] = {}
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class MethodBenchmark:
    """Assessment of one flexible-grid evaluation method.

    Times are seconds and are populated only in benchmark mode. ``score`` is
    measured in seconds in benchmark mode and in approximate work units in
    estimate mode. ``execution_time`` is the median time for one invocation,
    including the requested batch size.
    """

    setup_time: float | None
    compilation_time: float | None
    execution_time: float | None
    estimated_setup_work: float
    estimated_execution_work: float
    estimated_memory_bytes: int
    score: float
    score_unit: Literal["seconds", "work_units"]
    skipped_reason: str | None = None


@dataclass(frozen=True)
class MethodRecommendation:
    """Recommended flexible-grid method and the evidence behind it."""

    method: Literal["direct", "bluestein"]
    results: Mapping[str, MethodBenchmark]
    mode: Literal["estimate", "benchmark"]
    device: str
    dtype: str
    reason: str


def _validate_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer, got {value!r}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}.")
    return result


def _resolve_dtype(dtype: Any) -> np.dtype:
    requested = _complex_dtype() if dtype is None else dtype
    try:
        result = np.dtype(jax.dtypes.canonicalize_dtype(requested))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"dtype must be a real or complex floating dtype, got {dtype!r}."
        ) from exc

    if not (
        np.issubdtype(result, np.floating) or np.issubdtype(result, np.complexfloating)
    ):
        raise TypeError(
            f"dtype must be a real or complex floating dtype, got {dtype!r}."
        )
    return result


def _resolve_device(device: Any):
    if device is None:
        return jax.devices()[0]
    if isinstance(device, str):
        try:
            devices = jax.devices(device)
        except RuntimeError as exc:
            raise ValueError(
                f"No JAX devices are available for platform {device!r}."
            ) from exc
        if not devices:
            raise ValueError(f"No JAX devices are available for platform {device!r}.")
        return devices[0]
    if not all(hasattr(device, attribute) for attribute in ("platform", "id")):
        raise TypeError("device must be a JAX device, a platform name, or None.")
    return device


def _device_label(device: Any) -> str:
    kind = getattr(device, "device_kind", type(device).__name__)
    return f"{device.platform}:{device.id} ({kind})"


def _estimated_values(
    N: int, M: int, *, batch_size: int
) -> dict[str, tuple[float, float, int]]:
    """Return setup work, execution work, and persistent plan bytes."""
    length = N + M
    log_length = math.log2(max(length, 2))
    internal_itemsize = np.dtype(_complex_dtype()).itemsize

    direct_setup = 12.0 * N * M
    direct_execution = 8.0 * batch_size * N * M
    direct_memory = N * M * internal_itemsize

    bluestein_setup = 5.0 * length * log_length + 8.0 * length
    bluestein_execution = batch_size * (10.0 * length * log_length + 8.0 * length)
    # Centering chirps, FRDFT chirps, and the transformed convolution kernel.
    bluestein_memory = 3 * length * internal_itemsize

    return {
        "direct": (
            direct_setup,
            direct_execution,
            direct_memory,
        ),
        "bluestein": (
            bluestein_setup,
            bluestein_execution,
            bluestein_memory,
        ),
    }


def _score(
    setup: float, compilation: float, execution: float, expected_calls: int | None
):
    if expected_calls is None:
        return execution
    return setup + compilation + expected_calls * execution


def _synchronize_plan(plan: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(vars(plan)):
        block_until_ready = getattr(leaf, "block_until_ready", None)
        if block_until_ready is not None:
            block_until_ready()


def _time_call(function, argument) -> float:
    start = time.perf_counter()
    result = function(argument)
    result.block_until_ready()
    return time.perf_counter() - start


def _initialize_benchmark(*, dtype: np.dtype, device: Any, jit: bool) -> None:
    """Exclude one-time backend and compiler startup from candidate timings."""
    with jax.default_device(device):
        sample = jax.device_put(jnp.ones((), dtype=dtype), device)
        sample.block_until_ready()
        if jit:
            operation = jax.jit(lambda value: value + 1)
            operation(sample).block_until_ready()


def _benchmark_method(
    method: str,
    *,
    N: int,
    M: int,
    dtype: np.dtype,
    device: Any,
    batch_size: int,
    expected_calls: int | None,
    jit: bool,
    warmup: int,
    repeats: int,
    estimated_setup_work: float,
    estimated_execution_work: float,
    estimated_memory_bytes: int,
) -> MethodBenchmark:
    plan_type = CenteredDirectFRDFT if method == "direct" else CenteredFRDFT
    alpha = math.sqrt(2.0) / (N + M)

    try:
        with jax.default_device(device):
            start = time.perf_counter()
            plan = plan_type(N, alpha, M=M)
            _synchronize_plan(plan)
            setup_time = time.perf_counter() - start

            shape = (N,) if batch_size == 1 else (batch_size, N)
            samples = jax.device_put(jnp.ones(shape, dtype=dtype), device)
            operation = plan if batch_size == 1 else jax.vmap(plan)

            compilation_time = 0.0
            if jit:
                operation = jax.jit(operation)
                first_call_time = _time_call(operation, samples)
            else:
                first_call_time = 0.0

            for _ in range(warmup):
                _time_call(operation, samples)

            execution_samples = [_time_call(operation, samples) for _ in range(repeats)]
            execution_time = statistics.median(execution_samples)
            if jit:
                # The first invocation includes one execution as well as compilation.
                compilation_time = max(0.0, first_call_time - execution_time)
    except (MemoryError, RuntimeError) as exc:
        return MethodBenchmark(
            setup_time=None,
            compilation_time=None,
            execution_time=None,
            estimated_setup_work=estimated_setup_work,
            estimated_execution_work=estimated_execution_work,
            estimated_memory_bytes=estimated_memory_bytes,
            score=math.inf,
            score_unit="seconds",
            skipped_reason=f"{type(exc).__name__}: {exc}",
        )

    return MethodBenchmark(
        setup_time=setup_time,
        compilation_time=compilation_time,
        execution_time=execution_time,
        estimated_setup_work=estimated_setup_work,
        estimated_execution_work=estimated_execution_work,
        estimated_memory_bytes=estimated_memory_bytes,
        score=_score(setup_time, compilation_time, execution_time, expected_calls),
        score_unit="seconds",
    )


def _select(results: Mapping[str, MethodBenchmark]) -> str:
    available = [
        method for method in _METHODS if results[method].skipped_reason is None
    ]
    if not available:
        reasons = "; ".join(
            f"{method}: {results[method].skipped_reason}" for method in _METHODS
        )
        raise RuntimeError(f"No FlexFT method could be assessed ({reasons}).")
    return min(available, key=lambda method: results[method].score)


def recommend_method(
    N,
    M=None,
    *,
    mode="estimate",
    dtype=None,
    batch_size=1,
    expected_calls=None,
    jit=True,
    device=None,
    warmup=2,
    repeats=10,
    max_direct_bytes=_DEFAULT_MAX_DIRECT_BYTES,
    cache=True,
) -> MethodRecommendation:
    """Recommend ``"direct"`` or ``"bluestein"`` for an ``N``-to-``M`` sum.

    ``mode="estimate"`` uses transparent work and memory estimates without
    executing either transform. ``mode="benchmark"`` times both candidates on
    the selected JAX device, synchronizing every call and reporting setup,
    compilation, and median execution times separately.

    The recommendation does not consider ordinary FFT evaluation because an
    FFT fixes the output grid and is therefore a construction choice rather
    than merely an implementation choice. Use ``FlexFT.fft`` for that grid.

    Estimate mode models direct execution as ``8 * batch_size * N * M`` work
    units and Bluestein execution as
    ``batch_size * (10 * L * log2(L) + 8 * L)``, where ``L=N+M``. Setup uses
    ``12 * N * M`` and ``5 * L * log2(L) + 8 * L``, respectively. These are
    deliberately approximate arithmetic models, not predicted wall times.

    Parameters
    ----------
    N, M
        Input and output lengths. ``M`` defaults to ``N``.
    mode
        ``"estimate"`` or ``"benchmark"``.
    dtype
        Representative input dtype. Defaults to the active JAX complex dtype.
    batch_size
        Number of vectors evaluated together in each invocation.
    expected_calls
        Expected number of plan invocations. If omitted, compare steady-state
        execution only. Otherwise include setup and compilation in the score.
    jit
        JIT-compile candidates in benchmark mode.
    device
        JAX device, platform name such as ``"cpu"``, or ``None`` for the
        default device.
    warmup, repeats
        Untimed warm-up calls and timed repetitions in benchmark mode.
    max_direct_bytes
        Skip direct evaluation when its estimated persistent kernel exceeds
        this many bytes. Defaults to 512 MiB; use ``None`` to disable the guard.
    cache
        Reuse an in-process recommendation for the same configuration.
    """
    N = _validate_positive_int(N)
    M = N if M is None else _validate_positive_int(M, name="M")
    if mode not in ("estimate", "benchmark"):
        raise ValueError("mode must be 'estimate' or 'benchmark'.")

    dtype = _resolve_dtype(dtype)
    batch_size = _validate_positive_int(batch_size, name="batch_size")
    warmup = _validate_nonnegative_int(warmup, name="warmup")
    repeats = _validate_positive_int(repeats, name="repeats")
    if expected_calls is not None:
        expected_calls = _validate_positive_int(expected_calls, name="expected_calls")
    if max_direct_bytes is not None:
        max_direct_bytes = _validate_nonnegative_int(
            max_direct_bytes, name="max_direct_bytes"
        )
    if not isinstance(jit, bool):
        raise TypeError(f"jit must be a boolean, got {jit!r}.")
    if not isinstance(cache, bool):
        raise TypeError(f"cache must be a boolean, got {cache!r}.")

    resolved_device = _resolve_device(device) if mode == "benchmark" else None
    device_label = (
        _device_label(resolved_device)
        if resolved_device is not None
        else "not benchmarked"
    )
    cache_key = (
        N,
        M,
        mode,
        dtype.str,
        batch_size,
        expected_calls,
        jit,
        device_label,
        warmup,
        repeats,
        max_direct_bytes,
        jax.__version__,
        jax.config.x64_enabled,
        str(jax.config.jax_default_matmul_precision),
    )
    if cache:
        with _CACHE_LOCK:
            cached = _RECOMMENDATION_CACHE.get(cache_key)
        if cached is not None:
            return cached

    estimates = _estimated_values(N, M, batch_size=batch_size)
    if mode == "benchmark":
        _initialize_benchmark(dtype=dtype, device=resolved_device, jit=jit)
    results: dict[str, MethodBenchmark] = {}
    for method in _METHODS:
        setup_work, execution_work, estimated_memory = estimates[method]
        if (
            method == "direct"
            and max_direct_bytes is not None
            and N * M * np.dtype(_complex_dtype()).itemsize > max_direct_bytes
        ):
            results[method] = MethodBenchmark(
                setup_time=None,
                compilation_time=None,
                execution_time=None,
                estimated_setup_work=setup_work,
                estimated_execution_work=execution_work,
                estimated_memory_bytes=estimated_memory,
                score=math.inf,
                score_unit="seconds" if mode == "benchmark" else "work_units",
                skipped_reason=(
                    "estimated direct kernel exceeds max_direct_bytes="
                    f"{max_direct_bytes}"
                ),
            )
            continue

        if mode == "estimate":
            score = (
                execution_work
                if expected_calls is None
                else setup_work + expected_calls * execution_work
            )
            results[method] = MethodBenchmark(
                setup_time=None,
                compilation_time=None,
                execution_time=None,
                estimated_setup_work=setup_work,
                estimated_execution_work=execution_work,
                estimated_memory_bytes=estimated_memory,
                score=score,
                score_unit="work_units",
            )
        else:
            results[method] = _benchmark_method(
                method,
                N=N,
                M=M,
                dtype=dtype,
                device=resolved_device,
                batch_size=batch_size,
                expected_calls=expected_calls,
                jit=jit,
                warmup=warmup,
                repeats=repeats,
                estimated_setup_work=setup_work,
                estimated_execution_work=execution_work,
                estimated_memory_bytes=estimated_memory,
            )

    selected = _select(results)
    selected_result = results[selected]
    basis = (
        "lowest measured score"
        if mode == "benchmark"
        else "lowest approximate work score"
    )
    reason = f"{selected!r} has the {basis} ({selected_result.score:.6g})."
    recommendation = MethodRecommendation(
        method=selected,
        results=MappingProxyType(results),
        mode=mode,
        device=device_label,
        dtype=dtype.name,
        reason=reason,
    )

    if cache:
        with _CACHE_LOCK:
            recommendation = _RECOMMENDATION_CACHE.setdefault(cache_key, recommendation)
    return recommendation
