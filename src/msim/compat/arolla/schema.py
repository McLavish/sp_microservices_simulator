"""Arolla's experiment schema, read strictly.

The shape of an Arolla scenario file (`simulator/config/schema.py` in their
repository), as plain dataclasses, validated once when the file is read. It
accepts every scenario their own schema accepts and means, and is stricter where
theirs lets a mistake through:

* a key the schema does not know is an ERROR naming its path (theirs ignores it,
  so a misspelt `querue_capacity` silently means "unbounded");
* ranges are checked (`p_fail` in [0, 1], a positive median, `end_s > start_s`);
* a service with more than one limiter is an error (theirs keeps one by a fixed
  priority and drops the rest);
* a key their runner accepts and then ignores is read and reported as a WARNING,
  so a scenario that relies on it is not silently different.

A scenario is validated AFTER its sweep point has been applied (`sweeps.py`): a
list where a scalar belongs is therefore an error here too.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class SchemaError(ValueError):
    """An Arolla scenario that does not say what it means."""


_REQUIRED: Any = object()


class _Reader:
    """One YAML mapping being read: typed access, and a list of what was not read."""

    def __init__(self, data: object, path: str, warnings: list[str]) -> None:
        if not isinstance(data, Mapping):
            raise SchemaError(f"{path}: expected a mapping, got {_show(data)}")
        self._data: Mapping[Any, Any] = data
        self._path = path
        self._seen: set[str] = set()
        self.warnings = warnings

    def path(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def has(self, key: str) -> bool:
        return self._data.get(key) is not None

    def _raw(self, key: str, default: Any) -> Any:
        self._seen.add(key)
        value = self._data.get(key)
        if value is None:
            if default is _REQUIRED:
                raise SchemaError(f"{self.path(key)}: required")
            return default
        return value

    def number(
        self,
        key: str,
        default: Any = _REQUIRED,
        *,
        gt: float | None = None,
        ge: float | None = None,
        lt: float | None = None,
        le: float | None = None,
    ) -> Any:
        value = self._raw(key, default)
        if value is default:
            return value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SchemaError(f"{self.path(key)}: expected a number, got {_show(value)}")
        if not math.isfinite(value):
            raise SchemaError(f"{self.path(key)}: must be finite, got {value!r}")
        for bound, ok, text in (
            (gt, gt is None or value > gt, "> "),
            (ge, ge is None or value >= ge, ">= "),
            (lt, lt is None or value < lt, "< "),
            (le, le is None or value <= le, "<= "),
        ):
            if not ok:
                raise SchemaError(f"{self.path(key)}: must be {text}{bound:g}, got {value!r}")
        return value

    def integer(self, key: str, default: Any = _REQUIRED, **bounds: float | None) -> Any:
        value = self.number(key, default, **bounds)
        if value is default:
            return value
        if isinstance(value, float):
            if not value.is_integer():
                raise SchemaError(f"{self.path(key)}: expected a whole number, got {value!r}")
            value = int(value)
        return value

    def string(
        self, key: str, default: Any = _REQUIRED, *, choices: tuple[str, ...] | None = None
    ) -> Any:
        value = self._raw(key, default)
        if value is default:
            return value
        if not isinstance(value, str):
            raise SchemaError(f"{self.path(key)}: expected a string, got {_show(value)}")
        if choices is not None and value not in choices:
            raise SchemaError(
                f"{self.path(key)}: must be one of {', '.join(choices)}; got {value!r}"
            )
        return value

    def boolean(self, key: str, default: Any = _REQUIRED) -> Any:
        value = self._raw(key, default)
        if value is default:
            return value
        if not isinstance(value, bool):
            raise SchemaError(f"{self.path(key)}: expected true or false, got {_show(value)}")
        return value

    def mapping(self, key: str) -> "_Reader | None":
        value = self._raw(key, None)
        if value is None:
            return None
        return _Reader(value, self.path(key), self.warnings)

    def mappings(self, key: str) -> "list[_Reader]":
        value = self._raw(key, None)
        if value is None:
            return []
        if not isinstance(value, list):
            raise SchemaError(f"{self.path(key)}: expected a list, got {_show(value)}")
        return [
            _Reader(item, f"{self.path(key)}[{index}]", self.warnings)
            for index, item in enumerate(value)
        ]

    def ignored(self, *keys: str) -> None:
        """Keys that are read and mean nothing here (their runner's output settings)."""
        self._seen.update(keys)

    def done(self, known: tuple[str, ...]) -> None:
        self._seen.update(key for key in known if key not in self._data)
        unknown = sorted(str(key) for key in self._data if key not in self._seen)
        if unknown:
            raise SchemaError(
                f"{self.path(unknown[0])}: unknown key (the keys here are: {', '.join(known)})"
            )


def _show(value: object) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


# --------------------------------------------------------------------- policies


@dataclass(frozen=True)
class Latency:
    median_ms: float
    lognorm_sigma: float = 0.5


@dataclass(frozen=True)
class Retry:
    type: str  # none | fixed | exponential | jittered
    max_attempts: int = 3
    delay_ms: float | None = None
    initial_delay_ms: float | None = None
    max_delay_ms: float | None = None
    jitter_mode: str | None = None  # full | equal


@dataclass(frozen=True)
class Timeout:
    global_ms: float | None = None
    attempt_ms: float | None = None


@dataclass(frozen=True)
class CircuitBreaker:
    type: str  # count_based | time_based | retry_circuit_breaker
    failure_threshold: float
    half_open_delay_ms: float
    success_threshold: float | None = None
    failure_window_size: int | None = None
    success_window_size: int | None = None
    window_duration_ms: float | None = None
    min_requests: int | None = None


@dataclass(frozen=True)
class RateLimiter:
    type: str  # leaky | bursty | fixed_window
    max_requests: int
    period_ms: float
    max_attempts: int | None = None
    refill_rate: int | None = None


@dataclass(frozen=True)
class RetryBudget:
    budget_ratio: float
    max_retries: int
    min_retries_per_sec: float = 10
    shared_budget_id: str | None = None


@dataclass(frozen=True)
class GlobalRetryBudget:
    target_rps: float
    max_burst: int = 10


@dataclass(frozen=True)
class AimdRetryBudget:
    min_rps: float
    max_rps: float
    initial_rps: float
    max_burst: int = 10
    additive_step: float = 1
    decrease_factor: float = 0.5
    window_ms: float = 1000.0
    failure_threshold: float = 0.1


@dataclass(frozen=True)
class ArollaRetryBudget:
    alpha: float = 0.1
    beta_down: float = 0.3
    beta_up: float = 0.05
    window_ms: float = 1000.0
    success_rate_threshold: float | None = None
    success_rate_beta: float = 0.1
    max_retry_ratio: float | None = None


# ----------------------------------------------------------------------- faults


@dataclass(frozen=True)
class LatencyInjection:
    start_s: float
    end_s: float
    add_latency_ms: float = 0
    multiplier: int = 1


@dataclass(frozen=True)
class PartialFailure:
    start_s: float
    end_s: float
    p_fail: float = 0.0


@dataclass(frozen=True)
class LoadSpike:
    start_s: float
    end_s: float
    rps_multiplier: float


# -------------------------------------------------------------------- structure


@dataclass(frozen=True)
class Dependency:
    service: str
    optional: bool = False


#: A service's limiter keys. At most one may be set.
LIMITER_KEYS = (
    "circuit_breaker",
    "rate_limiter",
    "retry_budget",
    "global_retry_budget",
    "aimd_global_retry_budget",
    "arolla_retry_budget",
)


@dataclass(frozen=True)
class Service:
    name: str
    latency: Latency
    workers: int
    queue_capacity: int | None = None
    retry: Retry | None = None
    timeout: Timeout | None = None
    circuit_breaker: CircuitBreaker | None = None
    rate_limiter: RateLimiter | None = None
    retry_budget: RetryBudget | None = None
    global_retry_budget: GlobalRetryBudget | None = None
    aimd_global_retry_budget: AimdRetryBudget | None = None
    arolla_retry_budget: ArollaRetryBudget | None = None
    latency_injections: tuple[LatencyInjection, ...] = ()
    partial_failures: tuple[PartialFailure, ...] = ()
    dependencies: tuple[Dependency, ...] = ()
    dependency_call_pattern: str = "sequential"  # sequential | parallel

    @property
    def limiter(self) -> tuple[str, Any] | None:
        """`(key, config)` of the service's limiter, if it has one."""
        for key in LIMITER_KEYS:
            value = getattr(self, key)
            if value is not None:
                return key, value
        return None


@dataclass(frozen=True)
class Workload:
    base_rps: float
    duration_s: int
    load_spikes: tuple[LoadSpike, ...] = ()
    rng_seed: int | None = None


@dataclass(frozen=True)
class Client:
    name: str
    workload: Workload
    target_service: str | None = None
    retry: Retry | None = None
    timeout: Timeout | None = None
    circuit_breaker: CircuitBreaker | None = None
    retry_budget: RetryBudget | None = None
    e2e_retry_budget: int | None = None
    replicas: int = 1


@dataclass(frozen=True)
class Experiment:
    name: str
    services: tuple[Service, ...]
    seed: int = 42
    #: Single-client mode: one implicit client, named "client", on the first service.
    workload: Workload | None = None
    clients: tuple[Client, ...] = ()
    #: What was read and will not be acted on, in plain words.
    warnings: tuple[str, ...] = field(default=(), compare=False)

    def service(self, name: str) -> Service:
        for service in self.services:
            if service.name == name:
                return service
        raise KeyError(name)

    @property
    def entry(self) -> Service:
        """The service a client without `target_service` calls: the first one."""
        return self.services[0]


# ---------------------------------------------------------------------- readers

_RETRY_KEYS = (
    "type",
    "max_attempts",
    "delay_ms",
    "initial_delay_ms",
    "max_delay_ms",
    "jitter_mode",
)


def _retry(reader: _Reader | None) -> Retry | None:
    if reader is None:
        return None
    kind = reader.string("type", choices=("none", "fixed", "exponential", "jittered"))
    retry = Retry(
        type=kind,
        max_attempts=reader.integer("max_attempts", 3, ge=1),
        delay_ms=reader.number("delay_ms", None, ge=0),
        initial_delay_ms=reader.number("initial_delay_ms", None, ge=0),
        max_delay_ms=reader.number("max_delay_ms", None, ge=0),
        jitter_mode=reader.string("jitter_mode", None, choices=("full", "equal", "decorrelated")),
    )
    reader.done(_RETRY_KEYS)
    if kind == "fixed" and retry.delay_ms is None:
        raise SchemaError(f"{reader.path('delay_ms')}: required for a fixed backoff")
    if kind in ("exponential", "jittered"):
        for key in ("initial_delay_ms", "max_delay_ms"):
            if getattr(retry, key) is None:
                raise SchemaError(f"{reader.path(key)}: required for {kind} backoff")
    if retry.jitter_mode == "decorrelated":
        raise SchemaError(
            f"{reader.path('jitter_mode')}: decorrelated jitter is accepted by Arolla's schema "
            "and implemented by neither simulator"
        )
    return retry


def _timeout(reader: _Reader | None) -> Timeout | None:
    if reader is None:
        return None
    timeout = Timeout(
        global_ms=reader.number("global_ms", None, ge=0),
        attempt_ms=reader.number("attempt_ms", None, ge=0),
    )
    reader.done(("global_ms", "attempt_ms"))
    return timeout


_BREAKER_KEYS = (
    "type",
    "failure_threshold",
    "success_threshold",
    "half_open_delay_ms",
    "failure_window_size",
    "success_window_size",
    "window_duration_ms",
    "min_requests",
)


def _breaker(reader: _Reader | None, *, on_client: bool) -> CircuitBreaker | None:
    if reader is None:
        return None
    kind = reader.string("type", choices=("count_based", "time_based", "retry_circuit_breaker"))
    breaker = CircuitBreaker(
        type=kind,
        failure_threshold=reader.number("failure_threshold", ge=0, le=1),
        success_threshold=reader.number("success_threshold", None, ge=0, le=1),
        half_open_delay_ms=reader.number("half_open_delay_ms", ge=0),
        failure_window_size=reader.integer("failure_window_size", None, ge=1),
        success_window_size=reader.integer("success_window_size", None, ge=1),
        window_duration_ms=reader.number("window_duration_ms", None, gt=0),
        min_requests=reader.integer("min_requests", None, ge=1),
    )
    reader.done(_BREAKER_KEYS)
    if kind == "count_based":
        if on_client:
            reader.warnings.append(
                f"{reader.path('type')}: a count_based circuit breaker on a client is accepted "
                "and ignored by Arolla's runner; it is ignored here too"
            )
        for key in ("success_threshold", "failure_window_size", "success_window_size"):
            if getattr(breaker, key) is None and not on_client:
                raise SchemaError(f"{reader.path(key)}: required for a count_based breaker")
    elif not on_client:
        # Their runner hands these to the breaker as they are: a missing one is None.
        for key in ("window_duration_ms", "min_requests"):
            if getattr(breaker, key) is None and kind == "time_based":
                raise SchemaError(f"{reader.path(key)}: required for a time_based breaker")
    return breaker


def _rate_limiter(reader: _Reader | None) -> RateLimiter | None:
    if reader is None:
        return None
    limiter = RateLimiter(
        type=reader.string("type", choices=("leaky", "bursty", "fixed_window")),
        max_requests=reader.integer("max_requests", ge=1),
        period_ms=reader.number("period_ms", gt=0),
        max_attempts=reader.integer("max_attempts", None, ge=1),
        refill_rate=reader.integer("refill_rate", None, ge=1),
    )
    reader.done(("type", "max_requests", "period_ms", "max_attempts", "refill_rate"))
    return limiter


def _retry_budget(reader: _Reader | None, *, on_client: bool) -> RetryBudget | None:
    if reader is None:
        return None
    budget = RetryBudget(
        budget_ratio=reader.number("budget_ratio", ge=0, le=1),
        min_retries_per_sec=reader.number("min_retries_per_sec", 10, ge=0),
        max_retries=reader.integer("max_retries", ge=1),
        shared_budget_id=reader.string("shared_budget_id", None),
    )
    reader.done(("budget_ratio", "min_retries_per_sec", "max_retries", "shared_budget_id"))
    if not on_client:
        for key in ("min_retries_per_sec", "shared_budget_id"):
            if reader.has(key):
                reader.warnings.append(
                    f"{reader.path(key)}: only a client's retry budget uses it; on a service "
                    "Arolla's runner ignores it, and so does this reader"
                )
    return budget


def _global_budget(reader: _Reader | None) -> GlobalRetryBudget | None:
    if reader is None:
        return None
    budget = GlobalRetryBudget(
        target_rps=reader.number("target_rps", ge=1),
        max_burst=reader.integer("max_burst", 10, ge=1),
    )
    reader.done(("target_rps", "max_burst"))
    return budget


def _aimd_budget(reader: _Reader | None) -> AimdRetryBudget | None:
    if reader is None:
        return None
    budget = AimdRetryBudget(
        min_rps=reader.number("min_rps", ge=1),
        max_rps=reader.number("max_rps", ge=1),
        initial_rps=reader.number("initial_rps", ge=1),
        max_burst=reader.integer("max_burst", 10, ge=1),
        additive_step=reader.number("additive_step", 1, ge=1),
        decrease_factor=reader.number("decrease_factor", 0.5, gt=0, lt=1),
        window_ms=reader.number("window_ms", 1000.0, gt=0),
        failure_threshold=reader.number("failure_threshold", 0.1, ge=0, le=1),
    )
    reader.done(
        (
            "min_rps",
            "max_rps",
            "initial_rps",
            "max_burst",
            "additive_step",
            "decrease_factor",
            "window_ms",
            "failure_threshold",
        )
    )
    if not budget.min_rps <= budget.initial_rps <= budget.max_rps:
        raise SchemaError(
            f"{reader.path('initial_rps')}: must lie within [min_rps, max_rps] = "
            f"[{budget.min_rps:g}, {budget.max_rps:g}], got {budget.initial_rps:g}"
        )
    return budget


def _arolla_budget(reader: _Reader | None) -> ArollaRetryBudget | None:
    if reader is None:
        return None
    budget = ArollaRetryBudget(
        alpha=reader.number("alpha", 0.1, ge=0, le=1),
        beta_down=reader.number("beta_down", 0.3, gt=0, le=1),
        beta_up=reader.number("beta_up", 0.05, gt=0, le=1),
        window_ms=reader.number("window_ms", 1000.0, gt=0),
        success_rate_threshold=reader.number("success_rate_threshold", None, ge=0, le=1),
        success_rate_beta=reader.number("success_rate_beta", 0.1, gt=0, le=1),
        max_retry_ratio=reader.number("max_retry_ratio", None, gt=0, le=1),
    )
    reader.done(
        (
            "alpha",
            "beta_down",
            "beta_up",
            "window_ms",
            "success_rate_threshold",
            "success_rate_beta",
            "max_retry_ratio",
        )
    )
    return budget


def _window(reader: _Reader) -> tuple[float, float]:
    start = reader.number("start_s", ge=0)
    end = reader.number("end_s", ge=0)
    if end <= start:
        raise SchemaError(f"{reader.path('end_s')}: must be after start_s ({start:g}), got {end:g}")
    return start, end


def _latency_injection(reader: _Reader) -> LatencyInjection:
    reader.string("type", "latency_injection", choices=("latency_injection",))
    start, end = _window(reader)
    injection = LatencyInjection(
        start_s=start,
        end_s=end,
        add_latency_ms=reader.number("add_latency_ms", 0, ge=0),
        multiplier=reader.integer("multiplier", 1, ge=1),
    )
    reader.done(("type", "start_s", "end_s", "add_latency_ms", "multiplier"))
    return injection


def _partial_failure(reader: _Reader) -> PartialFailure:
    reader.string("type", "partial_failure", choices=("partial_failure",))
    start, end = _window(reader)
    failure = PartialFailure(
        start_s=start, end_s=end, p_fail=reader.number("p_fail", 0.0, ge=0, le=1)
    )
    reader.done(("type", "start_s", "end_s", "p_fail"))
    return failure


def _load_spike(reader: _Reader) -> LoadSpike:
    reader.string("type", "load_spike", choices=("load_spike",))
    start, end = _window(reader)
    spike = LoadSpike(
        start_s=start, end_s=end, rps_multiplier=reader.number("rps_multiplier", gt=0)
    )
    reader.done(("type", "start_s", "end_s", "rps_multiplier"))
    return spike


_SERVICE_KEYS = (
    "name",
    "latency",
    "workers",
    "queue_capacity",
    "retry",
    "timeout",
    *LIMITER_KEYS,
    "latency_injections",
    "partial_failures",
    "dependencies",
    "dependency_call_pattern",
    "dependency",
)


def _service(reader: _Reader) -> Service:
    name = reader.string("name")
    latency_reader = reader.mapping("latency")
    if latency_reader is None:
        raise SchemaError(f"{reader.path('latency')}: required")
    latency = Latency(
        median_ms=latency_reader.number("median_ms", gt=0),
        lognorm_sigma=latency_reader.number("lognorm_sigma", 0.5, ge=0),
    )
    latency_reader.done(("median_ms", "lognorm_sigma"))

    dependencies = []
    for dependency in reader.mappings("dependencies"):
        dependencies.append(
            Dependency(
                service=dependency.string("service"),
                optional=dependency.boolean("optional", False),
            )
        )
        dependency.done(("service", "optional"))
    legacy = reader.string("dependency", None)
    if legacy is not None:
        if dependencies:
            reader.warnings.append(
                f"{reader.path('dependency')}: ignored because `dependencies` is set too, "
                "as in Arolla's runner"
            )
        else:
            dependencies.append(Dependency(service=legacy))

    service = Service(
        name=name,
        latency=latency,
        workers=reader.integer("workers", ge=1),
        queue_capacity=reader.integer("queue_capacity", None, ge=0),
        retry=_retry(reader.mapping("retry")),
        timeout=_timeout(reader.mapping("timeout")),
        circuit_breaker=_breaker(reader.mapping("circuit_breaker"), on_client=False),
        rate_limiter=_rate_limiter(reader.mapping("rate_limiter")),
        retry_budget=_retry_budget(reader.mapping("retry_budget"), on_client=False),
        global_retry_budget=_global_budget(reader.mapping("global_retry_budget")),
        aimd_global_retry_budget=_aimd_budget(reader.mapping("aimd_global_retry_budget")),
        arolla_retry_budget=_arolla_budget(reader.mapping("arolla_retry_budget")),
        latency_injections=tuple(map(_latency_injection, reader.mappings("latency_injections"))),
        partial_failures=tuple(map(_partial_failure, reader.mappings("partial_failures"))),
        dependencies=tuple(dependencies),
        dependency_call_pattern=reader.string(
            "dependency_call_pattern", "sequential", choices=("sequential", "parallel")
        ),
    )
    reader.done(_SERVICE_KEYS)
    limiters = [key for key in LIMITER_KEYS if getattr(service, key) is not None]
    if len(limiters) > 1:
        raise SchemaError(
            f"{reader.path(limiters[1])}: service {name!r} has more than one limiter "
            f"({', '.join(limiters)}); Arolla's runner would keep one and silently drop the rest"
        )
    return service


def _workload(reader: _Reader | None, path: str) -> Workload:
    if reader is None:
        raise SchemaError(f"{path}: required")
    workload = Workload(
        base_rps=reader.number("base_rps", gt=0),
        duration_s=reader.integer("duration_s", gt=0),
        load_spikes=tuple(map(_load_spike, reader.mappings("load_spikes"))),
        rng_seed=reader.integer("rng_seed", None),
    )
    reader.done(("base_rps", "duration_s", "load_spikes", "rng_seed"))
    return workload


_CLIENT_KEYS = (
    "name",
    "workload",
    "target_service",
    "retry",
    "timeout",
    "circuit_breaker",
    "retry_budget",
    "e2e_retry_budget",
    "replicas",
)


def _client(reader: _Reader) -> Client:
    client = Client(
        name=reader.string("name"),
        workload=_workload(reader.mapping("workload"), reader.path("workload")),
        target_service=reader.string("target_service", None),
        retry=_retry(reader.mapping("retry")),
        timeout=_timeout(reader.mapping("timeout")),
        circuit_breaker=_breaker(reader.mapping("circuit_breaker"), on_client=True),
        retry_budget=_retry_budget(reader.mapping("retry_budget"), on_client=True),
        e2e_retry_budget=reader.integer("e2e_retry_budget", None, ge=0),
        replicas=reader.integer("replicas", 1, ge=1),
    )
    reader.done(_CLIENT_KEYS)
    if client.retry is None:
        # Without a policy of its own the client only relays the service's
        # attempts, and their builder never looks at these two.
        for key in ("circuit_breaker", "retry_budget"):
            if getattr(client, key) is not None:
                reader.warnings.append(
                    f"{reader.path(key)}: a client without a `retry` policy has no retries of "
                    "its own to gate; Arolla's runner ignores this key, and so does this reader"
                )
    return client


_EXPERIMENT_KEYS = (
    "name",
    "seed",
    "services",
    "workload",
    "clients",
    "output_csv",
    "fault_events_json",
    "granularity_s",
    "plotting_script",
    "plotting_script_args",
    "sweeps",
)


def parse(data: object) -> Experiment:
    """Read one concrete scenario (sweep lists already replaced by their values)."""
    warnings: list[str] = []
    reader = _Reader(data, "", warnings)
    services = tuple(map(_service, reader.mappings("services")))
    if not services:
        raise SchemaError("services: at least one service is required")
    workload_reader = reader.mapping("workload")
    clients = tuple(map(_client, reader.mappings("clients")))
    # Their runner's output settings: they say where files go, not what happens.
    reader.ignored(
        "output_csv",
        "fault_events_json",
        "granularity_s",
        "plotting_script",
        "plotting_script_args",
    )
    if reader.has("sweeps"):
        reader.ignored("sweeps")
        warnings.append(
            "sweeps: Arolla's runner reads this block and does not use it (a sweep is an "
            "inline list of values); it is ignored here too"
        )
    experiment_name = reader.string("name")
    seed = reader.integer("seed", 42)
    reader.done(_EXPERIMENT_KEYS)

    if workload_reader is None and not clients:
        raise SchemaError("one of `workload` (a single client) or `clients` is required")
    if workload_reader is not None and clients:
        raise SchemaError("`workload` and `clients` are both set: choose one")
    workload = None if workload_reader is None else _workload(workload_reader, "workload")

    names = [service.name for service in services]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SchemaError(f"services: the name {duplicates[0]!r} is used more than once")
    for service in services:
        for dependency in service.dependencies:
            if dependency.service not in names:
                raise SchemaError(
                    f"service {service.name!r} depends on {dependency.service!r}, "
                    f"which is not a service (services: {', '.join(names)})"
                )
    _require_acyclic(services)
    client_names = [client.name for client in clients]
    duplicates = sorted({name for name in client_names if client_names.count(name) > 1})
    if duplicates:
        raise SchemaError(f"clients: the name {duplicates[0]!r} is used more than once")
    for client in clients:
        if client.target_service is not None and client.target_service not in names:
            raise SchemaError(
                f"client {client.name!r} targets {client.target_service!r}, "
                f"which is not a service (services: {', '.join(names)})"
            )

    experiment = Experiment(
        name=experiment_name, seed=seed, services=services, workload=workload, clients=clients
    )
    warnings.extend(_unused_global_timeouts(experiment))
    return Experiment(
        name=experiment_name,
        seed=seed,
        services=services,
        workload=workload,
        clients=clients,
        warnings=tuple(warnings),
    )


def _require_acyclic(services: tuple[Service, ...]) -> None:
    remaining = {service.name: {d.service for d in service.dependencies} for service in services}
    while remaining:
        leaves = [name for name, needs in remaining.items() if not needs & remaining.keys()]
        if not leaves:
            raise SchemaError(
                f"services: the dependencies form a cycle through {', '.join(sorted(remaining))}"
            )
        for name in leaves:
            del remaining[name]


def _unused_global_timeouts(experiment: Experiment) -> list[str]:
    """A service's `global_ms` is only read for a client without a timeout of its own."""
    read_by: set[str] = set()
    if experiment.workload is not None:
        read_by.add(experiment.entry.name)
    for client in experiment.clients:
        if client.timeout is None:
            read_by.add(client.target_service or experiment.entry.name)
    warnings = []
    for service in experiment.services:
        if (
            service.timeout is not None
            and service.timeout.global_ms is not None
            and service.name not in read_by
        ):
            warnings.append(
                f"service {service.name!r}: timeout.global_ms is read only for a client that has "
                "no timeout of its own and calls this service directly; here nothing reads it "
                "(Arolla's runner ignores it for dependency calls and for clients with a timeout)"
            )
    return warnings
