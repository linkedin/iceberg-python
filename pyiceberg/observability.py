# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger("pyiceberg.perf")

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class PerfEvent:
    """A structured performance event emitted by instrumentation points.

    ``tags`` are low-cardinality dimensions for grouping/filtering (e.g. database,
    table, file_format).  ``metrics`` are measured values (e.g. row_count,
    batch_count, response_bytes).
    """

    operation: str
    duration_ms: float
    tags: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, int | float] = field(default_factory=dict)


@runtime_checkable
class PerfObserver(Protocol):
    """Protocol for receiving performance events."""

    def emit(self, event: PerfEvent) -> None: ...


class NullPerfObserver:
    """No-op observer that discards all events. Default — zero overhead."""

    def emit(self, event: PerfEvent) -> None:
        pass


class LoggingPerfObserver:
    """Observer that emits structured key=value log lines at DEBUG level."""

    def emit(self, event: PerfEvent) -> None:
        parts = [f"operation={event.operation}", f"duration_ms={event.duration_ms:.3f}"]
        for key, value in event.tags.items():
            parts.append(f"{key}={value}")
        for key, value in event.metrics.items():
            parts.append(f"{key}={value}")
        logger.debug(" ".join(parts))


class CompositeObserver:
    """Fans out events to multiple observers.

    If an observer raises, the exception is logged and remaining observers
    still receive the event.
    """

    def __init__(self, *observers: PerfObserver) -> None:
        self._observers = observers

    def emit(self, event: PerfEvent) -> None:
        for observer in self._observers:
            try:
                observer.emit(event)
            except Exception:
                logger.debug("Observer %s failed to emit event %s", type(observer).__name__, event.operation, exc_info=True)


_observer: PerfObserver = NullPerfObserver()
_observer_lock = threading.Lock()


def set_observer(observer: PerfObserver) -> None:
    """Set the global performance observer. Thread-safe."""
    global _observer
    with _observer_lock:
        _observer = observer


def get_observer() -> PerfObserver:
    """Get the current global performance observer. Thread-safe."""
    with _observer_lock:
        return _observer


class _PerfTimerContext:
    """Context object yielded by perf_timer.

    Use ``.tag()`` for dimensions (low-cardinality strings) and
    ``.metric()`` for measured values (counts, sizes, etc.).
    """

    __slots__ = ("_tags", "_metrics")

    def __init__(self, initial_tags: dict[str, str]) -> None:
        self._tags = initial_tags
        self._metrics: dict[str, int | float] = {}

    def tag(self, key: str, value: str) -> None:
        """Set a dimension tag on the performance event."""
        self._tags[key] = value

    def metric(self, key: str, value: int | float) -> None:
        """Set a metric value on the performance event."""
        self._metrics[key] = value


@contextmanager
def perf_timer(operation: str, **tags: str) -> Generator[_PerfTimerContext, None, None]:
    """Context manager for timing a block of code and emitting a PerfEvent.

    Keyword arguments are recorded as dimension tags. Use ``ctx.metric()``
    inside the block for measured values.

    When the active observer is NullPerfObserver, time.monotonic() is skipped entirely.
    """
    with _observer_lock:
        observer = _observer
    if isinstance(observer, NullPerfObserver):
        yield _PerfTimerContext(tags)
        return

    ctx = _PerfTimerContext(tags)
    start = time.monotonic()
    try:
        yield ctx
    finally:
        duration_ms = (time.monotonic() - start) * 1000.0
        observer.emit(PerfEvent(operation=operation, duration_ms=duration_ms, tags=ctx._tags, metrics=ctx._metrics))


def timed(operation: str, **decorator_tags: str) -> Callable[[F], F]:
    """Decorator that wraps a function body in perf_timer.

    Built on top of perf_timer internally — same PerfObserver/PerfEvent pipeline,
    same NullPerfObserver fast path, same structured log output.

    The decorated function receives an extra ``_perf_ctx`` keyword argument
    (a :class:`_PerfTimerContext`) that can be used to set tags/metrics::

        @timed("my.operation")
        def process(data, *, _perf_ctx=None):
            ...
            if _perf_ctx:
                _perf_ctx.metric("row_count", len(data))

    Functions that don't declare ``_perf_ctx`` in their signature can ignore it
    — the wrapper only passes it if the function accepts ``**kwargs`` or has an
    explicit ``_perf_ctx`` parameter.
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with perf_timer(operation, **decorator_tags) as t:
                kwargs["_perf_ctx"] = t
                try:
                    result = fn(*args, **kwargs)
                except TypeError:
                    # Function doesn't accept _perf_ctx — retry without it
                    del kwargs["_perf_ctx"]
                    result = fn(*args, **kwargs)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
