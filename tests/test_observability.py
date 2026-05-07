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
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from pyiceberg.observability import (
    CompositeObserver,
    LoggingPerfObserver,
    NullPerfObserver,
    PerfEvent,
    PerfObserver,
    _PerfTimerContext,
    get_observer,
    perf_timer,
    set_observer,
    timed,
)


class CollectingObserver:
    """Test observer that collects all emitted events."""

    def __init__(self) -> None:
        self.events: list[PerfEvent] = []

    def emit(self, event: PerfEvent) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def _reset_observer() -> Iterator[None]:
    """Reset the global observer to NullPerfObserver after each test."""
    set_observer(NullPerfObserver())
    yield
    set_observer(NullPerfObserver())


class TestPerfEvent:
    def test_perf_event_creation(self) -> None:
        event = PerfEvent(operation="test.op", duration_ms=42.5, tags={"db": "prod"}, metrics={"rows": 100})
        assert event.operation == "test.op"
        assert event.duration_ms == 42.5
        assert event.tags == {"db": "prod"}
        assert event.metrics == {"rows": 100}

    def test_perf_event_default_tags_and_metrics(self) -> None:
        event = PerfEvent(operation="test.op", duration_ms=0.0)
        assert event.tags == {}
        assert event.metrics == {}

    def test_tags_and_metrics_are_separate(self) -> None:
        event = PerfEvent(operation="test.op", duration_ms=1.0, tags={"db": "prod"}, metrics={"rows": 42})
        assert "db" not in event.metrics
        assert "rows" not in event.tags


class TestNullPerfObserver:
    def test_emit_does_nothing(self) -> None:
        observer = NullPerfObserver()
        event = PerfEvent(operation="test.op", duration_ms=1.0)
        observer.emit(event)  # should not raise

    def test_satisfies_protocol(self) -> None:
        observer = NullPerfObserver()
        assert isinstance(observer, PerfObserver)


class TestLoggingPerfObserver:
    def test_satisfies_protocol(self) -> None:
        observer = LoggingPerfObserver()
        assert isinstance(observer, PerfObserver)

    def test_emits_structured_log_line(self, caplog: pytest.LogCaptureFixture) -> None:
        observer = LoggingPerfObserver()
        event = PerfEvent(operation="test.op", duration_ms=123.456, tags={"table": "db.tbl"}, metrics={"rows": 42})
        with caplog.at_level(logging.DEBUG, logger="pyiceberg.perf"):
            observer.emit(event)
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert "operation=test.op" in msg
        assert "duration_ms=123.456" in msg
        assert "table=db.tbl" in msg
        assert "rows=42" in msg

    def test_emits_at_debug_level(self, caplog: pytest.LogCaptureFixture) -> None:
        observer = LoggingPerfObserver()
        event = PerfEvent(operation="test.op", duration_ms=1.0)
        with caplog.at_level(logging.DEBUG, logger="pyiceberg.perf"):
            observer.emit(event)
        assert caplog.records[0].levelno == logging.DEBUG

    def test_no_log_at_info_level(self, caplog: pytest.LogCaptureFixture) -> None:
        observer = LoggingPerfObserver()
        event = PerfEvent(operation="test.op", duration_ms=1.0)
        with caplog.at_level(logging.INFO, logger="pyiceberg.perf"):
            observer.emit(event)
        assert len(caplog.records) == 0


class TestCompositeObserver:
    def test_fans_out_to_all_observers(self) -> None:
        a = CollectingObserver()
        b = CollectingObserver()
        composite = CompositeObserver(a, b)
        event = PerfEvent(operation="test.op", duration_ms=1.0)
        composite.emit(event)
        assert len(a.events) == 1
        assert len(b.events) == 1
        assert a.events[0] is event
        assert b.events[0] is event

    def test_empty_composite_does_not_raise(self) -> None:
        composite = CompositeObserver()
        composite.emit(PerfEvent(operation="test.op", duration_ms=1.0))

    def test_satisfies_protocol(self) -> None:
        assert isinstance(CompositeObserver(), PerfObserver)


class TestSetGetObserver:
    def test_default_is_null_observer(self) -> None:
        assert isinstance(get_observer(), NullPerfObserver)

    def test_set_and_get(self) -> None:
        collecting = CollectingObserver()
        set_observer(collecting)
        assert get_observer() is collecting

    def test_set_back_to_null(self) -> None:
        set_observer(CollectingObserver())
        set_observer(NullPerfObserver())
        assert isinstance(get_observer(), NullPerfObserver)


class TestPerfTimer:
    def test_measures_duration(self) -> None:
        collecting = CollectingObserver()
        set_observer(collecting)
        with perf_timer("test.op"):
            pass
        assert len(collecting.events) == 1
        assert collecting.events[0].operation == "test.op"
        assert collecting.events[0].duration_ms >= 0

    def test_captures_initial_tags(self) -> None:
        collecting = CollectingObserver()
        set_observer(collecting)
        with perf_timer("test.op", table="db.tbl", db="prod"):
            pass
        tags = collecting.events[0].tags
        assert tags["table"] == "db.tbl"
        assert tags["db"] == "prod"

    def test_tag_and_metric_in_body(self) -> None:
        collecting = CollectingObserver()
        set_observer(collecting)
        with perf_timer("test.op") as t:
            t.tag("db", "prod")
            t.metric("rows", 100)
            t.metric("bytes", 2048)
        event = collecting.events[0]
        assert event.tags["db"] == "prod"
        assert event.metrics["rows"] == 100
        assert event.metrics["bytes"] == 2048

    def test_tag_overwrites_initial(self) -> None:
        collecting = CollectingObserver()
        set_observer(collecting)
        with perf_timer("test.op", status="pending") as t:
            t.tag("status", "done")
        assert collecting.events[0].tags["status"] == "done"

    def test_emits_on_exception(self) -> None:
        collecting = CollectingObserver()
        set_observer(collecting)
        try:
            with perf_timer("test.op"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert len(collecting.events) == 1
        assert collecting.events[0].operation == "test.op"
        assert collecting.events[0].duration_ms >= 0

    def test_null_observer_skips_timing(self) -> None:
        set_observer(NullPerfObserver())
        with patch("pyiceberg.observability.time.monotonic") as mock_monotonic:
            with perf_timer("test.op"):
                pass
            mock_monotonic.assert_not_called()

    def test_active_observer_calls_timing(self) -> None:
        set_observer(CollectingObserver())
        with patch("pyiceberg.observability.time.monotonic", side_effect=[1.0, 2.0]) as mock_monotonic:
            with perf_timer("test.op"):
                pass
            assert mock_monotonic.call_count == 2

    def test_yields_perf_timer_context(self) -> None:
        collecting = CollectingObserver()
        set_observer(collecting)
        with perf_timer("test.op") as t:
            assert isinstance(t, _PerfTimerContext)


class TestTimedDecorator:
    def test_basic_function(self) -> None:
        collecting = CollectingObserver()
        set_observer(collecting)

        @timed("test.add")
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5
        assert len(collecting.events) == 1
        assert collecting.events[0].operation == "test.add"
        assert collecting.events[0].duration_ms >= 0

    def test_with_decorator_tags(self) -> None:
        collecting = CollectingObserver()
        set_observer(collecting)

        @timed("test.op", source="test")
        def noop() -> None:
            pass

        noop()
        assert collecting.events[0].tags["source"] == "test"

    def test_preserves_function_name(self) -> None:
        @timed("test.op")
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"

    def test_null_observer_skips_timing(self) -> None:
        set_observer(NullPerfObserver())

        @timed("test.op")
        def noop() -> None:
            pass

        with patch("pyiceberg.observability.time.monotonic") as mock_monotonic:
            noop()
            mock_monotonic.assert_not_called()

    def test_propagates_exception(self) -> None:
        collecting = CollectingObserver()
        set_observer(collecting)

        @timed("test.op")
        def fail() -> None:
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            fail()
        assert len(collecting.events) == 1
