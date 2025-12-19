"""Tests for alloy.progress module."""

import time
import threading
import pytest
from multiprocessing import Queue
from unittest.mock import patch, MagicMock

from alloy.progress import (
    ProgressPhase,
    ProgressStep,
    ProgressEvent,
    ProgressReporter,
    ProgressDisplay,
    consume_progress_queue,
    PHASE_WEIGHTS,
    STEP_WEIGHTS,
)


class TestProgressPhase:
    """Tests for ProgressPhase enum."""

    def test_has_expected_phases(self):
        """Test all expected phases exist."""
        assert ProgressPhase.DOWNLOAD.value == "download"
        assert ProgressPhase.PART1.value == "part1"
        assert ProgressPhase.PART2.value == "part2"
        assert ProgressPhase.ASSEMBLY.value == "assembly"


class TestProgressStep:
    """Tests for ProgressStep enum."""

    def test_has_expected_steps(self):
        """Test all expected steps exist."""
        assert ProgressStep.LOAD.value == "load"
        assert ProgressStep.TRACE.value == "trace"
        assert ProgressStep.CONVERT.value == "convert"
        assert ProgressStep.QUANTIZE.value == "quantize"
        assert ProgressStep.SAVE.value == "save"


class TestProgressEvent:
    """Tests for ProgressEvent dataclass."""

    def test_creates_with_required_field(self):
        """Test creating event with only required field."""
        event = ProgressEvent(event_type="phase_start")
        assert event.event_type == "phase_start"
        assert event.phase is None
        assert event.step is None
        assert event.message is None
        assert event.timestamp > 0

    def test_creates_with_all_fields(self):
        """Test creating event with all fields."""
        event = ProgressEvent(
            event_type="step_start",
            phase="part1",
            step="load",
            message="Loading model",
            memory_used_gb=16.0,
            memory_total_gb=64.0,
        )
        assert event.event_type == "step_start"
        assert event.phase == "part1"
        assert event.step == "load"
        assert event.message == "Loading model"
        assert event.memory_used_gb == 16.0
        assert event.memory_total_gb == 64.0


class TestProgressReporter:
    """Tests for ProgressReporter class."""

    def test_phase_start_sends_event(self):
        """Test phase_start sends correct event to queue."""
        queue = Queue()
        reporter = ProgressReporter(queue, "TestModel")

        reporter.phase_start("part1", "Starting Part 1")

        event = queue.get(timeout=1)
        assert event.event_type == "phase_start"
        assert event.phase == "part1"
        assert event.message == "Starting Part 1"

    def test_phase_end_sends_event(self):
        """Test phase_end sends correct event to queue."""
        queue = Queue()
        reporter = ProgressReporter(queue, "TestModel")

        reporter.phase_end("part1")

        # Allow time for queue
        time.sleep(0.1)
        events = []
        for _ in range(2):
            try:
                events.append(queue.get(timeout=1))
            except Exception:
                break

        phase_end_events = [e for e in events if e.event_type == "phase_end"]
        assert len(phase_end_events) >= 1
        assert phase_end_events[0].phase == "part1"

    def test_step_start_sends_event(self):
        """Test step_start sends correct event to queue."""
        queue = Queue()
        reporter = ProgressReporter(queue, "TestModel")

        reporter.step_start("load", "Loading transformer")

        event = queue.get(timeout=1)
        assert event.event_type == "step_start"
        assert event.step == "load"
        assert event.message == "Loading transformer"

    def test_step_end_sends_event(self):
        """Test step_end sends correct event to queue."""
        queue = Queue()
        reporter = ProgressReporter(queue, "TestModel")

        reporter.step_end("load")

        event = queue.get(timeout=1)
        assert event.event_type == "step_end"
        assert event.step == "load"


class TestProgressDisplay:
    """Tests for ProgressDisplay class."""

    def test_initialization(self):
        """Test display initializes with correct state."""
        display = ProgressDisplay("Flux", "int4")

        assert display.model_name == "Flux"
        assert display.quantization == "int4"
        assert display.current_phase is None
        assert display.current_step is None
        assert display.completed_phases == set()

    def test_process_phase_start_event(self):
        """Test processing phase_start event updates state."""
        display = ProgressDisplay("Flux", "int4")
        event = ProgressEvent(
            event_type="phase_start",
            phase="part1",
            message="Converting Part 1",
        )

        display.process_event(event)

        assert display.current_phase == "part1"
        assert display.phase_message == "Converting Part 1"

    def test_process_phase_end_event(self):
        """Test processing phase_end event updates completed phases."""
        display = ProgressDisplay("Flux", "int4")
        display.current_phase = "part1"

        event = ProgressEvent(event_type="phase_end", phase="part1")
        display.process_event(event)

        assert "part1" in display.completed_phases

    def test_process_step_event(self):
        """Test processing step_start event updates state."""
        display = ProgressDisplay("Flux", "int4")
        event = ProgressEvent(
            event_type="step_start",
            step="load",
            message="Loading model",
        )

        display.process_event(event)

        assert display.current_step == "load"
        assert display.step_message == "Loading model"

    def test_process_memory_event(self):
        """Test processing memory event updates state."""
        display = ProgressDisplay("Flux", "int4")
        event = ProgressEvent(
            event_type="memory",
            memory_used_gb=24.0,
            memory_total_gb=64.0,
        )

        display.process_event(event)

        assert display.memory_used_gb == 24.0
        assert display.memory_total_gb == 64.0

    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        display = ProgressDisplay("Flux", "int4")
        assert display._format_duration(45) == "45s"

    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        display = ProgressDisplay("Flux", "int4")
        assert display._format_duration(125) == "2m 5s"

    def test_format_duration_hours(self):
        """Test duration formatting for hours."""
        display = ProgressDisplay("Flux", "int4")
        assert display._format_duration(3725) == "1h 2m"


class TestConsumeProgressQueue:
    """Tests for consume_progress_queue function."""

    def test_processes_events_until_stop(self):
        """Test consumer processes events until stop event is set."""
        queue = Queue()
        display = MagicMock()
        stop_event = threading.Event()

        # Put some events
        queue.put(ProgressEvent(event_type="phase_start", phase="part1"))
        queue.put(ProgressEvent(event_type="step_start", step="load"))
        queue.put(None)  # Sentinel

        # Run consumer
        consume_progress_queue(queue, display, stop_event)

        # Verify events were processed
        assert display.process_event.call_count == 2

    def test_stops_on_stop_event(self):
        """Test consumer stops when stop event is set."""
        queue = Queue()
        display = MagicMock()
        stop_event = threading.Event()

        # Start consumer in thread
        thread = threading.Thread(
            target=consume_progress_queue,
            args=(queue, display, stop_event),
        )
        thread.start()

        # Give it a moment then stop
        time.sleep(0.1)
        stop_event.set()
        thread.join(timeout=1)

        assert not thread.is_alive()


class TestWeights:
    """Tests for phase and step weight constants."""

    def test_phase_weights_sum_to_one(self):
        """Test phase weights sum to approximately 1.0."""
        total = sum(PHASE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_step_weights_sum_to_one(self):
        """Test step weights sum to approximately 1.0."""
        total = sum(STEP_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_phases_have_weights(self):
        """Test all ProgressPhase values have weights."""
        for phase in ProgressPhase:
            assert phase in PHASE_WEIGHTS

    def test_all_steps_have_weights(self):
        """Test all ProgressStep values have weights."""
        for step in ProgressStep:
            assert step in STEP_WEIGHTS
