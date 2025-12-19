"""Progress tracking infrastructure for Alloy conversions.

Provides progress events, reporting, and Rich-based display for real-time
conversion progress with phases, steps, elapsed time, and memory monitoring.
"""

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Queue
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from alloy.monitor import get_memory_status


class ProgressPhase(str, Enum):
    """Phases of the conversion process."""

    DOWNLOAD = "download"
    PART1 = "part1"
    PART2 = "part2"
    ASSEMBLY = "assembly"


class ProgressStep(str, Enum):
    """Steps within a conversion phase."""

    LOAD = "load"
    TRACE = "trace"
    CONVERT = "convert"
    QUANTIZE = "quantize"
    SAVE = "save"


# Phase weights for ETA estimation (must sum to 1.0)
PHASE_WEIGHTS = {
    ProgressPhase.DOWNLOAD: 0.10,
    ProgressPhase.PART1: 0.40,
    ProgressPhase.PART2: 0.35,
    ProgressPhase.ASSEMBLY: 0.15,
}

# Step weights within a phase (must sum to 1.0)
STEP_WEIGHTS = {
    ProgressStep.LOAD: 0.15,
    ProgressStep.TRACE: 0.25,
    ProgressStep.CONVERT: 0.35,
    ProgressStep.QUANTIZE: 0.20,
    ProgressStep.SAVE: 0.05,
}


@dataclass
class ProgressEvent:
    """A progress event sent from worker to parent."""

    event_type: str  # "phase_start", "phase_end", "step_start", "step_end", "memory"
    phase: Optional[str] = None
    step: Optional[str] = None
    message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    memory_used_gb: Optional[float] = None
    memory_total_gb: Optional[float] = None


class ProgressReporter:
    """Send progress events from worker to parent via queue."""

    def __init__(self, queue: Queue, model_name: str):
        """Initialize reporter.

        Args:
            queue: Multiprocessing queue for sending events.
            model_name: Name of the model being converted.
        """
        self.queue = queue
        self.model_name = model_name

    def phase_start(self, phase: str, message: str) -> None:
        """Signal start of a phase."""
        self._send_event("phase_start", phase=phase, message=message)

    def phase_end(self, phase: str) -> None:
        """Signal end of a phase."""
        self._send_event("phase_end", phase=phase)

    def step_start(self, step: str, message: str) -> None:
        """Signal start of a step within a phase."""
        self._send_event("step_start", step=step, message=message)

    def step_end(self, step: str) -> None:
        """Signal end of a step."""
        self._send_event("step_end", step=step)

    def memory_snapshot(self) -> None:
        """Send current memory status."""
        status = get_memory_status()
        if status:
            self._send_event(
                "memory",
                memory_used_gb=status.used_gb,
                memory_total_gb=status.total_gb,
            )

    def _send_event(self, event_type: str, **kwargs) -> None:
        """Send an event to the queue."""
        event = ProgressEvent(event_type=event_type, **kwargs)
        try:
            self.queue.put_nowait(event)
        except Exception:
            pass  # Don't fail conversion if queue is full


class ProgressDisplay:
    """Rich Live display consuming progress events."""

    def __init__(self, model_name: str, quantization: str):
        """Initialize display.

        Args:
            model_name: Name of the model being converted.
            quantization: Quantization type.
        """
        self.model_name = model_name
        self.quantization = quantization
        self.console = Console()
        self.live: Optional[Live] = None

        # State
        self.current_phase: Optional[str] = None
        self.current_step: Optional[str] = None
        self.phase_message: str = ""
        self.step_message: str = ""
        self.start_time = time.time()
        self.memory_used_gb: float = 0.0
        self.memory_total_gb: float = 0.0
        self.completed_phases: set = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the live display."""
        self.start_time = time.time()
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=2,
            transient=True,
        )
        self.live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self.live:
            self.live.stop()
            self.live = None

    def process_event(self, event: ProgressEvent) -> None:
        """Process a progress event and update display."""
        with self._lock:
            if event.event_type == "phase_start":
                self.current_phase = event.phase
                self.phase_message = event.message or ""
                self.current_step = None
                self.step_message = ""
            elif event.event_type == "phase_end":
                if event.phase:
                    self.completed_phases.add(event.phase)
            elif event.event_type == "step_start":
                self.current_step = event.step
                self.step_message = event.message or ""
            elif event.event_type == "step_end":
                pass  # Step will be updated by next step_start
            elif event.event_type == "memory":
                if event.memory_used_gb is not None:
                    self.memory_used_gb = event.memory_used_gb
                if event.memory_total_gb is not None:
                    self.memory_total_gb = event.memory_total_gb

        self._update()

    def _update(self) -> None:
        """Update the live display."""
        if self.live:
            self.live.update(self._render())

    def _render(self) -> Panel:
        """Render the progress panel."""
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left")
        table.add_column(justify="right")

        # Header
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_duration(elapsed)
        status = "Running" if self.current_phase else "Starting"

        table.add_row(
            f"[bold]{self.model_name}[/bold] Conversion ({self.quantization})",
            f"[cyan]{status}[/cyan]",
        )
        table.add_row("", "")

        # Phase info
        phase_display = self.phase_message or self.current_phase or "Initializing..."
        table.add_row(f"[dim]Phase:[/dim] {phase_display}", "")

        # Step info
        if self.step_message:
            table.add_row(f"[dim]Step:[/dim]  {self.step_message}", "")
        else:
            table.add_row("[dim]Step:[/dim]  Waiting...", "")

        table.add_row("", "")

        # Time info
        eta_str = self._estimate_eta(elapsed)
        table.add_row(f"[dim]Elapsed:[/dim] {elapsed_str}", f"[dim]ETA:[/dim] {eta_str}")

        table.add_row("", "")

        # Memory info
        if self.memory_total_gb > 0:
            pct = (self.memory_used_gb / self.memory_total_gb) * 100
            mem_str = f"{self.memory_used_gb:.1f} / {self.memory_total_gb:.1f} GB ({pct:.0f}%)"
            table.add_row(f"[dim]Memory:[/dim] {mem_str}", "")

            # Memory bar
            bar_width = 40
            filled = int((pct / 100) * bar_width)
            bar = "[green]" + "█" * filled + "[/green]" + "░" * (bar_width - filled)
            table.add_row(bar, "")

        return Panel(table, title="Conversion Progress", border_style="blue")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _estimate_eta(self, elapsed: float) -> str:
        """Estimate remaining time based on completed phases."""
        if not self.completed_phases and not self.current_phase:
            return "calculating..."

        # Calculate completed weight
        completed_weight = sum(
            PHASE_WEIGHTS.get(ProgressPhase(p), 0) for p in self.completed_phases
        )

        # Add partial weight for current phase
        if self.current_phase and self.current_phase not in self.completed_phases:
            phase_weight = PHASE_WEIGHTS.get(ProgressPhase(self.current_phase), 0)
            # Assume halfway through current phase if no step info
            step_progress = 0.5
            if self.current_step:
                # Calculate step progress
                steps = list(ProgressStep)
                try:
                    step_idx = steps.index(ProgressStep(self.current_step))
                    step_progress = (step_idx + 0.5) / len(steps)
                except ValueError:
                    pass
            completed_weight += phase_weight * step_progress

        if completed_weight <= 0:
            return "calculating..."

        # Estimate total time
        total_estimated = elapsed / completed_weight
        remaining = total_estimated - elapsed

        if remaining <= 0:
            return "almost done"

        return f"~{self._format_duration(remaining)} remaining"


def consume_progress_queue(
    queue: Queue,
    display: ProgressDisplay,
    stop_event: threading.Event,
) -> None:
    """Consumer thread that processes progress events.

    Args:
        queue: Queue to consume events from.
        display: Display to update.
        stop_event: Event to signal stop.
    """
    while not stop_event.is_set():
        try:
            event = queue.get(timeout=0.1)
            if event is None:  # Sentinel to stop
                break
            display.process_event(event)
        except Exception:
            continue  # Timeout or error, keep trying
