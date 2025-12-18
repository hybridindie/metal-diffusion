"""
Alloy logging configuration module.

Provides centralized logging setup with Rich console formatting,
subprocess worker support, and optional file/JSON logging.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from enum import IntEnum
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from multiprocessing import Queue
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


class Verbosity(IntEnum):
    """Logging verbosity levels."""

    QUIET = 0  # Only errors
    NORMAL = 1  # Info and above (default)
    VERBOSE = 2  # Debug for alloy.*
    DEBUG = 3  # Debug for all loggers


# Global state for multiprocessing queue listener
_log_queue: Optional[Queue] = None
_queue_listener: Optional[QueueListener] = None
_initialized: bool = False


class AlloyJSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "model_name"):
            log_obj["model_name"] = record.model_name
        if hasattr(record, "phase"):
            log_obj["phase"] = record.phase
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


class MarkupStripFormatter(logging.Formatter):
    """Formatter that strips Rich markup for file logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format and strip Rich markup tags."""
        message = super().format(record)
        # Strip common Rich markup tags
        import re

        return re.sub(r"\[/?[a-zA-Z_ ]+\]", "", message)


def setup_logging(
    verbosity: Verbosity = Verbosity.NORMAL,
    log_file: Optional[str] = None,
    json_logging: bool = False,
    force_color: Optional[bool] = None,
) -> logging.Logger:
    """
    Configure the root logger for Alloy CLI.

    Args:
        verbosity: Logging verbosity level
        log_file: Optional path to log file
        json_logging: Enable JSON structured logging
        force_color: Force color output (None = auto-detect)

    Returns:
        The configured root logger for 'alloy'
    """
    global _log_queue, _queue_listener, _initialized

    if _initialized:
        return logging.getLogger("alloy")

    # Determine log levels based on verbosity
    if verbosity == Verbosity.QUIET:
        alloy_level = logging.ERROR
        root_level = logging.ERROR
    elif verbosity == Verbosity.NORMAL:
        alloy_level = logging.INFO
        root_level = logging.WARNING
    elif verbosity == Verbosity.VERBOSE:
        alloy_level = logging.DEBUG
        root_level = logging.INFO
    else:  # DEBUG
        alloy_level = logging.DEBUG
        root_level = logging.DEBUG

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create console handler with Rich
    console = Console(force_terminal=force_color)
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(root_level)

    if json_logging:
        rich_handler.setFormatter(AlloyJSONFormatter())
    else:
        rich_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.addHandler(rich_handler)

    # Configure alloy logger
    alloy_logger = logging.getLogger("alloy")
    alloy_logger.setLevel(alloy_level)

    # Add file handler if requested
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file

        if json_logging:
            file_handler.setFormatter(AlloyJSONFormatter())
        else:
            file_handler.setFormatter(
                MarkupStripFormatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

        root_logger.addHandler(file_handler)

    # Setup multiprocessing queue for worker logging
    _log_queue = Queue()
    _queue_listener = QueueListener(_log_queue, *root_logger.handlers)
    _queue_listener.start()

    _initialized = True
    return alloy_logger


def get_log_queue() -> Optional[Queue]:
    """
    Get the multiprocessing queue for subprocess logging.

    Returns:
        The logging queue, or None if not initialized.
    """
    return _log_queue


def setup_worker_logging(queue: Optional[Queue] = None) -> logging.Logger:
    """
    Configure logging for a subprocess worker.

    Should be called at the start of each worker function.
    Logs are forwarded to the main process via queue.

    Args:
        queue: The logging queue from the parent process.
               If None, falls back to basic stderr logging.

    Returns:
        Logger configured for the worker.
    """
    # Clear any existing handlers in worker process
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    if queue is not None:
        # Send logs to parent process via queue
        queue_handler = QueueHandler(queue)
        root_logger.addHandler(queue_handler)
    else:
        # Fallback: basic stderr handler with Rich
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)

    return logging.getLogger("alloy")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name, prefixed with 'alloy.'.

    This is the primary way modules should obtain loggers.

    Args:
        name: Module name (typically __name__)

    Returns:
        A configured logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing model...")
    """
    # If name already starts with alloy, use as-is
    if name.startswith("alloy.") or name == "alloy":
        return logging.getLogger(name)

    # Otherwise, extract the relevant part and prefix with alloy
    # Handle both 'alloy.module' and 'src.alloy.module' patterns
    parts = name.split(".")
    if "alloy" in parts:
        idx = parts.index("alloy")
        return logging.getLogger(".".join(parts[idx:]))

    return logging.getLogger(f"alloy.{name}")


def shutdown_logging() -> None:
    """
    Clean up logging resources.

    Should be called before program exit to ensure all log
    messages are flushed and the queue listener is stopped.
    """
    global _queue_listener, _log_queue, _initialized

    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None

    if _log_queue is not None:
        _log_queue.close()
        _log_queue.join_thread()
        _log_queue = None

    _initialized = False


def parse_log_level(level_str: str) -> Verbosity:
    """
    Parse a log level string to Verbosity enum.

    Args:
        level_str: One of 'debug', 'verbose', 'info', 'warning', 'error', 'quiet'

    Returns:
        Corresponding Verbosity level
    """
    level_str = level_str.lower().strip()
    mapping = {
        "debug": Verbosity.DEBUG,
        "verbose": Verbosity.VERBOSE,
        "info": Verbosity.NORMAL,
        "warning": Verbosity.NORMAL,
        "error": Verbosity.QUIET,
        "quiet": Verbosity.QUIET,
    }
    return mapping.get(level_str, Verbosity.NORMAL)
