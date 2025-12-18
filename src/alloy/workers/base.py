"""Base utilities for worker subprocess execution.

These utilities standardize subprocess execution patterns across all worker functions,
providing consistent error handling, memory cleanup, and logging.
"""

import gc
import os
import uuid
import tempfile
from contextlib import contextmanager
from multiprocessing import Queue
from typing import Optional, Any

from alloy.logging import setup_worker_logging, get_logger
from alloy.utils.coreml import safe_quantize_model

logger = get_logger(__name__)


@contextmanager
def worker_context(model_name: str, part_description: str, log_queue: Optional[Queue] = None):
    """Context manager for worker execution with logging and cleanup.

    Logs a start message on entry and a completion message on successful exit.
    Garbage collection runs on exit regardless of success or failure.

    Note: The completion message is only printed on success. If an exception
    occurs, cleanup runs but the completion message is skipped.

    Usage:
        with worker_context("Flux", "Part 1", log_queue):
            # ... do conversion work ...

    Args:
        model_name: Name of the model being converted (e.g., "Flux", "Wan")
        part_description: Description of the conversion phase (e.g., "Part 1", "Part 2")
        log_queue: Optional multiprocessing queue for forwarding logs to parent
    """
    # Setup worker logging if queue provided
    if log_queue is not None:
        setup_worker_logging(log_queue)

    logger.info(
        "[cyan]Worker: Starting %s %s (PID: %d)[/cyan]",
        model_name,
        part_description,
        os.getpid(),
        extra={"markup": True},
    )

    try:
        yield
    finally:
        gc.collect()

    logger.info(
        "[green]Worker: %s %s Complete[/green]",
        model_name,
        part_description,
        extra={"markup": True},
    )


def quantize_and_save(
    model,
    output_path: str,
    quantization: Optional[str],
    intermediates_dir: Optional[str] = None,
    part_name: str = "model",
) -> None:
    """Apply quantization if needed and save the model.

    Handles memory-efficient quantization by saving to intermediate files
    and reloading to prevent OOM during quantization.

    Args:
        model: CoreML model to quantize
        output_path: Final output path for the model
        quantization: Quantization type (int4, int8, etc.) or None for no quantization
        intermediates_dir: Directory for intermediate files (uses temp dir if None)
        part_name: Name for intermediate file (e.g., "part1", "part2")
    """
    if not quantization:
        model.save(output_path)
        return

    logger.debug(
        "[dim]Worker: Quantizing %s (%s)...[/dim]",
        part_name,
        quantization,
        extra={"markup": True},
    )

    if intermediates_dir:
        fp16_path = os.path.join(
            intermediates_dir, f"{part_name}_fp16_{uuid.uuid4()}.mlpackage"
        )
        model.save(fp16_path)
        del model
        gc.collect()
        model = safe_quantize_model(
            fp16_path, quantization, intermediate_dir=intermediates_dir
        )
    else:
        with tempfile.TemporaryDirectory() as tmp:
            fp16_path = os.path.join(tmp, f"{part_name}_fp16.mlpackage")
            model.save(fp16_path)
            del model
            gc.collect()
            model = safe_quantize_model(fp16_path, quantization)

    logger.debug(
        "[dim]Worker: Saving %s to %s...[/dim]",
        part_name,
        output_path,
        extra={"markup": True},
    )
    model.save(output_path)


def load_transformer_with_fallback(
    model_class: type,
    model_id: str,
    torch_dtype,
    subfolder: str = "transformer",
    enable_logging: bool = True,
) -> Any:
    """Load transformer with subfolder fallback pattern.

    Attempts to load from a subfolder first (common for HuggingFace repos),
    then falls back to loading from the root.

    Args:
        model_class: The transformer model class to instantiate
        model_id: HuggingFace model ID or local path
        torch_dtype: Torch dtype for model weights
        subfolder: Subfolder to try first (default: "transformer")
        enable_logging: Whether to log progress messages

    Returns:
        The loaded transformer model
    """
    try:
        if enable_logging:
            logger.debug(
                "[dim]Attempting to load transformer from '%s' subfolder...[/dim]",
                subfolder,
                extra={"markup": True},
            )
        return model_class.from_pretrained(
            model_id,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
        )
    except (EnvironmentError, OSError):
        if enable_logging:
            logger.debug(
                "[dim]Subfolder load failed, trying root...[/dim]",
                extra={"markup": True},
            )
        return model_class.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
