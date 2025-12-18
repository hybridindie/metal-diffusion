"""Base utilities for worker subprocess execution.

These utilities standardize subprocess execution patterns across all worker functions,
providing consistent error handling, memory cleanup, and logging.
"""

import gc
import os
import uuid
import tempfile
from contextlib import contextmanager
from typing import Optional, Any

from rich.console import Console

from alloy.utils.coreml import safe_quantize_model

console = Console()


@contextmanager
def worker_context(model_name: str, part_description: str):
    """Context manager for worker execution with logging and cleanup.

    Provides standardized start/complete logging and ensures garbage collection
    runs on exit.

    Usage:
        with worker_context("Flux", "Part 1"):
            # ... do conversion work ...

    Args:
        model_name: Name of the model being converted (e.g., "Flux", "Wan")
        part_description: Description of the conversion phase (e.g., "Part 1", "Part 2")
    """
    console.print(
        f"[cyan]Worker: Starting {model_name} {part_description} "
        f"(PID: {os.getpid()})[/cyan]"
    )

    try:
        yield
    finally:
        gc.collect()

    console.print(f"[green]Worker: {model_name} {part_description} Complete[/green]")


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

    console.print(f"[dim]Worker: Quantizing {part_name} ({quantization})...[/dim]")

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

    console.print(f"[dim]Worker: Saving {part_name} to {output_path}...[/dim]")
    model.save(output_path)


def load_transformer_with_fallback(
    model_class: type,
    model_id: str,
    torch_dtype,
    subfolder: str = "transformer",
    console_logging: bool = True,
) -> Any:
    """Load transformer with subfolder fallback pattern.

    Attempts to load from a subfolder first (common for HuggingFace repos),
    then falls back to loading from the root.

    Args:
        model_class: The transformer model class to instantiate
        model_id: HuggingFace model ID or local path
        torch_dtype: Torch dtype for model weights
        subfolder: Subfolder to try first (default: "transformer")
        console_logging: Whether to log progress messages

    Returns:
        The loaded transformer model
    """
    try:
        if console_logging:
            console.print(
                f"[dim]Attempting to load transformer from '{subfolder}' subfolder...[/dim]"
            )
        return model_class.from_pretrained(
            model_id,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
        )
    except (EnvironmentError, OSError):
        if console_logging:
            console.print("[dim]Subfolder load failed, trying root...[/dim]")
        return model_class.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
