"""Error suggestion helpers for actionable error messages.

This module provides functions to generate contextual suggestions
based on error conditions, helping users resolve issues.
"""

from typing import Optional


def get_worker_suggestions(exit_code: Optional[int]) -> list[str]:
    """Generate suggestions based on worker failure context.

    Args:
        exit_code: The exit code from the failed subprocess

    Returns:
        List of actionable suggestion strings
    """
    suggestions = []

    if exit_code == -9:  # SIGKILL - typically OOM killer on Linux/macOS
        suggestions.extend([
            "Close other applications to free memory",
            "Try a different quantization: --quantization int8",
            "Ensure at least 32GB RAM for large models",
        ])
    elif exit_code == -15:  # SIGTERM - graceful termination
        suggestions.append("Process was terminated externally")
    elif exit_code == 1:  # Generic failure
        suggestions.extend([
            "Check the error output above for details",
            "Ensure all dependencies are installed",
        ])
    elif exit_code is not None and exit_code < 0:
        # Other signals (negative exit codes)
        suggestions.append(f"Process terminated by signal {-exit_code}")

    return suggestions


def get_download_suggestions(repo_id: Optional[str] = None) -> list[str]:
    """Generate suggestions for download failures.

    Args:
        repo_id: The Hugging Face repository ID that failed

    Returns:
        List of actionable suggestion strings
    """
    suggestions = []

    if repo_id:
        suggestions.append(f"Verify the model ID is correct: {repo_id}")

    suggestions.extend([
        "Check your internet connection",
        "If this is a private model, ensure HF_TOKEN is set",
        "Check Hugging Face status: https://status.huggingface.co",
    ])

    return suggestions


def get_config_suggestions(missing_fields: Optional[list[str]] = None) -> list[str]:
    """Generate suggestions for configuration errors.

    Args:
        missing_fields: List of missing required fields

    Returns:
        List of actionable suggestion strings
    """
    suggestions = []

    if missing_fields:
        suggestions.append(f"Add missing fields: {', '.join(missing_fields)}")

    suggestions.extend([
        "Check the batch file format (JSON or YAML)",
        "Each entry requires 'model' and 'type' fields",
    ])

    return suggestions


def get_dependency_suggestions(package_name: Optional[str] = None) -> list[str]:
    """Generate suggestions for missing dependency errors.

    Args:
        package_name: Name of the missing package

    Returns:
        List of actionable suggestion strings
    """
    suggestions = []

    if package_name:
        suggestions.append(f"Install with: pip install {package_name}")

    suggestions.append("Ensure you're using the correct Python environment")

    return suggestions
