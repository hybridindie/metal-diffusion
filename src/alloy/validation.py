"""
Pre-flight validation for Alloy conversions.

Validates system state before starting expensive conversion operations
to catch issues early and provide actionable error messages.
"""

import os
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from alloy.exceptions import DependencyError, ValidationError
from alloy.logging import get_logger

if TYPE_CHECKING:
    from alloy.converters.base import ModelConverter

logger = get_logger(__name__)


class Severity(Enum):
    """Severity levels for validation results."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    passed: bool
    message: str
    severity: Severity
    suggestion: Optional[str] = None


class PreflightValidator:
    """
    Validates system state before conversion.

    Checks disk space, memory, dependencies, and model accessibility
    to catch issues before expensive operations begin.
    """

    # HuggingFace repo ID pattern: org/repo or org/repo/revision
    HF_REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+(/[\w.-]+)?$")

    def __init__(self, converter: "ModelConverter"):
        """
        Initialize validator with converter instance.

        Args:
            converter: The ModelConverter to validate
        """
        self.converter = converter
        self._model_size_gb: Optional[float] = None

    def validate_all(self) -> List[ValidationResult]:
        """
        Run all validation checks.

        Returns:
            List of ValidationResult objects
        """
        results = []
        results.append(self._check_output_directory())
        results.append(self._check_model_access())
        results.append(self._check_disk_space())
        results.append(self._check_memory())
        results.append(self._check_dependencies())
        return results

    def _get_model_size_gb(self) -> Optional[float]:
        """
        Get model size in GB by inspecting the actual model.

        For local paths: sum file sizes in directory or single file
        For HF repos: use huggingface_hub API to get repo size
        Caches result for reuse.

        Returns:
            Model size in GB, or None if unavailable
        """
        if self._model_size_gb is not None:
            return self._model_size_gb

        model_id = self.converter.model_id

        if os.path.exists(model_id):
            # Local path - calculate actual size
            if os.path.isfile(model_id):
                size_bytes = os.path.getsize(model_id)
            else:
                size_bytes = sum(
                    os.path.getsize(os.path.join(root, f))
                    for root, _, files in os.walk(model_id)
                    for f in files
                )
            self._model_size_gb = size_bytes / (1024**3)
        else:
            # HF repo - try to get size from API
            try:
                from huggingface_hub import HfApi

                api = HfApi()
                info = api.repo_info(model_id)
                # siblings contains file info
                size_bytes = sum(s.size or 0 for s in info.siblings)
                self._model_size_gb = size_bytes / (1024**3)
            except Exception:
                # Fallback: estimate based on model type
                self._model_size_gb = self._estimate_size_by_type()

        return self._model_size_gb

    def _estimate_size_by_type(self) -> float:
        """
        Fallback size estimation when actual size unavailable.

        Returns:
            Estimated model size in GB
        """
        defaults = {
            "flux": 24,
            "hunyuan": 25,
            "ltx": 5,
            "wan": 6,
            "lumina": 8,
            "sd": 4,
        }
        model_name = getattr(self.converter, "model_name", "sd").lower()
        return defaults.get(model_name, 10)

    def _estimate_disk_requirement(self) -> float:
        """
        Estimate disk space needed for conversion.

        Formula: source + converted + intermediates = ~4x source
        """
        source_size = self._get_model_size_gb() or 10
        return source_size * 4

    def _estimate_ram_requirement(self) -> float:
        """
        Estimate RAM needed for conversion.

        Based on model size: typically need ~2x model size in RAM
        for loading + tracing overhead.
        """
        source_size = self._get_model_size_gb() or 10
        return source_size * 2

    def _check_output_directory(self) -> ValidationResult:
        """Check output directory exists and is writable."""
        output_dir = self.converter.output_dir

        # Check if parent directory exists (we'll create output_dir)
        parent_dir = os.path.dirname(output_dir) or "."

        if not os.path.exists(parent_dir):
            return ValidationResult(
                passed=False,
                message=f"Parent directory does not exist: {parent_dir}",
                severity=Severity.ERROR,
                suggestion=f"Create the directory with: mkdir -p {parent_dir}",
            )

        # Check if parent is writable
        if not os.access(parent_dir, os.W_OK):
            return ValidationResult(
                passed=False,
                message=f"Cannot write to directory: {parent_dir}",
                severity=Severity.ERROR,
                suggestion="Check directory permissions or choose a different output location",
            )

        # If output_dir exists, check if writable
        if os.path.exists(output_dir) and not os.access(output_dir, os.W_OK):
            return ValidationResult(
                passed=False,
                message=f"Output directory is not writable: {output_dir}",
                severity=Severity.ERROR,
                suggestion="Check directory permissions or choose a different output location",
            )

        return ValidationResult(
            passed=True,
            message="Output directory is writable",
            severity=Severity.INFO,
        )

    def _check_disk_space(self) -> ValidationResult:
        """Check sufficient disk space for conversion."""
        output_dir = self.converter.output_dir
        parent_dir = os.path.dirname(output_dir) or "."

        # Use parent dir if output doesn't exist yet
        check_dir = output_dir if os.path.exists(output_dir) else parent_dir
        if not os.path.exists(check_dir):
            check_dir = "."

        try:
            usage = shutil.disk_usage(check_dir)
            available_gb = usage.free / (1024**3)
        except OSError:
            return ValidationResult(
                passed=True,
                message="Could not check disk space",
                severity=Severity.WARNING,
                suggestion="Ensure sufficient disk space is available",
            )

        required_gb = self._estimate_disk_requirement()
        warning_threshold = required_gb * 1.5  # 6x model size

        if available_gb < required_gb:
            return ValidationResult(
                passed=False,
                message=f"Insufficient disk space: {available_gb:.1f}GB available, {required_gb:.1f}GB required",
                severity=Severity.ERROR,
                suggestion=f"Free up at least {required_gb - available_gb:.1f}GB of disk space",
            )

        if available_gb < warning_threshold:
            return ValidationResult(
                passed=True,
                message=f"Low disk space: {available_gb:.1f}GB available, {required_gb:.1f}GB required",
                severity=Severity.WARNING,
                suggestion="Consider freeing up additional disk space",
            )

        return ValidationResult(
            passed=True,
            message=f"Disk space OK: {available_gb:.1f}GB available",
            severity=Severity.INFO,
        )

    def _check_memory(self) -> ValidationResult:
        """Check available RAM meets requirements."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
        except ImportError:
            return ValidationResult(
                passed=True,
                message="psutil not installed, skipping memory check",
                severity=Severity.INFO,
                suggestion="Install psutil for memory validation: pip install psutil",
            )
        except Exception:
            return ValidationResult(
                passed=True,
                message="Could not check memory",
                severity=Severity.WARNING,
            )

        required_gb = self._estimate_ram_requirement()

        if available_gb < required_gb:
            return ValidationResult(
                passed=True,  # Warning, not error
                message=f"Low memory: {available_gb:.1f}GB available, {required_gb:.1f}GB recommended",
                severity=Severity.WARNING,
                suggestion=f"Close other applications to free memory. Total RAM: {total_gb:.1f}GB",
            )

        return ValidationResult(
            passed=True,
            message=f"Memory OK: {available_gb:.1f}GB available",
            severity=Severity.INFO,
        )

    def _check_model_access(self) -> ValidationResult:
        """Check model is accessible (local path or valid HF ID)."""
        model_id = self.converter.model_id

        # Check if local path
        if os.path.exists(model_id):
            if os.path.isfile(model_id):
                return ValidationResult(
                    passed=True,
                    message=f"Local model file found: {model_id}",
                    severity=Severity.INFO,
                )
            else:
                return ValidationResult(
                    passed=True,
                    message=f"Local model directory found: {model_id}",
                    severity=Severity.INFO,
                )

        # Check if valid HuggingFace repo format
        if self.HF_REPO_PATTERN.match(model_id):
            return ValidationResult(
                passed=True,
                message=f"HuggingFace repo format valid: {model_id}",
                severity=Severity.INFO,
            )

        # Neither local path nor valid HF format
        return ValidationResult(
            passed=False,
            message=f"Model not found: {model_id}",
            severity=Severity.ERROR,
            suggestion="Provide a valid local path or HuggingFace repo ID (org/repo)",
        )

    def _check_dependencies(self) -> ValidationResult:
        """Check required packages are installed."""
        missing = []

        # Check coremltools
        try:
            import coremltools  # noqa: F401
        except ImportError:
            missing.append("coremltools")

        # Check torch
        try:
            import torch  # noqa: F401
        except ImportError:
            missing.append("torch")

        if missing:
            return ValidationResult(
                passed=False,
                message=f"Missing required packages: {', '.join(missing)}",
                severity=Severity.ERROR,
                suggestion=f"Install with: pip install {' '.join(missing)}",
            )

        return ValidationResult(
            passed=True,
            message="All required dependencies installed",
            severity=Severity.INFO,
        )


def run_preflight_validation(converter: "ModelConverter") -> None:
    """
    Run pre-flight validation and handle results.

    Logs info/warnings and raises exceptions for errors.

    Args:
        converter: The ModelConverter to validate

    Raises:
        ValidationError: If any validation check fails with ERROR severity
        DependencyError: If required dependencies are missing
    """
    validator = PreflightValidator(converter)
    results = validator.validate_all()

    errors = []
    for result in results:
        if result.severity == Severity.INFO:
            logger.debug(result.message)
        elif result.severity == Severity.WARNING:
            if result.suggestion:
                logger.warning(f"{result.message}. {result.suggestion}")
            else:
                logger.warning(result.message)
        elif result.severity == Severity.ERROR and not result.passed:
            errors.append(result)

    if errors:
        # Collect all error messages
        messages = [e.message for e in errors]
        suggestions = [e.suggestion for e in errors if e.suggestion]

        # Check if it's a dependency error
        dep_errors = [e for e in errors if "Missing required packages" in e.message]
        if dep_errors:
            # Extract package names from message
            pkg_info = dep_errors[0].message.split(": ")
            pkg_names = pkg_info[1] if len(pkg_info) > 1 else "unknown"
            raise DependencyError(
                message=f"Missing required packages: {pkg_names}",
                package_name=pkg_names,
                suggestions=suggestions,
            )

        # General validation error
        raise ValidationError(
            message="; ".join(messages),
            suggestions=suggestions,
        )

    logger.info("Pre-flight validation passed")
