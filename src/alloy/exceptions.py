"""
Custom exception hierarchy for Alloy.

Hierarchy:
    AlloyError (Base)
    ├── ConversionError
    │   ├── WorkerError          # Subprocess failures with exit codes
    │   └── UnsupportedModelError # Model format not supported
    ├── ValidationError
    │   └── ConfigError          # Batch config issues
    ├── DownloadError
    │   └── HuggingFaceError     # HF Hub failures
    └── DependencyError          # Missing packages
"""

from typing import Optional


class AlloyError(Exception):
    """Base exception for all Alloy errors."""

    def __init__(self, message: str, model_name: Optional[str] = None):
        self.model_name = model_name
        self.message = message
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.model_name:
            return f"[{self.model_name}] {self.message}"
        return self.message


# --- Conversion Errors ---


class ConversionError(AlloyError):
    """Base class for errors during model conversion."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        phase: Optional[str] = None,
    ):
        self.phase = phase
        super().__init__(message, model_name)

    def _format_message(self) -> str:
        parts = []
        if self.model_name:
            parts.append(f"[{self.model_name}]")
        if self.phase:
            parts.append(f"({self.phase})")
        parts.append(self.message)
        return " ".join(parts)


class WorkerError(ConversionError):
    """Raised when a subprocess worker fails."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        phase: Optional[str] = None,
        exit_code: Optional[int] = None,
        suggestions: Optional[list] = None,
    ):
        self.exit_code = exit_code
        self.suggestions = suggestions or []
        super().__init__(message, model_name, phase)

    def _format_message(self) -> str:
        base = super()._format_message()
        if self.exit_code is not None:
            base += f" (exit code: {self.exit_code})"
        return base


class UnsupportedModelError(ConversionError):
    """Raised when a model type or format is not supported."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        supported_types: Optional[list] = None,
    ):
        self.model_type = model_type
        self.supported_types = supported_types or []
        super().__init__(message, model_name)

    def _format_message(self) -> str:
        base = super()._format_message()
        if self.supported_types:
            base += f" Supported: {', '.join(self.supported_types)}"
        return base


# --- Validation Errors ---


class ValidationError(AlloyError):
    """Base class for validation errors."""

    pass


class ConfigError(ValidationError):
    """Raised for batch config or model config issues."""

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        missing_fields: Optional[list] = None,
        suggestions: Optional[list] = None,
    ):
        self.config_file = config_file
        self.missing_fields = missing_fields or []
        self.suggestions = suggestions or []
        super().__init__(message)

    def _format_message(self) -> str:
        base = self.message
        if self.config_file:
            base = f"[{self.config_file}] {base}"
        if self.missing_fields:
            base += f" Missing: {', '.join(self.missing_fields)}"
        return base


# --- Download Errors ---


class DownloadError(AlloyError):
    """Base class for download-related errors."""

    pass


class HuggingFaceError(DownloadError):
    """Raised for Hugging Face Hub API failures."""

    def __init__(
        self,
        message: str,
        repo_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
        suggestions: Optional[list] = None,
    ):
        self.repo_id = repo_id
        self.original_error = original_error
        self.suggestions = suggestions or []
        super().__init__(message)

    def _format_message(self) -> str:
        base = self.message
        if self.repo_id:
            base = f"[{self.repo_id}] {base}"
        return base


# --- Dependency Errors ---


class DependencyError(AlloyError):
    """Raised when required packages are missing."""

    def __init__(
        self,
        message: str,
        package_name: Optional[str] = None,
        install_command: Optional[str] = None,
        suggestions: Optional[list] = None,
    ):
        self.package_name = package_name
        self.install_command = install_command
        self.suggestions = suggestions or []
        super().__init__(message)

    def _format_message(self) -> str:
        base = self.message
        if self.install_command:
            base += f"\nInstall with: {self.install_command}"
        return base
