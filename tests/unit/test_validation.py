"""Tests for the pre-flight validation module."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from alloy.exceptions import DependencyError, ValidationError
from alloy.validation import (
    PreflightValidator,
    Severity,
    ValidationResult,
    run_preflight_validation,
)


class MockConverter:
    """Mock converter for testing."""

    def __init__(self, model_id="test/model", output_dir="/tmp/output", model_name="test"):
        self.model_id = model_id
        self.output_dir = output_dir
        self.model_name = model_name


class TestValidationResult(unittest.TestCase):
    """Tests for ValidationResult dataclass."""

    def test_create_passed_result(self):
        """Test creating a passing result."""
        result = ValidationResult(
            passed=True,
            message="Check passed",
            severity=Severity.INFO,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.message, "Check passed")
        self.assertEqual(result.severity, Severity.INFO)
        self.assertIsNone(result.suggestion)

    def test_create_failed_result_with_suggestion(self):
        """Test creating a failed result with suggestion."""
        result = ValidationResult(
            passed=False,
            message="Check failed",
            severity=Severity.ERROR,
            suggestion="Try this fix",
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.suggestion, "Try this fix")


class TestSeverity(unittest.TestCase):
    """Tests for Severity enum."""

    def test_severity_values(self):
        """Test severity level values."""
        self.assertEqual(Severity.ERROR.value, "error")
        self.assertEqual(Severity.WARNING.value, "warning")
        self.assertEqual(Severity.INFO.value, "info")


class TestPreflightValidatorOutputDirectory(unittest.TestCase):
    """Tests for output directory validation."""

    def test_output_directory_writable(self):
        """Test that writable directory passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = MockConverter(output_dir=os.path.join(tmpdir, "output"))
            validator = PreflightValidator(converter)
            result = validator._check_output_directory()
            self.assertTrue(result.passed)

    def test_output_directory_parent_not_exists(self):
        """Test that non-existent parent directory fails."""
        converter = MockConverter(output_dir="/nonexistent/path/output")
        validator = PreflightValidator(converter)
        result = validator._check_output_directory()
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, Severity.ERROR)
        self.assertIn("does not exist", result.message)

    @patch("os.access")
    def test_output_directory_not_writable(self, mock_access):
        """Test that non-writable directory fails."""
        mock_access.return_value = False
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = MockConverter(output_dir=os.path.join(tmpdir, "output"))
            validator = PreflightValidator(converter)
            result = validator._check_output_directory()
            self.assertFalse(result.passed)
            self.assertEqual(result.severity, Severity.ERROR)


class TestPreflightValidatorDiskSpace(unittest.TestCase):
    """Tests for disk space validation."""

    def test_disk_space_sufficient(self):
        """Test that sufficient disk space passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = MockConverter(output_dir=tmpdir)
            validator = PreflightValidator(converter)
            # Mock model size to be small
            validator._model_size_gb = 0.001  # 1 MB
            result = validator._check_disk_space()
            self.assertTrue(result.passed)
            self.assertEqual(result.severity, Severity.INFO)

    @patch("shutil.disk_usage")
    def test_disk_space_insufficient(self, mock_disk_usage):
        """Test that insufficient disk space fails."""
        # Return very low free space
        mock_disk_usage.return_value = MagicMock(free=1024 * 1024)  # 1 MB
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = MockConverter(output_dir=tmpdir)
            validator = PreflightValidator(converter)
            validator._model_size_gb = 10  # 10 GB model needs 40 GB
            result = validator._check_disk_space()
            self.assertFalse(result.passed)
            self.assertEqual(result.severity, Severity.ERROR)
            self.assertIn("Insufficient disk space", result.message)

    @patch("shutil.disk_usage")
    def test_disk_space_warning(self, mock_disk_usage):
        """Test that low disk space generates warning."""
        # Return space between required and warning threshold
        mock_disk_usage.return_value = MagicMock(free=50 * 1024**3)  # 50 GB
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = MockConverter(output_dir=tmpdir)
            validator = PreflightValidator(converter)
            validator._model_size_gb = 10  # 10 GB model, needs 40 GB, warns at 60 GB
            result = validator._check_disk_space()
            self.assertTrue(result.passed)
            self.assertEqual(result.severity, Severity.WARNING)


class TestPreflightValidatorMemory(unittest.TestCase):
    """Tests for memory validation."""

    @patch("psutil.virtual_memory")
    def test_memory_sufficient(self, mock_memory):
        """Test that sufficient memory passes."""
        mock_memory.return_value = MagicMock(
            available=64 * 1024**3,  # 64 GB
            total=128 * 1024**3,
        )
        converter = MockConverter()
        validator = PreflightValidator(converter)
        validator._model_size_gb = 10  # 10 GB model needs 20 GB RAM
        result = validator._check_memory()
        self.assertTrue(result.passed)
        self.assertEqual(result.severity, Severity.INFO)

    @patch("psutil.virtual_memory")
    def test_memory_insufficient_warning(self, mock_memory):
        """Test that insufficient memory generates warning (not error)."""
        mock_memory.return_value = MagicMock(
            available=8 * 1024**3,  # 8 GB
            total=16 * 1024**3,
        )
        converter = MockConverter()
        validator = PreflightValidator(converter)
        validator._model_size_gb = 10  # 10 GB model needs 20 GB RAM
        result = validator._check_memory()
        self.assertTrue(result.passed)  # Warning, not failure
        self.assertEqual(result.severity, Severity.WARNING)
        self.assertIn("Low memory", result.message)

    def test_memory_check_without_psutil(self):
        """Test that missing psutil is handled gracefully."""
        converter = MockConverter()
        validator = PreflightValidator(converter)

        with patch.dict("sys.modules", {"psutil": None}):
            # Force reimport failure
            with patch("builtins.__import__", side_effect=ImportError):
                result = validator._check_memory()
                self.assertTrue(result.passed)
                self.assertIn("psutil", result.message.lower())


class TestPreflightValidatorModelAccess(unittest.TestCase):
    """Tests for model access validation."""

    def test_local_file_exists(self):
        """Test that existing local file passes."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            converter = MockConverter(model_id=f.name)
            validator = PreflightValidator(converter)
            result = validator._check_model_access()
            self.assertTrue(result.passed)
            self.assertIn("Local model file found", result.message)

    def test_local_directory_exists(self):
        """Test that existing local directory passes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = MockConverter(model_id=tmpdir)
            validator = PreflightValidator(converter)
            result = validator._check_model_access()
            self.assertTrue(result.passed)
            self.assertIn("Local model directory found", result.message)

    def test_valid_hf_repo_format(self):
        """Test that valid HuggingFace repo format passes."""
        converter = MockConverter(model_id="black-forest-labs/FLUX.1-schnell")
        validator = PreflightValidator(converter)
        result = validator._check_model_access()
        self.assertTrue(result.passed)
        self.assertIn("HuggingFace repo format valid", result.message)

    def test_invalid_model_id(self):
        """Test that invalid model ID fails."""
        converter = MockConverter(model_id="not-a-valid-path-or-repo")
        validator = PreflightValidator(converter)
        result = validator._check_model_access()
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, Severity.ERROR)
        self.assertIn("Model not found", result.message)

    def test_local_path_not_exists(self):
        """Test that non-existent local path fails."""
        converter = MockConverter(model_id="/path/that/does/not/exist.safetensors")
        validator = PreflightValidator(converter)
        result = validator._check_model_access()
        self.assertFalse(result.passed)


class TestPreflightValidatorDependencies(unittest.TestCase):
    """Tests for dependency validation."""

    def test_dependencies_available(self):
        """Test that available dependencies pass."""
        converter = MockConverter()
        validator = PreflightValidator(converter)
        result = validator._check_dependencies()
        self.assertTrue(result.passed)
        self.assertIn("All required dependencies installed", result.message)

    def test_missing_coremltools(self):
        """Test that missing coremltools fails."""
        converter = MockConverter()
        validator = PreflightValidator(converter)

        with patch.dict("sys.modules", {"coremltools": None}):
            with patch("builtins.__import__") as mock_import:

                def import_side_effect(name, *args, **kwargs):
                    if name == "coremltools":
                        raise ImportError("No module named 'coremltools'")
                    return MagicMock()

                mock_import.side_effect = import_side_effect
                result = validator._check_dependencies()
                self.assertFalse(result.passed)
                self.assertEqual(result.severity, Severity.ERROR)
                self.assertIn("coremltools", result.message)


class TestPreflightValidatorModelSize(unittest.TestCase):
    """Tests for model size calculation."""

    def test_local_file_size(self):
        """Test calculating size of local file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 1024 * 1024)  # 1 MB
            f.flush()
            try:
                converter = MockConverter(model_id=f.name)
                validator = PreflightValidator(converter)
                size = validator._get_model_size_gb()
                self.assertAlmostEqual(size, 1 / 1024, places=5)  # ~0.001 GB
            finally:
                os.unlink(f.name)

    def test_local_directory_size(self):
        """Test calculating size of local directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            for i in range(3):
                with open(os.path.join(tmpdir, f"file{i}.bin"), "wb") as f:
                    f.write(b"x" * 1024 * 1024)  # 1 MB each

            converter = MockConverter(model_id=tmpdir)
            validator = PreflightValidator(converter)
            size = validator._get_model_size_gb()
            self.assertAlmostEqual(size, 3 / 1024, places=5)  # ~0.003 GB

    def test_fallback_size_estimation(self):
        """Test fallback size estimation by model type."""
        converter = MockConverter(model_id="org/repo")
        converter.model_name = "flux"
        validator = PreflightValidator(converter)

        # Mock HF API to fail
        with patch("alloy.validation.PreflightValidator._get_model_size_gb") as mock:
            mock.return_value = None
            size = validator._estimate_size_by_type()
            self.assertEqual(size, 24)  # Flux default

    def test_disk_requirement_calculation(self):
        """Test disk requirement is 4x model size."""
        converter = MockConverter()
        validator = PreflightValidator(converter)
        validator._model_size_gb = 10
        required = validator._estimate_disk_requirement()
        self.assertEqual(required, 40)

    def test_ram_requirement_calculation(self):
        """Test RAM requirement is 2x model size."""
        converter = MockConverter()
        validator = PreflightValidator(converter)
        validator._model_size_gb = 10
        required = validator._estimate_ram_requirement()
        self.assertEqual(required, 20)


class TestPreflightValidatorValidateAll(unittest.TestCase):
    """Tests for validate_all method."""

    def test_validate_all_returns_results(self):
        """Test that validate_all returns list of results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, "model.safetensors")
            with open(model_file, "wb") as f:
                f.write(b"x" * 1024)

            converter = MockConverter(
                model_id=model_file,
                output_dir=os.path.join(tmpdir, "output"),
            )
            validator = PreflightValidator(converter)
            results = validator.validate_all()

            self.assertEqual(len(results), 5)
            self.assertTrue(all(isinstance(r, ValidationResult) for r in results))


class TestRunPreflightValidation(unittest.TestCase):
    """Tests for run_preflight_validation function."""

    def test_passes_with_valid_config(self):
        """Test that valid configuration passes without exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, "model.safetensors")
            with open(model_file, "wb") as f:
                f.write(b"x" * 1024)

            converter = MockConverter(
                model_id=model_file,
                output_dir=os.path.join(tmpdir, "output"),
            )
            # Should not raise
            run_preflight_validation(converter)

    def test_raises_validation_error_on_failure(self):
        """Test that validation error is raised on failure."""
        converter = MockConverter(
            model_id="/nonexistent/model",
            output_dir="/nonexistent/output",
        )
        with self.assertRaises((ValidationError, DependencyError)):
            run_preflight_validation(converter)

    @patch.object(PreflightValidator, "_check_dependencies")
    def test_raises_dependency_error(self, mock_check):
        """Test that DependencyError is raised for missing dependencies."""
        mock_check.return_value = ValidationResult(
            passed=False,
            message="Missing required packages: coremltools",
            severity=Severity.ERROR,
            suggestion="Install with: pip install coremltools",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, "model.safetensors")
            with open(model_file, "wb") as f:
                f.write(b"x" * 1024)

            converter = MockConverter(
                model_id=model_file,
                output_dir=os.path.join(tmpdir, "output"),
            )
            with self.assertRaises(DependencyError):
                run_preflight_validation(converter)


if __name__ == "__main__":
    unittest.main()
