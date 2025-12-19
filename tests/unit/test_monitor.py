"""Tests for alloy.monitor module."""

import pytest
from unittest.mock import patch, MagicMock

from alloy.monitor import (
    MemoryStatus,
    get_memory_status,
    estimate_memory_requirement,
    check_memory_warning,
    MODEL_MEMORY_REQUIREMENTS,
    QUANTIZATION_FACTORS,
)


class TestMemoryStatus:
    """Tests for MemoryStatus dataclass."""

    def test_memory_status_fields(self):
        """Test MemoryStatus has all expected fields."""
        status = MemoryStatus(
            used_gb=16.0,
            available_gb=48.0,
            total_gb=64.0,
            percent_used=25.0,
        )
        assert status.used_gb == 16.0
        assert status.available_gb == 48.0
        assert status.total_gb == 64.0
        assert status.percent_used == 25.0


class TestGetMemoryStatus:
    """Tests for get_memory_status function."""

    @patch("alloy.monitor.PSUTIL_AVAILABLE", True)
    @patch("alloy.monitor.psutil")
    def test_returns_memory_status_when_psutil_available(self, mock_psutil):
        """Test that get_memory_status returns MemoryStatus when psutil is available."""
        mock_mem = MagicMock()
        mock_mem.used = 16 * (1024**3)  # 16 GB
        mock_mem.available = 48 * (1024**3)  # 48 GB
        mock_mem.total = 64 * (1024**3)  # 64 GB
        mock_mem.percent = 25.0
        mock_psutil.virtual_memory.return_value = mock_mem

        status = get_memory_status()

        assert status is not None
        assert abs(status.used_gb - 16.0) < 0.01
        assert abs(status.available_gb - 48.0) < 0.01
        assert abs(status.total_gb - 64.0) < 0.01
        assert status.percent_used == 25.0

    @patch("alloy.monitor.PSUTIL_AVAILABLE", False)
    def test_returns_none_when_psutil_unavailable(self):
        """Test that get_memory_status returns None when psutil is not available."""
        status = get_memory_status()
        assert status is None


class TestEstimateMemoryRequirement:
    """Tests for estimate_memory_requirement function."""

    def test_flux_base_requirement(self):
        """Test Flux base memory requirement."""
        result = estimate_memory_requirement("flux", None)
        assert result == MODEL_MEMORY_REQUIREMENTS["flux"]

    def test_flux_int4_requirement(self):
        """Test Flux with int4 quantization has reduced requirement."""
        result = estimate_memory_requirement("flux", "int4")
        expected = MODEL_MEMORY_REQUIREMENTS["flux"] * QUANTIZATION_FACTORS["int4"]
        assert result == expected

    def test_unknown_model_defaults_to_30(self):
        """Test unknown model type defaults to 30 GB."""
        result = estimate_memory_requirement("unknown_model", None)
        assert result == 30

    def test_case_insensitive_model_lookup(self):
        """Test model lookup is case insensitive."""
        result_lower = estimate_memory_requirement("flux", None)
        result_upper = estimate_memory_requirement("FLUX", None)
        assert result_lower == result_upper


class TestCheckMemoryWarning:
    """Tests for check_memory_warning function."""

    @patch("alloy.monitor.get_memory_status")
    def test_returns_warning_when_memory_low(self, mock_get_status):
        """Test returns warning message when available memory is low."""
        mock_get_status.return_value = MemoryStatus(
            used_gb=60.0,
            available_gb=4.0,
            total_gb=64.0,
            percent_used=93.75,
        )

        warning = check_memory_warning(48.0)

        assert warning is not None
        assert "Low memory warning" in warning
        assert "4.0 GB available" in warning

    @patch("alloy.monitor.get_memory_status")
    def test_returns_none_when_memory_sufficient(self, mock_get_status):
        """Test returns None when available memory is sufficient."""
        mock_get_status.return_value = MemoryStatus(
            used_gb=16.0,
            available_gb=48.0,
            total_gb=64.0,
            percent_used=25.0,
        )

        warning = check_memory_warning(30.0)

        assert warning is None

    @patch("alloy.monitor.get_memory_status")
    def test_returns_none_when_psutil_unavailable(self, mock_get_status):
        """Test returns None when memory status is unavailable."""
        mock_get_status.return_value = None

        warning = check_memory_warning(48.0)

        assert warning is None


class TestModelRequirements:
    """Tests for model requirement constants."""

    def test_all_expected_models_present(self):
        """Test all expected model types have memory requirements."""
        expected_models = ["flux", "hunyuan", "ltx", "wan", "lumina", "sd"]
        for model in expected_models:
            assert model in MODEL_MEMORY_REQUIREMENTS

    def test_quantization_factors_present(self):
        """Test quantization factors for expected types."""
        expected_quants = ["int4", "int8", "float16", None]
        for quant in expected_quants:
            assert quant in QUANTIZATION_FACTORS
