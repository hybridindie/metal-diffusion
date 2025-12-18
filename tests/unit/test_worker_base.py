"""Tests for worker base utilities."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from alloy.workers.base import (
    worker_context,
    quantize_and_save,
    load_transformer_with_fallback,
)


class TestWorkerContext(unittest.TestCase):
    """Tests for worker_context context manager."""

    @patch("alloy.workers.base.logger")
    @patch("alloy.workers.base.gc.collect")
    def test_logs_start_and_complete(self, mock_gc, mock_logger):
        """Test that worker_context logs start and complete messages."""
        with worker_context("TestModel", "Part 1"):
            pass

        # Check start and complete messages were logged
        calls = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("Starting" in c and "TestModel" in c for c in calls))
        self.assertTrue(any("Complete" in c for c in calls))

    @patch("alloy.workers.base.logger")
    @patch("alloy.workers.base.gc.collect")
    def test_includes_pid_in_start_message(self, mock_gc, mock_logger):
        """Test that start message includes PID."""
        with worker_context("Flux", "Part 2"):
            pass

        start_call = mock_logger.info.call_args_list[0]
        self.assertIn("PID:", str(start_call))

    @patch("alloy.workers.base.logger")
    @patch("alloy.workers.base.gc.collect")
    def test_calls_gc_collect_on_exit(self, mock_gc, mock_logger):
        """Test that gc.collect is called on context exit."""
        with worker_context("Model", "Part 1"):
            pass

        mock_gc.assert_called()

    @patch("alloy.workers.base.logger")
    @patch("alloy.workers.base.gc.collect")
    def test_calls_gc_collect_on_exception(self, mock_gc, mock_logger):
        """Test that gc.collect is called even when exception occurs."""
        with self.assertRaises(ValueError):
            with worker_context("Model", "Part 1"):
                raise ValueError("Test error")

        mock_gc.assert_called()


class TestQuantizeAndSave(unittest.TestCase):
    """Tests for quantize_and_save function."""

    def test_no_quantization_saves_directly(self):
        """Test that model is saved directly when no quantization."""
        mock_model = MagicMock()
        output_path = "/tmp/test.mlpackage"

        quantize_and_save(mock_model, output_path, None)

        mock_model.save.assert_called_once_with(output_path)

    def test_empty_quantization_saves_directly(self):
        """Test that empty string quantization saves directly."""
        mock_model = MagicMock()
        output_path = "/tmp/test.mlpackage"

        quantize_and_save(mock_model, output_path, "")

        mock_model.save.assert_called_once_with(output_path)

    @patch("alloy.workers.base.safe_quantize_model")
    @patch("alloy.workers.base.logger")
    @patch("alloy.workers.base.gc.collect")
    def test_quantization_with_intermediates_dir(
        self, mock_gc, mock_logger, mock_safe_quantize
    ):
        """Test quantization saves to intermediates dir then quantizes."""
        mock_model = MagicMock()
        mock_quantized = MagicMock()
        mock_safe_quantize.return_value = mock_quantized

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.mlpackage")

            quantize_and_save(
                mock_model, output_path, "int8", intermediates_dir=tmpdir, part_name="part1"
            )

            # Original model should be saved first
            self.assertEqual(mock_model.save.call_count, 1)
            # Quantized model should be saved to output
            mock_quantized.save.assert_called_with(output_path)
            # safe_quantize_model should be called
            mock_safe_quantize.assert_called_once()

    @patch("alloy.workers.base.safe_quantize_model")
    @patch("alloy.workers.base.logger")
    @patch("alloy.workers.base.gc.collect")
    def test_quantization_without_intermediates_uses_temp(
        self, mock_gc, mock_logger, mock_safe_quantize
    ):
        """Test quantization uses temp directory when no intermediates_dir."""
        mock_model = MagicMock()
        mock_quantized = MagicMock()
        mock_safe_quantize.return_value = mock_quantized

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.mlpackage")

            quantize_and_save(mock_model, output_path, "int4", part_name="part2")

            # Original model should be saved first
            self.assertEqual(mock_model.save.call_count, 1)
            # Quantized model should be saved to output
            mock_quantized.save.assert_called_with(output_path)

    @patch("alloy.workers.base.logger")
    def test_logs_quantization_progress(self, mock_logger):
        """Test that quantization logs progress messages."""
        mock_model = MagicMock()
        # Make safe_quantize_model return a mock
        with patch("alloy.workers.base.safe_quantize_model") as mock_safe_quantize:
            mock_safe_quantize.return_value = MagicMock()

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "output.mlpackage")
                quantize_and_save(mock_model, output_path, "int8", part_name="test_part")

        calls = [str(c) for c in mock_logger.debug.call_args_list]
        self.assertTrue(any("Quantizing" in c and "test_part" in c for c in calls))
        self.assertTrue(any("Saving" in c and "test_part" in c for c in calls))


class TestLoadTransformerWithFallback(unittest.TestCase):
    """Tests for load_transformer_with_fallback function."""

    @patch("alloy.workers.base.logger")
    def test_loads_from_subfolder_first(self, mock_logger):
        """Test that it tries subfolder first."""
        mock_class = MagicMock()
        mock_model = MagicMock()
        mock_class.from_pretrained.return_value = mock_model

        result = load_transformer_with_fallback(
            mock_class, "org/model", "float32", subfolder="transformer"
        )

        mock_class.from_pretrained.assert_called_once_with(
            "org/model", subfolder="transformer", torch_dtype="float32"
        )
        self.assertEqual(result, mock_model)

    @patch("alloy.workers.base.logger")
    def test_falls_back_to_root_on_error(self, mock_logger):
        """Test that it falls back to root on subfolder error."""
        mock_class = MagicMock()
        mock_model = MagicMock()

        # First call raises EnvironmentError, second succeeds
        mock_class.from_pretrained.side_effect = [
            EnvironmentError("Not found"),
            mock_model,
        ]

        result = load_transformer_with_fallback(
            mock_class, "org/model", "float32", subfolder="transformer"
        )

        # Should be called twice
        self.assertEqual(mock_class.from_pretrained.call_count, 2)
        # Second call should be without subfolder
        mock_class.from_pretrained.assert_called_with("org/model", torch_dtype="float32")
        self.assertEqual(result, mock_model)

    @patch("alloy.workers.base.logger")
    def test_falls_back_on_os_error(self, mock_logger):
        """Test that it falls back on OSError as well."""
        mock_class = MagicMock()
        mock_model = MagicMock()

        mock_class.from_pretrained.side_effect = [OSError("Not found"), mock_model]

        result = load_transformer_with_fallback(mock_class, "org/model", "float32")

        self.assertEqual(mock_class.from_pretrained.call_count, 2)
        self.assertEqual(result, mock_model)

    @patch("alloy.workers.base.logger")
    def test_logs_fallback_attempt(self, mock_logger):
        """Test that fallback attempt is logged."""
        mock_class = MagicMock()
        mock_class.from_pretrained.side_effect = [
            EnvironmentError("Not found"),
            MagicMock(),
        ]

        load_transformer_with_fallback(
            mock_class, "org/model", "float32", enable_logging=True
        )

        calls = [str(c) for c in mock_logger.debug.call_args_list]
        self.assertTrue(any("Subfolder load failed" in c for c in calls))

    @patch("alloy.workers.base.logger")
    def test_no_logging_when_disabled(self, mock_logger):
        """Test that no logging occurs when enable_logging=False."""
        mock_class = MagicMock()
        mock_class.from_pretrained.return_value = MagicMock()

        load_transformer_with_fallback(
            mock_class, "org/model", "float32", enable_logging=False
        )

        mock_logger.debug.assert_not_called()


if __name__ == "__main__":
    unittest.main()
