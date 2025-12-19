"""Tests for the ModelConverter and TwoPhaseConverter base classes."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from alloy.converters.base import ModelConverter


class ConcreteConverter(ModelConverter):
    """Concrete implementation for testing abstract base class."""

    def convert(self):
        pass


class TestDownloadSourceWeights(unittest.TestCase):
    """Tests for ModelConverter.download_source_weights()."""

    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.converter = ConcreteConverter("org/model", self.output_dir, "float16")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_returns_original_for_local_path_without_slash(self):
        """Test that local paths without '/' are returned unchanged."""
        result = self.converter.download_source_weights(
            "local_model", self.output_dir
        )
        self.assertEqual(result, "local_model")

    def test_returns_original_for_single_file(self):
        """Test that single file paths are returned unchanged."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            try:
                result = self.converter.download_source_weights(
                    f.name, self.output_dir
                )
                self.assertEqual(result, f.name)
            finally:
                os.unlink(f.name)

    @patch("huggingface_hub.snapshot_download")
    def test_download_failure_returns_original_repo_id(self, mock_download):
        """Test that download failure falls back to original repo_id."""
        mock_download.side_effect = Exception("Network error")
        logger_messages = []

        result = self.converter.download_source_weights(
            "org/model",
            self.output_dir,
            logger_fn=logger_messages.append
        )

        self.assertEqual(result, "org/model")
        self.assertTrue(any("Failed to download" in msg for msg in logger_messages))

    @patch("huggingface_hub.snapshot_download")
    def test_successful_download_returns_source_dir(self, mock_download):
        """Test that successful download returns source directory path."""
        mock_download.return_value = None  # Return value is not used by the implementation
        logger_messages = []

        result = self.converter.download_source_weights(
            "org/model",
            self.output_dir,
            logger_fn=logger_messages.append
        )

        expected_source_dir = os.path.join(self.output_dir, "source")
        self.assertEqual(result, expected_source_dir)
        self.assertTrue(any("Originals saved to" in msg for msg in logger_messages))

    @patch("huggingface_hub.snapshot_download")
    def test_uses_default_patterns(self, mock_download):
        """Test that default allow/ignore patterns are used."""
        mock_download.return_value = None

        self.converter.download_source_weights("org/model", self.output_dir)

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        self.assertIn("transformer/*", call_kwargs["allow_patterns"])
        self.assertIn("*.msgpack", call_kwargs["ignore_patterns"])

    @patch("huggingface_hub.snapshot_download")
    def test_uses_custom_patterns(self, mock_download):
        """Test that custom allow/ignore patterns override defaults."""
        mock_download.return_value = None

        self.converter.download_source_weights(
            "org/model",
            self.output_dir,
            allow_patterns=["*.bin"],
            ignore_patterns=["*.txt"]
        )

        call_kwargs = mock_download.call_args[1]
        self.assertEqual(call_kwargs["allow_patterns"], ["*.bin"])
        self.assertEqual(call_kwargs["ignore_patterns"], ["*.txt"])


class TestValidateOrReconvert(unittest.TestCase):
    """Tests for ModelConverter.validate_or_reconvert()."""

    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.converter = ConcreteConverter("org/model", self.output_dir, "float16")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_nonexistent_path_calls_convert_fn(self):
        """Test that missing file triggers conversion."""
        convert_fn = MagicMock()
        nonexistent_path = os.path.join(self.output_dir, "nonexistent.mlpackage")

        result = self.converter.validate_or_reconvert(nonexistent_path, convert_fn)

        self.assertTrue(result)
        convert_fn.assert_called_once()

    @patch("alloy.converters.base.ct.models.MLModel")
    def test_valid_intermediate_skips_conversion(self, mock_mlmodel):
        """Test that valid intermediate skips conversion and returns False."""
        # Create a dummy intermediate directory
        intermediate_path = os.path.join(self.output_dir, "valid.mlpackage")
        os.makedirs(intermediate_path)

        mock_mlmodel.return_value = MagicMock()  # Valid model
        convert_fn = MagicMock()
        logger_messages = []

        result = self.converter.validate_or_reconvert(
            intermediate_path, convert_fn, logger_fn=logger_messages.append
        )

        self.assertFalse(result)
        convert_fn.assert_not_called()
        self.assertTrue(any("Found valid intermediate" in msg for msg in logger_messages))

    @patch("alloy.converters.base.shutil.rmtree")
    @patch("alloy.converters.base.ct.models.MLModel")
    def test_invalid_intermediate_reconverts(self, mock_mlmodel, mock_rmtree):
        """Test that invalid intermediate triggers cleanup and conversion."""
        # Create a dummy intermediate directory
        intermediate_path = os.path.join(self.output_dir, "invalid.mlpackage")
        os.makedirs(intermediate_path)

        mock_mlmodel.side_effect = Exception("Invalid MLModel")
        convert_fn = MagicMock()
        logger_messages = []

        result = self.converter.validate_or_reconvert(
            intermediate_path, convert_fn, logger_fn=logger_messages.append
        )

        self.assertTrue(result)
        mock_rmtree.assert_called_once_with(intermediate_path)
        convert_fn.assert_called_once()
        self.assertTrue(any("Invalid intermediate found" in msg for msg in logger_messages))

    @patch("alloy.converters.base.ct.models.MLModel")
    def test_uses_cpu_only_for_validation(self, mock_mlmodel):
        """Test that validation uses CPU_ONLY compute units."""
        intermediate_path = os.path.join(self.output_dir, "test.mlpackage")
        os.makedirs(intermediate_path)

        mock_mlmodel.return_value = MagicMock()

        self.converter.validate_or_reconvert(intermediate_path, MagicMock())

        mock_mlmodel.assert_called_once()
        call_kwargs = mock_mlmodel.call_args[1]
        # Check compute_units has CPU_ONLY value (3)
        self.assertEqual(call_kwargs["compute_units"].value, 3)


class TestModelConverterInit(unittest.TestCase):
    """Tests for ModelConverter initialization."""

    def test_init_stores_parameters(self):
        """Test that init stores all parameters correctly."""
        converter = ConcreteConverter("org/model", "/output", "int8")

        self.assertEqual(converter.model_id, "org/model")
        self.assertEqual(converter.output_dir, "/output")
        self.assertEqual(converter.quantization, "int8")

    def test_default_quantization(self):
        """Test that default quantization is float16."""
        converter = ConcreteConverter("org/model", "/output")

        self.assertEqual(converter.quantization, "float16")


if __name__ == "__main__":
    unittest.main()
