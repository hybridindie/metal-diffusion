"""Tests for the BaseCoreMLRunner base class and runner utilities."""

import numpy as np
import torch
import unittest
from unittest.mock import MagicMock, patch

from alloy.runners.core import BaseCoreMLRunner
from alloy.runners.utils import apply_classifier_free_guidance


class ConcreteRunner(BaseCoreMLRunner):
    """Concrete implementation for testing abstract base class."""

    @property
    def model_name(self) -> str:
        return "TestModel"

    @property
    def transformer_filename(self) -> str:
        return "Test_Transformer.mlpackage"

    @property
    def pipeline_class(self):
        return MagicMock

    @property
    def default_model_id(self) -> str:
        return "test/model"

    def _load_pipeline(self) -> None:
        self.pipe = MagicMock()

    def _load_coreml_models(self) -> None:
        self.coreml_transformer = MagicMock()

    def generate(self, prompt: str, output_path: str, **kwargs) -> None:
        pass


class TestDeviceDetection(unittest.TestCase):
    """Tests for BaseCoreMLRunner._detect_device()."""

    @patch("torch.backends.mps.is_available", return_value=True)
    def test_detects_mps_when_available(self, mock_mps):
        """Test that MPS is detected when available."""
        device = BaseCoreMLRunner._detect_device()
        self.assertEqual(device, "mps")

    @patch("torch.backends.mps.is_available", return_value=False)
    def test_falls_back_to_cpu(self, mock_mps):
        """Test that CPU is used when MPS is unavailable."""
        device = BaseCoreMLRunner._detect_device()
        self.assertEqual(device, "cpu")


class TestTensorConversion(unittest.TestCase):
    """Tests for to_numpy() and from_numpy() utilities."""

    def setUp(self):
        """Create a runner instance with mocked initialization."""
        with patch.object(ConcreteRunner, "__init__", lambda self: None):
            self.runner = ConcreteRunner.__new__(ConcreteRunner)
            self.runner.device = "cpu"

    def test_to_numpy_float32(self):
        """Test tensor to numpy conversion with float32."""
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        result = self.runner.to_numpy(tensor)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_to_numpy_int32(self):
        """Test tensor to numpy conversion with int32."""
        tensor = torch.tensor([1, 2, 3])
        result = self.runner.to_numpy(tensor, dtype=np.int32)

        self.assertEqual(result.dtype, np.int32)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_to_numpy_preserves_shape(self):
        """Test that tensor shape is preserved during conversion."""
        tensor = torch.randn(2, 3, 4)
        result = self.runner.to_numpy(tensor)

        self.assertEqual(result.shape, (2, 3, 4))

    def test_from_numpy_basic(self):
        """Test numpy to tensor conversion."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self.runner.from_numpy(array)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.device.type, "cpu")
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_from_numpy_with_dtype(self):
        """Test numpy to tensor conversion with explicit dtype."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = self.runner.from_numpy(array, dtype=torch.float16)

        self.assertEqual(result.dtype, torch.float16)

    def test_from_numpy_preserves_shape(self):
        """Test that array shape is preserved during conversion."""
        array = np.random.randn(2, 3, 4).astype(np.float32)
        result = self.runner.from_numpy(array)

        self.assertEqual(result.shape, torch.Size([2, 3, 4]))


class TestPredictCoreML(unittest.TestCase):
    """Tests for predict_coreml() method."""

    def setUp(self):
        """Create a runner instance with mocked initialization."""
        with patch.object(ConcreteRunner, "__init__", lambda self: None):
            self.runner = ConcreteRunner.__new__(ConcreteRunner)
            self.runner.device = "cpu"
            # output_key is a property with default "sample", no need to set

    def test_predict_coreml_default_key(self):
        """Test CoreML prediction with default output key."""
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "sample": np.array([1.0, 2.0, 3.0], dtype=np.float32)
        }

        result = self.runner.predict_coreml(mock_model, {"input": np.zeros(3)})

        self.assertIsInstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_predict_coreml_custom_key(self):
        """Test CoreML prediction with custom output key."""
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "hidden_states": np.array([4.0, 5.0, 6.0], dtype=np.float32)
        }

        result = self.runner.predict_coreml(
            mock_model, {"input": np.zeros(3)}, output_key="hidden_states"
        )

        torch.testing.assert_close(result, torch.tensor([4.0, 5.0, 6.0]))


class TestClassifierFreeGuidance(unittest.TestCase):
    """Tests for apply_classifier_free_guidance utility."""

    def test_basic_cfg(self):
        """Test basic CFG computation."""
        noise_uncond = torch.tensor([1.0, 1.0, 1.0])
        noise_text = torch.tensor([3.0, 3.0, 3.0])
        guidance_scale = 2.0

        result = apply_classifier_free_guidance(noise_uncond, noise_text, guidance_scale)

        # uncond + scale * (text - uncond) = 1 + 2 * (3 - 1) = 1 + 4 = 5
        expected = torch.tensor([5.0, 5.0, 5.0])
        torch.testing.assert_close(result, expected)

    def test_cfg_scale_zero(self):
        """Test CFG with scale 0 returns unconditional."""
        noise_uncond = torch.tensor([1.0, 2.0])
        noise_text = torch.tensor([5.0, 6.0])

        result = apply_classifier_free_guidance(noise_uncond, noise_text, 0.0)

        torch.testing.assert_close(result, noise_uncond)

    def test_cfg_scale_one(self):
        """Test CFG with scale 1 returns text-conditioned."""
        noise_uncond = torch.tensor([1.0, 2.0])
        noise_text = torch.tensor([5.0, 6.0])

        result = apply_classifier_free_guidance(noise_uncond, noise_text, 1.0)

        torch.testing.assert_close(result, noise_text)

    def test_cfg_multidimensional(self):
        """Test CFG works with multi-dimensional tensors."""
        noise_uncond = torch.randn(1, 4, 32, 32)
        noise_text = torch.randn(1, 4, 32, 32)
        guidance_scale = 7.5

        result = apply_classifier_free_guidance(noise_uncond, noise_text, guidance_scale)

        self.assertEqual(result.shape, (1, 4, 32, 32))


class TestRunnerProperties(unittest.TestCase):
    """Tests for BaseCoreMLRunner default properties."""

    def setUp(self):
        """Create a runner instance with mocked initialization."""
        with patch.object(ConcreteRunner, "__init__", lambda self: None):
            self.runner = ConcreteRunner.__new__(ConcreteRunner)

    def test_supports_single_file_default(self):
        """Test default supports_single_file is True."""
        self.assertTrue(self.runner.supports_single_file)

    def test_default_dtype(self):
        """Test default dtype is float16."""
        self.assertEqual(self.runner.default_dtype, torch.float16)

    def test_output_key_default(self):
        """Test default output key is 'sample'."""
        self.assertEqual(self.runner.output_key, "sample")


class TestRunnerInit(unittest.TestCase):
    """Tests for BaseCoreMLRunner initialization."""

    @patch.object(ConcreteRunner, "_load_coreml_models")
    @patch.object(ConcreteRunner, "_load_pipeline")
    @patch("alloy.runners.core.BaseCoreMLRunner._detect_device", return_value="cpu")
    def test_init_stores_parameters(self, mock_device, mock_pipeline, mock_coreml):
        """Test that init stores all parameters correctly."""
        runner = ConcreteRunner("/path/to/models", "custom/model", "CPU_ONLY")

        self.assertEqual(runner.model_dir, "/path/to/models")
        self.assertEqual(runner.model_id, "custom/model")
        self.assertEqual(runner.compute_unit, "CPU_ONLY")
        self.assertEqual(runner.device, "cpu")

    @patch.object(ConcreteRunner, "_load_coreml_models")
    @patch.object(ConcreteRunner, "_load_pipeline")
    @patch("alloy.runners.core.BaseCoreMLRunner._detect_device", return_value="mps")
    def test_init_uses_default_model_id(self, mock_device, mock_pipeline, mock_coreml):
        """Test that init uses default_model_id when none provided."""
        runner = ConcreteRunner("/path/to/models")

        self.assertEqual(runner.model_id, "test/model")

    @patch.object(ConcreteRunner, "_load_coreml_models")
    @patch.object(ConcreteRunner, "_load_pipeline")
    @patch("alloy.runners.core.BaseCoreMLRunner._detect_device", return_value="cpu")
    def test_init_calls_load_methods(self, mock_device, mock_pipeline, mock_coreml):
        """Test that init calls pipeline and CoreML loading methods."""
        ConcreteRunner("/path/to/models")

        mock_pipeline.assert_called_once()
        mock_coreml.assert_called_once()


if __name__ == "__main__":
    unittest.main()
