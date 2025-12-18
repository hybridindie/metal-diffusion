import unittest
from unittest.mock import patch, MagicMock, Mock
import torch
import tempfile
import os

from alloy.converters.workers import (
    FluxInputShapes,
    create_flux_dummy_inputs,
    quantize_and_save,
    FluxPart1Wrapper,
    FluxPart2Wrapper,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_TEXT_LEN,
    DEFAULT_BATCH_SIZE,
)


class TestFluxInputShapes(unittest.TestCase):
    """Tests for FluxInputShapes dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        shapes = FluxInputShapes()
        self.assertEqual(shapes.batch_size, DEFAULT_BATCH_SIZE)
        self.assertEqual(shapes.height, DEFAULT_HEIGHT)
        self.assertEqual(shapes.width, DEFAULT_WIDTH)
        self.assertEqual(shapes.text_len, DEFAULT_TEXT_LEN)

    def test_custom_values(self):
        """Test custom values can be set."""
        shapes = FluxInputShapes(batch_size=2, height=128, width=128, text_len=512)
        self.assertEqual(shapes.batch_size, 2)
        self.assertEqual(shapes.height, 128)
        self.assertEqual(shapes.width, 128)
        self.assertEqual(shapes.text_len, 512)

    def test_sequence_length_calculation(self):
        """Test sequence_length property calculates correctly."""
        shapes = FluxInputShapes(height=64, width=64)
        # sequence_length = (height // 2) * (width // 2) = 32 * 32 = 1024
        self.assertEqual(shapes.sequence_length, 1024)

        shapes2 = FluxInputShapes(height=128, width=64)
        # sequence_length = (128 // 2) * (64 // 2) = 64 * 32 = 2048
        self.assertEqual(shapes2.sequence_length, 2048)


class TestCreateFluxDummyInputs(unittest.TestCase):
    """Tests for create_flux_dummy_inputs helper function."""

    def setUp(self):
        """Create mock transformer with config."""
        self.mock_transformer = MagicMock()
        self.mock_transformer.config.in_channels = 64
        self.mock_transformer.config.joint_attention_dim = 4096
        self.mock_transformer.config.pooled_projection_dim = 768
        self.mock_transformer.config.num_attention_heads = 24
        self.mock_transformer.config.attention_head_dim = 128

    def test_part1_inputs_use_in_channels(self):
        """Test Part 1 uses in_channels for hidden dimension."""
        shapes = FluxInputShapes()
        inputs, names = create_flux_dummy_inputs(
            self.mock_transformer, shapes, use_hidden_size=False
        )

        # Check hidden_states shape uses in_channels (64)
        hidden_states = inputs[0]
        self.assertEqual(hidden_states.shape[2], 64)

        # Check encoder_hidden_states uses joint_attention_dim
        encoder_hidden_states = inputs[1]
        self.assertEqual(encoder_hidden_states.shape[2], 4096)

    def test_part2_inputs_use_hidden_size(self):
        """Test Part 2 uses hidden_size for hidden dimension."""
        shapes = FluxInputShapes()
        inputs, names = create_flux_dummy_inputs(
            self.mock_transformer, shapes, use_hidden_size=True
        )

        # hidden_size = num_attention_heads * attention_head_dim = 24 * 128 = 3072
        hidden_states = inputs[0]
        self.assertEqual(hidden_states.shape[2], 3072)

        # Part 2 encoder_hidden_states also uses hidden_size
        encoder_hidden_states = inputs[1]
        self.assertEqual(encoder_hidden_states.shape[2], 3072)

    def test_returns_correct_input_names(self):
        """Test correct input names are returned."""
        shapes = FluxInputShapes()
        inputs, names = create_flux_dummy_inputs(
            self.mock_transformer, shapes, use_hidden_size=False
        )

        expected_names = [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
        ]
        self.assertEqual(names, expected_names)

    def test_returns_seven_inputs(self):
        """Test that seven input tensors are returned."""
        shapes = FluxInputShapes()
        inputs, names = create_flux_dummy_inputs(
            self.mock_transformer, shapes, use_hidden_size=False
        )

        self.assertEqual(len(inputs), 7)
        self.assertEqual(len(names), 7)

    def test_all_inputs_are_float_tensors(self):
        """Test all returned inputs are float tensors."""
        shapes = FluxInputShapes()
        inputs, _ = create_flux_dummy_inputs(
            self.mock_transformer, shapes, use_hidden_size=False
        )

        for inp in inputs:
            self.assertIsInstance(inp, torch.Tensor)
            self.assertEqual(inp.dtype, torch.float32)

    def test_custom_shapes_reflected_in_outputs(self):
        """Test custom shapes are used in output tensors."""
        shapes = FluxInputShapes(batch_size=2, height=128, width=64, text_len=512)
        inputs, _ = create_flux_dummy_inputs(
            self.mock_transformer, shapes, use_hidden_size=False
        )

        hidden_states = inputs[0]
        self.assertEqual(hidden_states.shape[0], 2)  # batch_size
        # sequence_length = (128 // 2) * (64 // 2) = 2048
        self.assertEqual(hidden_states.shape[1], 2048)

        encoder_hidden_states = inputs[1]
        self.assertEqual(encoder_hidden_states.shape[1], 512)  # text_len


class TestQuantizeAndSave(unittest.TestCase):
    """Tests for quantize_and_save helper function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_no_quantization_saves_directly(self):
        """Test model is saved directly when no quantization is specified."""
        mock_model = MagicMock()
        output_path = os.path.join(self.temp_dir, "output.mlpackage")

        quantize_and_save(mock_model, output_path, None, None, "test")

        mock_model.save.assert_called_once_with(output_path)

    @patch("alloy.converters.workers.safe_quantize_model")
    def test_quantization_with_intermediates_dir(self, mock_quantize):
        """Test quantization uses intermediates directory when provided."""
        mock_model = MagicMock()
        mock_quantized = MagicMock()
        mock_quantize.return_value = mock_quantized

        output_path = os.path.join(self.temp_dir, "output.mlpackage")
        intermediates_dir = os.path.join(self.temp_dir, "intermediates")
        os.makedirs(intermediates_dir)

        quantize_and_save(mock_model, output_path, "int4", intermediates_dir, "part1")

        # Verify quantization was called
        mock_quantize.assert_called_once()
        # Verify final save
        mock_quantized.save.assert_called_with(output_path)

    @patch("alloy.converters.workers.safe_quantize_model")
    def test_quantization_without_intermediates_uses_temp(self, mock_quantize):
        """Test quantization uses temp directory when no intermediates_dir."""
        mock_model = MagicMock()
        mock_quantized = MagicMock()
        mock_quantize.return_value = mock_quantized

        output_path = os.path.join(self.temp_dir, "output.mlpackage")

        quantize_and_save(mock_model, output_path, "int8", None, "part2")

        mock_quantize.assert_called_once()
        mock_quantized.save.assert_called_with(output_path)


class TestFluxPart1Wrapper(unittest.TestCase):
    """Tests for FluxPart1Wrapper."""

    def test_compute_time_embedding_with_guidance(self):
        """Test time embedding computation with guidance."""
        mock_model = MagicMock()
        mock_model.time_text_embed.return_value = torch.tensor([1.0])

        wrapper = FluxPart1Wrapper(mock_model)

        timestep = torch.tensor([1.0])
        guidance = torch.tensor([7.5])
        pooled = torch.randn(1, 768)

        result = wrapper._compute_time_embedding(timestep, guidance, pooled)

        mock_model.time_text_embed.assert_called_once_with(timestep, guidance, pooled)

    def test_compute_time_embedding_without_guidance(self):
        """Test time embedding computation without guidance."""
        mock_model = MagicMock()
        mock_model.time_text_embed.return_value = torch.tensor([1.0])

        wrapper = FluxPart1Wrapper(mock_model)

        timestep = torch.tensor([1.0])
        pooled = torch.randn(1, 768)

        result = wrapper._compute_time_embedding(timestep, None, pooled)

        mock_model.time_text_embed.assert_called_once_with(timestep, pooled)

    def test_squeeze_ids_3d_input(self):
        """Test squeeze_ids squeezes 3D tensor."""
        mock_model = MagicMock()
        wrapper = FluxPart1Wrapper(mock_model)

        ids_3d = torch.randn(1, 100, 3)
        result = wrapper._squeeze_ids(ids_3d)

        self.assertEqual(result.shape, (100, 3))

    def test_squeeze_ids_2d_input(self):
        """Test squeeze_ids keeps 2D tensor unchanged."""
        mock_model = MagicMock()
        wrapper = FluxPart1Wrapper(mock_model)

        ids_2d = torch.randn(100, 3)
        result = wrapper._squeeze_ids(ids_2d)

        self.assertEqual(result.shape, (100, 3))
        self.assertTrue(torch.equal(result, ids_2d))


class TestFluxPart2Wrapper(unittest.TestCase):
    """Tests for FluxPart2Wrapper."""

    def test_compute_time_embedding_with_guidance(self):
        """Test time embedding computation with guidance."""
        mock_model = MagicMock()
        mock_model.time_text_embed.return_value = torch.tensor([1.0])

        wrapper = FluxPart2Wrapper(mock_model)

        timestep = torch.tensor([1.0])
        guidance = torch.tensor([7.5])
        pooled = torch.randn(1, 768)

        result = wrapper._compute_time_embedding(timestep, guidance, pooled)

        mock_model.time_text_embed.assert_called_once_with(timestep, guidance, pooled)

    def test_compute_time_embedding_without_guidance(self):
        """Test time embedding computation without guidance."""
        mock_model = MagicMock()
        mock_model.time_text_embed.return_value = torch.tensor([1.0])

        wrapper = FluxPart2Wrapper(mock_model)

        timestep = torch.tensor([1.0])
        pooled = torch.randn(1, 768)

        result = wrapper._compute_time_embedding(timestep, None, pooled)

        mock_model.time_text_embed.assert_called_once_with(timestep, pooled)

    def test_squeeze_ids_3d_input(self):
        """Test squeeze_ids squeezes 3D tensor."""
        mock_model = MagicMock()
        wrapper = FluxPart2Wrapper(mock_model)

        ids_3d = torch.randn(1, 100, 3)
        result = wrapper._squeeze_ids(ids_3d)

        self.assertEqual(result.shape, (100, 3))

    def test_squeeze_ids_2d_input(self):
        """Test squeeze_ids keeps 2D tensor unchanged."""
        mock_model = MagicMock()
        wrapper = FluxPart2Wrapper(mock_model)

        ids_2d = torch.randn(100, 3)
        result = wrapper._squeeze_ids(ids_2d)

        self.assertEqual(result.shape, (100, 3))
        self.assertTrue(torch.equal(result, ids_2d))


if __name__ == "__main__":
    unittest.main()
