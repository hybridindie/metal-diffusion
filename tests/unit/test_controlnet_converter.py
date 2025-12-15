
import unittest
from unittest.mock import MagicMock, patch
import torch
import coremltools as ct
from alloy.converters.controlnet import FluxControlNetConverter, FluxControlNetWrapper, NUM_DOUBLE_BLOCKS, NUM_SINGLE_BLOCKS

class TestFluxControlNetConverter(unittest.TestCase):
    def setUp(self):
        self.converter = FluxControlNetConverter(
            model_id="black-forest-labs/FLUX.1-Canny-dev",
            output_dir="dummy_output",
            quantization="float16"
        )

    @patch("alloy.converters.controlnet.FluxControlNetModel")
    @patch("alloy.converters.controlnet.ct")
    @patch("alloy.converters.controlnet.torch.jit.trace")
    def test_convert_flow(self, mock_trace, mock_ct, mock_model_cls):
        # Mock Model
        mock_model = MagicMock()
        mock_model.config.joint_attention_dim = 64
        mock_model.config.in_channels = 16 # Mock in_channels
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # Mock Wrapper Output
        # Wrapper returns flattened tuple of residuals
        num_residuals = NUM_DOUBLE_BLOCKS + NUM_SINGLE_BLOCKS
        dummy_residuals = tuple([torch.randn(1, 10, 64) for _ in range(num_residuals)])
        
        # Trace should return a mock script module
        mock_traced = MagicMock()
        mock_trace.return_value = mock_traced
        
        # Mocks
        mock_ml_model = MagicMock()
        mock_ct.convert.return_value = mock_ml_model
        
        # Run
        try:
            self.converter.convert()
        except Exception as e:
            self.fail(f"Convert raised exception: {e}")
            
        # Verify
        mock_model_cls.from_pretrained.assert_called_once()
        mock_trace.assert_called_once()
        mock_ct.convert.assert_called_once()
        
        # Check Inputs/Outputs passed to ct.convert
        args, kwargs = mock_ct.convert.call_args
        inputs = kwargs["inputs"]
        outputs = kwargs["outputs"]
        
        self.assertEqual(len(inputs), 7) # control, hidden, enc, time, img, txt, guide
        self.assertEqual(len(outputs), NUM_DOUBLE_BLOCKS + NUM_SINGLE_BLOCKS)
        self.assertEqual(outputs[0].name, "c_double_0")

class TestFluxControlNetWrapper(unittest.TestCase):
    def test_forward_flattening(self):
        # Mock Model returns (block_samples, single_block_samples)
        mock_model = MagicMock()
        
        block_samples = [torch.tensor([1.0]), torch.tensor([2.0])]
        single_samples = [torch.tensor([3.0])]
        
        mock_model.return_value = (block_samples, single_samples)
        
        wrapper = FluxControlNetWrapper(mock_model)
        
        # Dummy inputs
        dummy = torch.randn(1)
        out = wrapper(dummy, dummy, dummy, dummy, dummy, dummy, dummy)
        
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].item(), 1.0)
        self.assertEqual(out[2].item(), 3.0)

if __name__ == "__main__":
    unittest.main()
