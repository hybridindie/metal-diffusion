import pytest
from unittest.mock import patch, MagicMock
from alloy.ltx_converter import LTXConverter
import os
import torch

@patch("torch.jit.trace")
@patch("coremltools.convert")
@patch("coremltools.optimize.coreml.linear_quantize_weights")
@patch("coremltools.models.MLModel")
def test_ltx_conversion_pipeline_mocked(mock_mlmodel, mock_quantize, mock_convert, mock_trace, tmp_path):
    """
    Test the full LTX conversion flow orchestrator with mocked heavy ops.
    """
    # Setup
    model_id = "Lightricks/LTX-Video"
    output_dir = tmp_path / "converted_ltx"
    
    # Mocks
    mock_convert.return_value = MagicMock()
    mock_quantize.return_value = MagicMock()
    mock_trace.return_value = MagicMock()
    
    # Initialize Converter
    with patch("alloy.ltx_converter.LTXPipeline.from_pretrained") as mock_pipeline_cls:
        mock_pipe = MagicMock()
        mock_pipeline_cls.return_value = mock_pipe
        
        # Config mocks
        mock_pipe.transformer.config = MagicMock()
        mock_pipe.transformer.config.in_channels = 128
        
        converter = LTXConverter(model_id, str(output_dir), quantization="int4")
        converter.convert()
        
        # Verification
        
        # 1. Pipeline Loaded
        mock_pipeline_cls.assert_called_with(model_id)
        
        # 2. Transformer Conversion Called
        assert mock_trace.call_count >= 1
        
        # Check inputs to trace
        # wrapper = traced_model = torch.jit.trace(wrapper, example_inputs, strict=False)
        # We can check specific arg types
        args, _ = mock_trace.call_args
        example_inputs = args[1]
        assert len(example_inputs) == 7 # hidden, encoder_hidden, timestep, mask, frames, h, w
        assert example_inputs[4].item() == 8 # Default frames I set in converter
        
        # 3. CoreML Convert Called
        assert mock_convert.call_count >= 1
        
        # 4. Quantization Called
        assert mock_quantize.call_count >= 1
        
        # 5. Save Called
        mock_quantize.return_value.save.assert_called()
