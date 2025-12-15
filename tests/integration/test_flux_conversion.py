import pytest
from unittest.mock import patch, MagicMock
from alloy.flux_converter import FluxConverter
import os
import torch

@patch("torch.jit.trace")
@patch("coremltools.convert")
@patch("coremltools.optimize.coreml.linear_quantize_weights")
@patch("coremltools.models.MLModel")
def test_flux_conversion_pipeline_mocked(mock_mlmodel, mock_quantize, mock_convert, mock_trace, tmp_path):
    """
    Test the full Flux conversion flow orchestrator with mocked heavy ops.
    """
    # Setup
    model_id = "black-forest-labs/FLUX.1-schnell"
    output_dir = tmp_path / "converted_flux"
    
    # Mocks
    mock_convert.return_value = MagicMock()
    mock_quantize.return_value = MagicMock()
    mock_trace.return_value = MagicMock()
    
    # Initialize Converter
    with patch("alloy.flux_converter.DiffusionPipeline.from_pretrained") as mock_pipeline_cls:
        mock_pipe = MagicMock()
        mock_pipeline_cls.return_value = mock_pipe
        
        # Config mocks
        mock_pipe.transformer.config = MagicMock()
        # Flux Schnell defaults
        mock_pipe.transformer.config.in_channels = 64
        mock_pipe.transformer.config.joint_attention_dim = 4096
        mock_pipe.transformer.config.pooled_projection_dim = 768
        
        converter = FluxConverter(model_id, str(output_dir), quantization="int4")
        converter.convert()
        
        # Verification
        
        # 1. Pipeline Loaded
        mock_pipeline_cls.assert_called_with(model_id, torch_dtype=torch.float32)
        
        # 2. Transformer Conversion Called
        assert mock_trace.call_count >= 1
        
        # Check inputs to trace
        # wrapper = traced_model = torch.jit.trace(wrapper, example_inputs, strict=False)
        args, _ = mock_trace.call_args
        example_inputs = args[1]
        assert len(example_inputs) == 7 
        # hidden_states, encoder_hidden, pooled, timestep, img_ids, txt_ids, guidance
        
        # Timestep should be float
        assert example_inputs[3].dtype == torch.float32
        
        # Img IDs should be 3D
        # (S, 3)
        assert example_inputs[4].shape[1] == 3
        
        # 3. CoreML Convert Called
        assert mock_convert.call_count >= 1
        
        # 4. Quantization Called
        assert mock_quantize.call_count >= 1
        
        # 5. Save Called
        mock_quantize.return_value.save.assert_called()
