import pytest
from unittest.mock import patch, MagicMock
from alloy.converters.hunyuan import HunyuanConverter
import os
import torch

@patch("torch.jit.trace")
@patch("alloy.utils.coreml.ct.models.MLModel")
@patch("alloy.utils.coreml.ct.optimize.coreml.linear_quantize_weights")
@patch("alloy.converters.hunyuan.ct")
def test_hunyuan_conversion_pipeline_mocked(mock_ct, mock_quantize, mock_mlmodel_cls, mock_trace, tmp_path):
    """
    Test the full Hunyuan conversion flow orchestrator with mocked heavy ops.
    """
    # Setup
    model_id = "hunyuanvideo-community/HunyuanVideo"
    output_dir = tmp_path / "converted_hunyuan"
    
    # Mocks
    mock_ct.convert.return_value = MagicMock()
    mock_ct.optimize.coreml.linear_quantize_weights.return_value = MagicMock()
    mock_trace.return_value = MagicMock()
    
    # Initialize Converter
    with patch("alloy.converters.hunyuan.HunyuanVideoPipeline.from_pretrained") as mock_pipeline_cls:
        mock_pipe = MagicMock()
        mock_pipeline_cls.return_value = mock_pipe
        
        # Config mocks
        mock_pipe.transformer.config = MagicMock()
        mock_pipe.transformer.config.in_channels = 16
        mock_pipe.transformer.config.text_embed_dim = 4096
        mock_pipe.transformer.config.pooled_projection_dim = 768
        
        converter = HunyuanConverter(model_id, str(output_dir), quantization="int4")
        converter.convert()
        
        # Verification
        
        # 1. Pipeline Loaded
        mock_pipeline_cls.assert_called_with(model_id)
        
        # 2. Transformer Conversion Called
        assert mock_trace.call_count >= 1
        
        # Check inputs to trace (roughly)
        # wrapper = traced_model = torch.jit.trace(wrapper, example_inputs, strict=False)
        # We can inspect call_args if we want deeper verification
        
        # 3. CoreML Convert Called
        assert mock_ct.convert.call_count >= 1
        
        # 4. Quantization Called
        assert mock_quantize.call_count >= 1
        
        # 5. Save Called
        mock_quantize.return_value.save.assert_called()
