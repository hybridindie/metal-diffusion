import pytest
from unittest.mock import patch, MagicMock
from alloy.wan_converter import WanConverter
import os

@patch("torch.jit.trace")
@patch("coremltools.convert")
@patch("coremltools.optimize.coreml.linear_quantize_weights")
@patch("coremltools.models.MLModel")
def test_wan_conversion_pipeline_mocked(mock_mlmodel, mock_quantize, mock_convert, mock_trace, tmp_path):
    """
    Test the full conversion flow orchestrator with mocked heavy ops.
    Ensures files are output (mocked saving) and correct methods are called.
    """
    # Setup
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    output_dir = tmp_path / "converted_wan"
    
    # Mocks
    mock_convert.return_value = MagicMock()
    mock_quantize.return_value = MagicMock()
    mock_trace.return_value = MagicMock()
    
    # Initialize Converter
    # We mock the pipeline loading inside WanConverter.convert
    with patch("alloy.wan_converter.WanPipeline.from_pretrained") as mock_pipeline_cls:
        mock_pipe = MagicMock()
        mock_pipeline_cls.return_value = mock_pipe
        
        # Config mocks for determining input shapes
        mock_pipe.transformer.config = MagicMock()
        mock_pipe.transformer.config.in_channels = 16
        mock_pipe.transformer.config.patch_size = (1, 2, 2)
        mock_pipe.transformer.config.cross_attention_dim = 4096
        
        converter = WanConverter(model_id, str(output_dir), quantization="int4")
        converter.convert()
        
        # Verification
        
        # 1. Pipeline Loaded
        mock_pipeline_cls.assert_called_with(model_id, torch_dtype=pytest.importorskip("torch").float16, variant='fp16')
        
        # 2. Transformer Conversion Called
        # Check if torch.jit.trace was called
        assert mock_trace.call_count >= 1
        
        # 3. CoreML Convert Called
        assert mock_convert.call_count >= 1
        
        # 4. Quantization Called (since we asked for int4)
        assert mock_quantize.call_count >= 1
        
        # 5. Output structure
        # Since we mocked save(), files won't exist, but we can check if save() was called with right paths.
        # But wait, WanConverter calls model.save()
        mock_quantize.return_value.save.assert_called()
        # You can check args if needed:
        # call_args = mock_quantize.return_value.save.call_args
        # assert "Wan2.1_Transformer.mlpackage" in call_args[0][0]
