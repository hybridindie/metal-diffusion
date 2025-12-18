import pytest
from unittest.mock import patch, MagicMock
from alloy.cli import main
import sys
import os



from alloy.converters.workers import convert_flux_part1

@patch("alloy.converters.workers.FluxTransformer2DModel")
@patch("alloy.converters.workers.FluxPipeline")
@patch("os.path.isfile")
@patch("alloy.converters.workers.ct.convert")
@patch("alloy.utils.coreml.ct.optimize.coreml.linear_quantize_weights")
@patch("alloy.converters.workers.ct.models.MLModel")
@patch("alloy.converters.workers.torch.jit.trace")
@patch("alloy.converters.workers.FluxPart1Wrapper")
def test_flux_worker_int8_quantization(mock_wrapper_cls, mock_trace, mock_load_mlmodel, mock_quantize, mock_ct_convert, mock_isfile, mock_pipeline, mock_transformer, tmp_path):
    """
    Verify worker function loads single file and applies quantization.
    """
    # Setup
    fake_file = "/path/to/local/flux.safetensors"
    mock_isfile.return_value = True
    
    # Mock Transformer Config
    mock_trans_instance = MagicMock()
    mock_trans_instance.config.in_channels = 64
    mock_trans_instance.config.num_attention_heads = 4
    mock_trans_instance.config.attention_head_dim = 64
    mock_trans_instance.config.pooled_projection_dim = 64
    mock_trans_instance.config.joint_attention_dim = 64
    
    mock_transformer.from_single_file.return_value = mock_trans_instance
    
    # 1. CT Convert returns FP16 model
    mock_fp16_model = MagicMock()
    mock_ct_convert.return_value = mock_fp16_model
    
    # 2. Re-load from disk
    mock_reloaded = MagicMock()
    mock_load_mlmodel.return_value = mock_reloaded
    
    # 3. Quantize returns Int8 model
    mock_int8_model = MagicMock()
    mock_quantize.return_value = mock_int8_model
    
    # Run Worker Function directly
    # convert_flux_part_worker definition: 
    # def convert_flux_part1(model_id, output_dir, quantization, intermediates_dir=None):
    
    convert_flux_part1(
        fake_file, 
        str(tmp_path), 
        "int8", 
        intermediates_dir=str(tmp_path)
    )
    
    # Verifications
    
    # 1. loaded from single file?
    mock_transformer.from_single_file.assert_called_once()
    assert fake_file in mock_transformer.from_single_file.call_args[0]
    
    # 2. FP16 saved?
    assert mock_fp16_model.save.called
    
    # 3. Quantization used?
    mock_quantize.assert_called_once()
    assert mock_quantize.call_args[0][0] == mock_fp16_model # In workers.py we might pass the model directly or reloaded one.
    # In workers.py: model = safe_quantize_model(model ...)
    # safe_quantize_model does: model = ct.models.MLModel(...) if needed, or takes model. 
    # If we mocked safe_quantize_model (via linear_quantize_weights), we verify it was called.
    
    # 4. Final Int8 saved?
    assert mock_int8_model.save.called
