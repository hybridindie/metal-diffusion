import pytest
from unittest.mock import patch, MagicMock
from alloy.cli import main
import sys
import os

@patch("alloy.cli.detect_model_type")
@patch("alloy.converters.flux.FluxTransformer2DModel")
@patch("alloy.converters.flux.FluxPipeline")
@patch("os.path.isfile")
def test_flux_local_file_transformer_only(mock_isfile, mock_pipeline, mock_transformer, mock_detect, tmp_path):
    """
    Verify that providing a local file path to convert flux triggers 
    FluxTransformer2DModel.from_single_file instead of FluxPipeline.from_single_file
    to avoid unnecessary HF downloads.
    """
    
    # Setup
    fake_file = "/path/to/local/flux.safetensors"
    mock_detect.return_value = "flux"
    mock_isfile.return_value = True # It is a file
    
    # Mock transformer return
    mock_trans_instance = MagicMock()
    # Mock config for subsequent use
    mock_trans_instance.config.in_channels = 64
    mock_trans_instance.config.num_attention_heads = 2
    mock_trans_instance.config.attention_head_dim = 64
    mock_trans_instance.config.pooled_projection_dim = 768
    mock_trans_instance.config.joint_attention_dim = 4096
    
    # from_single_file returns the transformer instance
    mock_transformer.from_single_file.return_value = mock_trans_instance
    
    # Mock conversion trace/coreml to avoid actual heavy lifting
    with patch("alloy.converters.flux.FluxConverter.convert_transformer") as mock_convert_trans:
        # Args
        test_args = ["alloy", "convert", fake_file, "--output-dir", str(tmp_path), "--type", "flux"]
        
        with patch.object(sys, 'argv', test_args):
            main()
            
    # Verification
    # 1. Should check isfile
    mock_isfile.assert_called_with(fake_file)
    
    # 2. Should call FluxTransformer2DModel.from_single_file
    mock_transformer.from_single_file.assert_called_once()
    assert fake_file in mock_transformer.from_single_file.call_args[0]
    
    # 3. Should NOT call FluxPipeline.from_single_file
    mock_pipeline.from_single_file.assert_not_called()
    
    # 4. Should call convert_transformer with OUR mock transformer
    mock_convert_trans.assert_called_once()
    # Check first arg is the transformer mock
    assert mock_convert_trans.call_args[0][0] == mock_trans_instance
