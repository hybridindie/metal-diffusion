import pytest
import sys
from unittest.mock import patch, MagicMock
from metal_diffusion.cli import main

@patch("metal_diffusion.cli.detect_model_type")
@patch("metal_diffusion.cli.FluxConverter")
@patch("metal_diffusion.cli.SDConverter")
@patch("metal_diffusion.cli.LTXConverter")
def test_cli_auto_detect_flux(mock_ltx, mock_sd, mock_flux, mock_detect):
    """Test standard CLI conversion flow with auto-detection for Flux."""
    
    # Mock auto-detection to return 'flux'
    mock_detect.return_value = "flux"
    
    # Mock argv: convert flux.safetensors --output-dir out
    # Note: NO --type specified
    test_args = ["metal-diffusion", "convert", "flux.safetensors", "--output-dir", "out"]
    
    with patch.object(sys, 'argv', test_args):
        main()
    
    # Verify detect was called
    mock_detect.assert_called_with("flux.safetensors")
    
    # Verify FluxConverter was initialized and run
    mock_flux.assert_called_with("flux.safetensors", "out", "float16")
    mock_flux.return_value.convert.assert_called_once()
    
    # Verify others NOT called
    mock_sd.assert_not_called()
    mock_ltx.assert_not_called()

@patch("metal_diffusion.cli.detect_model_type")
@patch("metal_diffusion.cli.LTXConverter")
def test_cli_auto_detect_ltx(mock_ltx, mock_detect):
    """Test standard CLI conversion flow with auto-detection for LTX."""
    
    # Mock auto-detection to return 'ltx'
    mock_detect.return_value = "ltx"
    
    # Mock argv
    test_args = ["metal-diffusion", "convert", "ltx.safetensors", "--output-dir", "out"]
    
    with patch.object(sys, 'argv', test_args):
        main()
    
    # Verify LTXConverter
    mock_ltx.assert_called_with("ltx.safetensors", "out", "float16")
    mock_ltx.return_value.convert.assert_called_once()

@patch("metal_diffusion.cli.detect_model_type")
def test_cli_auto_detect_failure(mock_detect):
    """Test CLI exit when detection fails."""
    
    # Mock auto-detection to return None
    mock_detect.return_value = None
    
    test_args = ["metal-diffusion", "convert", "unknown.safetensors"]
    
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        
        assert excinfo.value.code == 1

