import pytest
from unittest.mock import patch, MagicMock
from alloy.cli import main
import sys

def test_cli_auto_detect_single_file_flux():
    with patch("sys.argv", ["alloy", "convert", "flux.safetensors"]):
        with patch("alloy.cli.detect_model_type") as mock_detect:
            mock_detect.return_value = "flux"
             
            with patch("alloy.cli.FluxConverter") as MockConverter:
                main()
                MockConverter.assert_called_once()
                args, kwargs = MockConverter.call_args

def test_cli_auto_detect_single_file_ltx():
   with patch("sys.argv", ["alloy", "convert", "ltx.safetensors"]):
        with patch("alloy.cli.detect_model_type") as mock_detect:
            mock_detect.return_value = "ltx"
             
            with patch("alloy.cli.SDConverter") as MockSD, \
                 patch("alloy.cli.LTXConverter") as MockLTX:
                main()
                MockLTX.assert_called_once()
                MockSD.assert_not_called()

def test_cli_auto_detect_fail_fallback_sd1():
   # If detection returns None, and no type arg, should error out? 
   # Actually current logic errors out if unspecified and folder.
   # For file, detect_model_type returns "sd" ? No, returns None if unknown.
   # If None, CLI prompts to specify --type.
    with patch("sys.argv", ["alloy", "convert", "unknown.safetensors"]):
        with patch("alloy.cli.detect_model_type") as mock_detect:
            mock_detect.return_value = None
            
            with pytest.raises(SystemExit):
                main()

def test_cli_override_detection():
    # User specifies --type even if detected
    with patch("sys.argv", ["alloy", "convert", "flux.safetensors", "--type", "ltx"]):
         with patch("alloy.cli.detect_model_type") as mock_detect:
            # Even if detection says flux
            mock_detect.return_value = "flux"
            
            with patch("alloy.cli.LTXConverter") as MockLTX:
                main()
                MockLTX.assert_called_once()

@patch("alloy.cli.detect_model_type")
@patch("alloy.cli.FluxConverter")
@patch("alloy.cli.SDConverter")
@patch("alloy.cli.LTXConverter")
def test_cli_auto_detect_flux(mock_ltx, mock_sd, mock_flux, mock_detect):
    """Test standard CLI conversion flow with auto-detection for Flux."""
    
    # Mock auto-detection to return 'flux'
    mock_detect.return_value = "flux"
    
    # Mock argv: convert flux.safetensors --output-dir out
    # Note: NO --type specified
    test_args = ["alloy", "convert", "flux.safetensors", "--output-dir", "out"]
    
    with patch.object(sys, 'argv', test_args):
        main()
    
    # Verify detect was called
    mock_detect.assert_called_with("flux.safetensors")
    
    # Verify FluxConverter was initialized and run
    # Verify FluxConverter was initialized and run
    mock_flux.assert_called_with("flux.safetensors", "out", "float16", loras=None, controlnet_compatible=False)
    mock_flux.return_value.convert.assert_called_once()
    
    # Verify others NOT called
    mock_sd.assert_not_called()
    mock_ltx.assert_not_called()

@patch("alloy.cli.detect_model_type")
@patch("alloy.cli.LTXConverter")
def test_cli_auto_detect_ltx(mock_ltx, mock_detect):
    """Test standard CLI conversion flow with auto-detection for LTX."""
    
    # Mock auto-detection to return 'ltx'
    mock_detect.return_value = "ltx"
    
    # Mock argv
    test_args = ["alloy", "convert", "ltx.safetensors", "--output-dir", "out"]
    
    with patch.object(sys, 'argv', test_args):
        main()
    
    # Verify LTXConverter
    mock_ltx.assert_called_with("ltx.safetensors", "out", "float16")
    mock_ltx.return_value.convert.assert_called_once()

@patch("alloy.cli.detect_model_type")
def test_cli_auto_detect_failure(mock_detect):
    """Test CLI exit when detection fails."""
    
    # Mock auto-detection to return None
    mock_detect.return_value = None
    
    test_args = ["alloy", "convert", "unknown.safetensors"]
    
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        
        assert excinfo.value.code == 1

@patch("alloy.cli.detect_model_type")
def test_cli_auto_detect_quantization_int8(mock_detect):
    """Test auto-detection of int8 quantization from filename."""
    mock_detect.return_value = "flux"
    
    with patch("alloy.cli.FluxConverter") as MockFlux:
        test_args = ["alloy", "convert", "flux1-dev_fp8.safetensors", "--output-dir", "out"]
        with patch.object(sys, 'argv', test_args):
            main()
            
        MockFlux.assert_called_with("flux1-dev_fp8.safetensors", "out", "int8", loras=None, controlnet_compatible=False)

@patch("alloy.cli.detect_model_type")
def test_cli_auto_detect_quantization_default(mock_detect):
    """Test default float16 if no quantization specified and no filename match."""
    mock_detect.return_value = "flux"
    
    with patch("alloy.cli.FluxConverter") as MockFlux:
        test_args = ["alloy", "convert", "flux1-dev.safetensors", "--output-dir", "out"]
        with patch.object(sys, 'argv', test_args):
            main()
            
        MockFlux.assert_called_with("flux1-dev.safetensors", "out", "float16", loras=None, controlnet_compatible=False)

@patch("alloy.cli.detect_model_type")
def test_cli_auto_detect_quantization_robust_int8(mock_detect):
    """Test auto-detection via robust safetensors inspection."""
    mock_detect.return_value = "flux"
    
    # Mock the precision detector directly, or mock safe_open?
    # Mocking detect_safetensors_precision is cleaner for CLI test, 
    # assuming we test detect_safetensors_precision separately (or trust it for now).
    # Since we imported it in CLI, we can patch it in alloy.cli
    
    with patch("alloy.cli.detect_safetensors_precision") as mock_precision:
        mock_precision.return_value = "int8"
        
        with patch("alloy.cli.FluxConverter") as MockFlux:
            # Filename implies nothing
            test_args = ["alloy", "convert", "flux_model.safetensors", "--output-dir", "out"]
            with patch.object(sys, 'argv', test_args):
                main()
                
            # Should follow robust detection
            mock_precision.assert_called_with("flux_model.safetensors")
            MockFlux.assert_called_with("flux_model.safetensors", "out", "int8", loras=None, controlnet_compatible=False)

@patch("alloy.cli.detect_safetensors_precision")
def test_cli_quantization_warning_redundant(mock_precision, capsys):
    """Test warning when user requests redundant quantization."""
    mock_precision.return_value = "int8"
    
    with patch("alloy.cli.FluxConverter"):
        with patch("alloy.cli.detect_model_type") as m_type:
            m_type.return_value = "flux"
            # User requests int8 on int8 file
            test_args = ["alloy", "convert", "flux.safetensors", "--quantization", "int8"]
            with patch.object(sys, 'argv', test_args):
                main()
    
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "same as the input" in captured.out

@patch("alloy.cli.detect_safetensors_precision")
def test_cli_quantization_warning_upscale(mock_precision, capsys):
    """Test warning when user requests higher precision than input."""
    mock_precision.return_value = "int8"
    
    with patch("alloy.cli.FluxConverter"):
        with patch("alloy.cli.detect_model_type") as m_type:
            m_type.return_value = "flux"
            # User requests float16 on int8 file
            test_args = ["alloy", "convert", "flux.safetensors", "--quantization", "float16"]
            with patch.object(sys, 'argv', test_args):
                main()
    
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "higher precision than the input" in captured.out

@patch("alloy.cli.detect_safetensors_precision")
def test_cli_quantization_warning_double_quant(mock_precision, capsys):
    """Test warning when user requests double quantization (e.g. Int8 -> Int4)."""
    mock_precision.return_value = "int8"
    
    with patch("alloy.cli.FluxConverter"):
        with patch("alloy.cli.detect_model_type") as m_type:
            m_type.return_value = "flux"
            # User requests int4 on int8 file
            test_args = ["alloy", "convert", "flux.safetensors", "--quantization", "int4"]
            with patch.object(sys, 'argv', test_args):
                main()
    
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "degradation due to double-quantization" in captured.out

