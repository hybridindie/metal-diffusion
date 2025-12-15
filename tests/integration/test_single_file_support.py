import os
import pytest
import torch
from unittest.mock import patch, MagicMock
from alloy.utils import detect_model_type
from alloy.flux_runner import FluxCoreMLRunner
from alloy.flux_converter import FluxConverter
from alloy.ltx_runner import LTXCoreMLRunner
from alloy.ltx_converter import LTXConverter

# Mock Pipeline classes same as before...
class MockPipeline:
    pass

# ... existing fixtures ...

@patch("alloy.utils.safe_open")
@patch("os.path.isfile")
def test_detect_model_type_flux(mock_isfile, mock_safe_open):
    mock_isfile.return_value = True
    mock_f = MagicMock()
    mock_f.keys.return_value = ["double_blocks.0.img_mod.lin.weight", "single_blocks.0.lin.weight"]
    mock_f.__enter__.return_value = mock_f
    mock_safe_open.return_value = mock_f
    
    encoded_type = detect_model_type("flux.safetensors")
    assert encoded_type == "flux"

@patch("alloy.utils.safe_open")
@patch("os.path.isfile")
def test_detect_model_type_ltx(mock_isfile, mock_safe_open):
    mock_isfile.return_value = True
    mock_f = MagicMock()
    mock_f.keys.return_value = ["transformer.blocks.0.scale_shift_table", "caption_projection.weight"]
    mock_f.__enter__.return_value = mock_f
    mock_safe_open.return_value = mock_f
    
    encoded_type = detect_model_type("ltx.safetensors")
    assert encoded_type == "ltx"

@patch("alloy.utils.safe_open")
@patch("os.path.isfile")
def test_detect_model_type_unknown(mock_isfile, mock_safe_open):
    """Test unknown key pattern returns None"""
    mock_isfile.return_value = True
    mock_f = MagicMock()
    mock_f.keys.return_value = ["random.keys.only"]
    mock_f.__enter__.return_value = mock_f
    mock_safe_open.return_value = mock_f
    
    encoded_type = detect_model_type("unknown.safetensors")
    assert encoded_type is None

@patch("alloy.utils.safe_open")
@patch("os.path.isfile")
def test_detect_model_type_exception(mock_isfile, mock_safe_open):
    """Test exception handling"""
    mock_isfile.return_value = True
    mock_safe_open.side_effect = Exception("Corrupt file")
    
    encoded_type = detect_model_type("corrupt.safetensors")
    assert encoded_type is None


@pytest.fixture
def mock_flux_pipeline():
    with patch("alloy.flux_runner.FluxPipeline") as mock_runner, \
         patch("alloy.flux_converter.FluxPipeline") as mock_converter, \
         patch("alloy.flux_runner.DiffusionPipeline"), \
         patch("alloy.flux_converter.DiffusionPipeline"):
        
        # Setup mocks
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_runner.from_single_file.return_value = mock_pipe
        mock_converter.from_single_file.return_value = mock_pipe
        
        yield mock_runner, mock_converter, mock_pipe

@pytest.fixture
def mock_ltx_pipeline():
    with patch("alloy.ltx_runner.LTXPipeline") as mock_runner, \
         patch("alloy.ltx_converter.LTXPipeline") as mock_converter:
        
        # Setup mocks
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_runner.from_single_file.return_value = mock_pipe
        mock_converter.from_single_file.return_value = mock_pipe
        
        yield mock_runner, mock_converter, mock_pipe

@patch("alloy.flux_runner.ct.models.MLModel")
@patch("alloy.ltx_runner.ct.models.MLModel")
@patch("alloy.flux_runner.FluxPipeline")
@patch("alloy.flux_runner.DiffusionPipeline")
def test_runner_initialization(mock_diff_pipe, mock_flux_pipe, mock_ltx_mlmodel, mock_flux_mlmodel):
    # Test Flux Runner
    mock_flux_mlmodel.return_value = MagicMock()
    
    flux_runner = FluxCoreMLRunner("dummy_path")
    assert flux_runner

@patch("coremltools.models.MLModel")
@patch("os.path.isfile")
def test_flux_single_file_runner(mock_isfile, mock_mlmodel, mock_flux_pipeline):
    """Test FluxCoreMLRunner uses from_single_file when detecting a file."""
    mock_isfile.return_value = True
    mock_runner_cls, _, _ = mock_flux_pipeline
    
    runner = FluxCoreMLRunner("dummy_dir", model_id="flux.safetensors")
    
    mock_isfile.assert_called_with("flux.safetensors")
    mock_runner_cls.from_single_file.assert_called_once()
    assert "flux.safetensors" in mock_runner_cls.from_single_file.call_args[0]

@patch("os.path.isfile")
def test_flux_single_file_converter(mock_isfile, mock_flux_pipeline, tmp_path):
    """Test FluxConverter uses from_single_file when detecting a file."""
    mock_isfile.return_value = True
    _, mock_converter_cls, mock_pipe = mock_flux_pipeline
    
    # Mock transformer for converter
    mock_pipe.transformer = MagicMock()
    
    converter = FluxConverter("flux.safetensors", str(tmp_path), "int4")
    
    # Mock convert_transformer so we don't actually run trace
    converter.convert_transformer = MagicMock()
    
    converter.convert()
    
    mock_isfile.assert_called()
    mock_converter_cls.from_single_file.assert_called_once()
    assert "flux.safetensors" in mock_converter_cls.from_single_file.call_args[0]

@patch("os.path.isfile")
@patch("coremltools.models.MLModel")
def test_ltx_single_file_runner(mock_mlmodel, mock_isfile, mock_ltx_pipeline):
    """Test LTXCoreMLRunner uses from_single_file when detecting a file."""
    mock_isfile.return_value = True
    mock_runner_cls, _, _ = mock_ltx_pipeline
    
    runner = LTXCoreMLRunner("dummy_dir", model_id="ltx.safetensors")
    
    mock_isfile.assert_called_with("ltx.safetensors")
    mock_runner_cls.from_single_file.assert_called_once()
    assert "ltx.safetensors" in mock_runner_cls.from_single_file.call_args[0]

@patch("os.path.isfile")
def test_ltx_single_file_converter(mock_isfile, mock_ltx_pipeline, tmp_path):
    """Test LTXConverter uses from_single_file when detecting a file."""
    mock_isfile.return_value = True
    _, mock_converter_cls, mock_pipe = mock_ltx_pipeline
    
    # Mock transformer for converter
    mock_pipe.transformer = MagicMock()
    
    converter = LTXConverter("ltx.safetensors", str(tmp_path), "int4")
    
    # Mock convert_transformer
    converter.convert_transformer = MagicMock()
    
    converter.convert()
    
    mock_isfile.assert_called()
    mock_converter_cls.from_single_file.assert_called_once()
    assert "ltx.safetensors" in mock_converter_cls.from_single_file.call_args[0]
