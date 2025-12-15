import pytest
from unittest.mock import patch, MagicMock
import os
from metal_diffusion.flux_runner import FluxCoreMLRunner
from metal_diffusion.flux_converter import FluxConverter
from metal_diffusion.ltx_runner import LTXCoreMLRunner
from metal_diffusion.ltx_converter import LTXConverter

# Mock Pipeline classes
class MockPipeline:
    pass

@pytest.fixture
def mock_flux_pipeline():
    with patch("metal_diffusion.flux_runner.FluxPipeline") as mock_runner, \
         patch("metal_diffusion.flux_converter.FluxPipeline") as mock_converter, \
         patch("metal_diffusion.flux_runner.DiffusionPipeline"), \
         patch("metal_diffusion.flux_converter.DiffusionPipeline"):
        
        # Setup mocks
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_runner.from_single_file.return_value = mock_pipe
        mock_converter.from_single_file.return_value = mock_pipe
        
        yield mock_runner, mock_converter, mock_pipe

@pytest.fixture
def mock_ltx_pipeline():
    with patch("metal_diffusion.ltx_runner.LTXPipeline") as mock_runner, \
         patch("metal_diffusion.ltx_converter.LTXPipeline") as mock_converter:
        
        # Setup mocks
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_runner.from_single_file.return_value = mock_pipe
        mock_converter.from_single_file.return_value = mock_pipe
        
        yield mock_runner, mock_converter, mock_pipe

@patch("os.path.isfile")
@patch("metal_diffusion.flux_runner.ct.models.MLModel")
def test_flux_single_file_runner(mock_mlmodel, mock_isfile, mock_flux_pipeline):
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
@patch("metal_diffusion.ltx_runner.ct.models.MLModel")
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
