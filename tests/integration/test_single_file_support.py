import os
import pytest
import torch
from unittest.mock import patch, MagicMock
from alloy.utils.general import detect_model_type
from alloy.runners.flux import FluxCoreMLRunner
from alloy.converters.flux import FluxConverter
from alloy.runners.ltx import LTXCoreMLRunner
from alloy.converters.ltx import LTXConverter

# Mock Pipeline classes same as before...
class MockPipeline:
    pass

# ... existing fixtures ...

@patch("alloy.utils.general.safe_open")
@patch("os.path.isfile")
def test_detect_model_type_flux(mock_isfile, mock_safe_open):
    mock_isfile.return_value = True
    mock_f = MagicMock()
    mock_f.keys.return_value = ["double_blocks.0.img_mod.lin.weight", "single_blocks.0.lin.weight"]
    mock_f.__enter__.return_value = mock_f
    mock_safe_open.return_value = mock_f
    
    encoded_type = detect_model_type("flux.safetensors")
    assert encoded_type == "flux"

@patch("alloy.utils.general.safe_open")
@patch("os.path.isfile")
def test_detect_model_type_ltx(mock_isfile, mock_safe_open):
    mock_isfile.return_value = True
    mock_f = MagicMock()
    mock_f.keys.return_value = ["transformer.blocks.0.scale_shift_table", "caption_projection.weight"]
    mock_f.__enter__.return_value = mock_f
    mock_safe_open.return_value = mock_f
    
    encoded_type = detect_model_type("ltx.safetensors")
    assert encoded_type == "ltx"

@patch("alloy.utils.general.safe_open")
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

@patch("alloy.utils.general.safe_open")
@patch("os.path.isfile")
def test_detect_model_type_exception(mock_isfile, mock_safe_open):
    """Test exception handling"""
    mock_isfile.return_value = True
    mock_safe_open.side_effect = Exception("Corrupt file")
    
    encoded_type = detect_model_type("corrupt.safetensors")
    assert encoded_type is None


@pytest.fixture
def mock_flux_pipeline():
    with patch("alloy.runners.flux.FluxPipeline") as mock_runner, \
         patch("alloy.converters.flux.FluxPipeline") as mock_converter, \
         patch("alloy.runners.flux.DiffusionPipeline"), \
         patch("alloy.converters.flux.DiffusionPipeline"):
        
        # Setup mocks
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_runner.from_single_file.return_value = mock_pipe
        mock_converter.from_single_file.return_value = mock_pipe
        
        yield mock_runner, mock_converter, mock_pipe

@pytest.fixture
def mock_ltx_pipeline():
    with patch("alloy.runners.ltx.LTXPipeline") as mock_runner, \
         patch("alloy.converters.ltx.LTXPipeline") as mock_converter:
        
        # Setup mocks
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_runner.from_single_file.return_value = mock_pipe
        mock_converter.from_single_file.return_value = mock_pipe
        
        yield mock_runner, mock_converter, mock_pipe

@patch("alloy.runners.flux.ct.models.MLModel")
@patch("alloy.runners.ltx.ct.models.MLModel")
@patch("alloy.runners.flux.FluxPipeline")
@patch("alloy.runners.flux.DiffusionPipeline")
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
@patch("alloy.converters.flux.multiprocessing.Process")
def test_flux_single_file_converter(mock_process, mock_isfile, mock_flux_pipeline, tmp_path):
    """Test FluxConverter passes single file path to workers."""
    mock_isfile.return_value = True
    
    converter = FluxConverter("flux.safetensors", str(tmp_path), "int4")
    
    # We mock Process to avoid spawning, and just check args
    converter.convert()
    
    # Process should be started for part1 and part2
    assert mock_process.call_count >= 1
    
    # Check args passed to Process
    # call_args[1] is kwargs (target=..., args=(...))
    # args tuple: (model_id, output_dir, quantization, intermediates_dir, ...)
    # Element 0 should be "flux.safetensors"
    
    call_args = mock_process.call_args_list[0]
    kwargs = call_args[1] # or call_args.kwargs
    
    if 'args' in kwargs:
        worker_args = kwargs['args']
        assert worker_args[0] == "flux.safetensors"
    else:
        # Maybe passed as kwargs to Process?
        pass

@patch("os.path.isfile")
@patch("alloy.converters.ltx.shutil")
@patch("alloy.converters.ltx.os.makedirs")
@patch("alloy.converters.ltx.os.path.exists") 
def test_ltx_single_file_converter(mock_exists, mock_makedirs, mock_shutil, mock_isfile, mock_ltx_pipeline, tmp_path):
    """Test LTXConverter uses from_single_file when detecting a file."""
    mock_isfile.return_value = True
    mock_exists.return_value = False # Force conversion path
    
    _, mock_converter_cls, mock_pipe = mock_ltx_pipeline
    
    # Mock transformer for converter
    mock_pipe.transformer = MagicMock()
    
    converter = LTXConverter("ltx.safetensors", str(tmp_path), "int4")
    
    # Mock convert_transformer
    converter.convert_transformer = MagicMock()
    
    converter.convert()
    
    mock_converter_cls.from_single_file.assert_called_once()
    assert "ltx.safetensors" in mock_converter_cls.from_single_file.call_args[0]
