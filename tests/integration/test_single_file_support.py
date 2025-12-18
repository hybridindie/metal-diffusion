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
    """Mock Flux pipeline for runner tests only (converters use subprocess workers now)."""
    with patch("alloy.runners.flux.FluxPipeline") as mock_runner, \
         patch("alloy.runners.flux.DiffusionPipeline"):

        # Setup mocks
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_runner.from_single_file.return_value = mock_pipe

        yield mock_runner, mock_pipe

@pytest.fixture
def mock_ltx_pipeline():
    """Mock LTX pipeline for runner tests only (converters use subprocess workers now)."""
    with patch("alloy.runners.ltx.LTXPipeline") as mock_runner:

        # Setup mocks
        mock_pipe = MagicMock()
        mock_pipe.to.return_value = mock_pipe
        mock_runner.from_single_file.return_value = mock_pipe

        yield mock_runner, mock_pipe

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
    mock_runner_cls, _ = mock_flux_pipeline

    runner = FluxCoreMLRunner("dummy_dir", model_id="flux.safetensors")

    mock_isfile.assert_called_with("flux.safetensors")
    mock_runner_cls.from_single_file.assert_called_once()
    assert "flux.safetensors" in mock_runner_cls.from_single_file.call_args[0]

@patch("alloy.converters.flux.ct")
@patch("os.path.isfile")
@patch("alloy.converters.flux.multiprocessing.Process")
@patch.object(FluxConverter, 'download_source_weights', return_value="flux.safetensors")
def test_flux_single_file_converter(mock_download, mock_process, mock_isfile, mock_ct, tmp_path):
    """Test FluxConverter passes single file path to workers."""
    mock_isfile.return_value = True

    # Mock Process to simulate successful worker execution
    mock_proc_instance = MagicMock()
    mock_proc_instance.exitcode = 0
    mock_process.return_value = mock_proc_instance

    # Mock intermediate model loading
    mock_model = MagicMock()
    mock_ct.models.MLModel.return_value = mock_model
    mock_ct.ComputeUnit.CPU_ONLY = "cpu_only"

    # Mock pipeline assembly
    mock_pipeline = MagicMock()
    mock_ct.utils.make_pipeline.return_value = mock_pipeline

    converter = FluxConverter("flux.safetensors", str(tmp_path), "int4")
    converter.convert()

    # Process should be started for part1 and part2
    assert mock_process.call_count == 2

    # Check args passed to Process - model_id should be passed
    call_args = mock_process.call_args_list[0]
    kwargs = call_args[1]

    if 'args' in kwargs:
        worker_args = kwargs['args']
        assert worker_args[0] == "flux.safetensors"

@patch("alloy.converters.ltx.ct")
@patch("os.path.isfile")
@patch("alloy.converters.ltx.multiprocessing.Process")
@patch.object(LTXConverter, 'download_source_weights', return_value="ltx.safetensors")
def test_ltx_single_file_converter(mock_download, mock_process, mock_isfile, mock_ct, tmp_path):
    """Test LTXConverter passes single file path to workers."""
    mock_isfile.return_value = True

    # Mock Process to simulate successful worker execution
    mock_proc_instance = MagicMock()
    mock_proc_instance.exitcode = 0
    mock_process.return_value = mock_proc_instance

    # Mock intermediate model loading
    mock_model = MagicMock()
    mock_ct.models.MLModel.return_value = mock_model
    mock_ct.ComputeUnit.CPU_ONLY = "cpu_only"

    # Mock pipeline assembly
    mock_pipeline = MagicMock()
    mock_ct.utils.make_pipeline.return_value = mock_pipeline

    converter = LTXConverter("ltx.safetensors", str(tmp_path), "int4")
    converter.convert()

    # Process should be started for part1 and part2
    assert mock_process.call_count == 2

    # Check args passed to Process - model_id should be passed
    call_args = mock_process.call_args_list[0]
    kwargs = call_args[1]

    if 'args' in kwargs:
        worker_args = kwargs['args']
        assert worker_args[0] == "ltx.safetensors"
