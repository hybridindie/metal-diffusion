import pytest
from unittest.mock import patch, MagicMock
from alloy.converters.flux import FluxConverter
import os


@patch("alloy.converters.flux.ct")
@patch("alloy.converters.flux.multiprocessing.Process")
def test_flux_conversion_pipeline_mocked(mock_process, mock_ct, tmp_path):
    """
    Test the full Flux conversion flow orchestrator with mocked subprocess workers.

    Since FluxConverter now uses 2-phase subprocess isolation, we mock:
    1. The multiprocessing.Process to prevent actual subprocess spawning
    2. The ct.models.MLModel to mock loading of intermediate files
    3. The ct.utils.make_pipeline to mock pipeline assembly

    Note: FluxConverter doesn't call download_source_weights (workers load directly).
    """
    # Setup
    model_id = "black-forest-labs/FLUX.1-schnell"
    output_dir = tmp_path / "converted_flux"

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

    # Initialize and run Converter
    converter = FluxConverter(model_id, str(output_dir), quantization="int4")
    converter.convert()

    # Verification

    # 1. Two subprocess workers should be spawned (Part 1 and Part 2)
    assert mock_process.call_count == 2

    # 2. Both processes should be started and joined
    assert mock_proc_instance.start.call_count == 2
    assert mock_proc_instance.join.call_count == 2

    # 3. Intermediate models should be loaded (2 parts)
    assert mock_ct.models.MLModel.call_count == 2

    # 4. Pipeline should be assembled from the two parts
    mock_ct.utils.make_pipeline.assert_called_once()

    # 5. Final pipeline should be saved
    mock_pipeline.save.assert_called_once()
