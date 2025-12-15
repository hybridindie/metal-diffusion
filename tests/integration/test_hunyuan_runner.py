import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import coremltools as ct
from alloy.hunyuan_runner import HunyuanCoreMLRunner

@patch("alloy.hunyuan_runner.ct.models.MLModel")
def test_hunyuan_runner_init(mock_mlmodel):
    # Mock the MLModel constructor
    mock_mlmodel.return_value = MagicMock()

    # Initialize the runner
    runner = HunyuanCoreMLRunner("dummy_model_dir")

    # Assert that MLModel was called with the correct path
    mock_mlmodel.assert_called_once_with("dummy_model_dir/HunyuanVideo_Transformer.mlpackage")
    assert runner.coreml_model is not None

@patch("diffusers.HunyuanVideoPipeline.from_pretrained")
@patch("alloy.hunyuan_runner.ct.models.MLModel")
def test_hunyuan_runner_generate_mocked(mock_pipeline_cls, mock_mlmodel_cls, tmp_path):
    """
    Test the Hunyuan Runner generation loop with mocked models.
    """
    # Setup Mocks
    mock_pipe = MagicMock()
    mock_pipeline_cls.return_value = mock_pipe
    # Fix chained .to() call returning a new mock (needs to return SAME mock)
    mock_pipe.to.return_value = mock_pipe
    
    # Mock Scheduler
    mock_pipe.scheduler.timesteps = [torch.tensor(1)] # Single step for test
    mock_pipe.scheduler.step.return_value.prev_sample = torch.randn(1, 16, 1, 32, 32)
    
    # Mock Encode Prompt
    # returns prompt_embeds, pooled_prompt_embeds, attention_mask
    mock_pipe.encode_prompt.return_value = (
        torch.randn(2, 10, 4096), 
        torch.randn(2, 768), 
        torch.randint(0, 2, (2, 10))
    )
    
    # Mock Core ML Model
    mock_coreml_model = MagicMock()
    mock_mlmodel_cls.return_value = mock_coreml_model
    # predict returns dict with "sample"
    # Output shape: similar to latent (1, 16, 1, 32, 32)
    dummy_output = np.random.randn(1, 16, 1, 32, 32).astype(np.float32)
    mock_coreml_model.predict.return_value = {"sample": dummy_output}
    
    # Mock VAE Decode
    # decode returns [image_tensor]
    # Input was (1, 16, 1, 32, 32), Output should be video (1, 3, 1, 256, 256)
    mock_pipe.vae.config.scaling_factor = 0.18215
    mock_pipe.vae.decode.return_value = [torch.randn(1, 3, 1, 256, 256)]
    
    # Mock Image Processor
    # postprocess returns list of PIL images
    mock_image = MagicMock()
    mock_pipe.image_processor.postprocess.return_value = [mock_image]
    
    # Init Runner
    runner = HunyuanCoreMLRunner("dummy_model_dir")
    
    # Run Generate
    output_path = tmp_path / "test_output.png"
    runner.generate("test prompt", str(output_path), steps=1, height=512, width=512)
    
    # Verifications
    
    # 1. Diffusers Pipeline components used
    mock_pipe.encode_prompt.assert_called()
    mock_pipe.vae.decode.assert_called()
    
    # 2. Core ML Predict called
    mock_coreml_model.predict.assert_called()
    
    # 3. Check inputs to Core ML
    call_args = mock_coreml_model.predict.call_args
    inputs = call_args[0][0]
    assert "hidden_states" in inputs
    assert "timestep" in inputs
    assert "encoder_hidden_states" in inputs # Checking key names
    assert "guidance" in inputs
    
    # 4. Save called
    mock_image.save.assert_called_with(str(output_path))
