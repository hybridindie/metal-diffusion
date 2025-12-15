import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import coremltools as ct
from alloy.ltx_runner import LTXCoreMLRunner

@patch("alloy.ltx_runner.LTXPipeline.from_pretrained")
@patch("alloy.ltx_runner.ct.models.MLModel")
def test_ltx_runner_init(mock_pipeline, mock_mlmodel, tmp_path):
    """
    Test the LTX Runner generation loop with mocked models.
    """
    # Setup Mocks
    mock_pipe = MagicMock()
    mock_pipeline.return_value = mock_pipe
    # Fix chained .to()
    mock_pipe.to.return_value = mock_pipe
    
    # Mock Scheduler
    mock_pipe.scheduler.timesteps = [torch.tensor(1)] # Single step for test
    # Scheduler step returns tuple(latents, ...) or object with prev_sample
    mock_pipe.scheduler.step.return_value.prev_sample = torch.randn(1, 256, 128) # Packed shape return
    
    # Mock Encode Prompt
    # returns prompt_embeds, prompt_attention_mask
    mock_pipe.encode_prompt.return_value = (
        torch.randn(2, 128, 4096), 
        torch.randn(2, 128)
    )
    
    # Mock Core ML Model
    mock_coreml_model = MagicMock()
    mock_mlmodel.return_value = mock_coreml_model
    # predict returns dict with "sample"
    # Output shape: Packed latents (1, 256, 128)
    dummy_output = np.random.randn(1, 256, 128).astype(np.float32)
    mock_coreml_model.predict.return_value = {"sample": dummy_output}
    
    # Mock VAE
    mock_pipe.vae.latents_mean = torch.zeros(128)
    mock_pipe.vae.latents_std = torch.ones(128)
    mock_pipe.vae.config.scaling_factor = 1.0
    mock_pipe.vae.decode.return_value = [torch.randn(1, 3, 1, 512, 512)]
    
    # Mock Video Processor
    mock_video = MagicMock()
    mock_image = MagicMock()
    mock_image.save = MagicMock()
    mock_video.__getitem__.return_value = [mock_image] # video[0][0]
    mock_pipe.video_processor.postprocess_video.return_value = mock_video
    
    # Init Runner
    runner = LTXCoreMLRunner("dummy_model_dir")
    
    # Run Generate
    output_path = tmp_path / "test_output.png"
    # Use defaults: steps=20, but loop mocked to 1.
    # height=512, width=512, num_frames=8 (default arg in generate)
    runner.generate("test prompt", str(output_path), steps=1)
    
    # Verifications
    
    # 1. Pipeline components called
    mock_pipe.encode_prompt.assert_called()
    mock_pipe.vae.decode.assert_called()
    
    # 2. Core ML Predict called twice (Uncond + Text)
    assert mock_coreml_model.predict.call_count == 2
    
    # 3. Check inputs to Core ML
    # Check dimensions
    call_args = mock_coreml_model.predict.call_args_list[0]
    inputs = call_args[0][0]
    
    # Packed Latents should be (1, 256, 128)
    # Calculation: 
    # H=512, W=512, F=8.
    # Latent H=16, W=16, F=1.
    # S = 16*16*1 = 256.
    assert inputs["hidden_states"].shape == (1, 256, 128)
    assert inputs["num_frames"] == 1
    assert inputs["height"] == 16
    assert inputs["width"] == 16
    
    # 4. Save called
    mock_image.save.assert_called_with(str(output_path))
