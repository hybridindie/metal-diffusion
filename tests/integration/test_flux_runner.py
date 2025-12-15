import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import coremltools as ct
from alloy.flux_runner import FluxCoreMLRunner

@patch("alloy.flux_runner.DiffusionPipeline.from_pretrained")
@patch("alloy.flux_runner.ct.models.MLModel")
def test_flux_runner_init(mock_mlmodel_cls, mock_pipeline, tmp_path):
    """
    Test the Flux Runner generation loop with mocked models.
    """
    # Setup Mocks
    mock_pipe = MagicMock()
    mock_pipeline.return_value = mock_pipe
    # Fix chained .to()
    mock_pipe.to.return_value = mock_pipe
    
    # Scheduler
    mock_pipe.scheduler.timesteps = [torch.tensor(1.0)] # Single step float
    # step returns tuple(latents, ...)
    mock_pipe.scheduler.step.return_value = (torch.randn(1, 16, 64),) # Packed shape (S=16 for 64x64 img)
    
    # Encode Prompt
    # returns prompt_embeds, pooled_prompt_embeds, text_ids
    mock_pipe.encode_prompt.return_value = (
        torch.randn(1, 512, 4096), 
        torch.randn(1, 768),
        torch.zeros(512, 3)
    )
    
    # VAE Config
    mock_pipe.vae.config.latent_channels = 16 # Flux VAE
    mock_pipe.vae_scale_factor = 8 # 64x64 input -> 8x8 latent
    mock_pipe.text_encoder.dtype = torch.float32
    
    # Core ML Model
    mock_coreml_model = MagicMock()
    mock_mlmodel_cls.return_value = mock_coreml_model
    # predict returns dict with "sample"
    # Output shape: Packed latents (1, S, 64)
    # If Input is 64x64, Latent is 8x8 -> Packed 4x4 -> 16 patches?
    # Wait, packing patches 2x2.
    # Latent (8x8) -> Pad to even (already even).
    # Patches: (8/2) * (8/2) = 4 * 4 = 16 patches.
    # Channel becomes 16 * 4 = 64.
    dummy_output = np.random.randn(1, 16, 64).astype(np.float32)
    mock_coreml_model.predict.return_value = {"sample": dummy_output}
    
    # Mock VAE Decode
    mock_pipe.vae.decode.return_value = [torch.randn(1, 3, 64, 64)]
    
    # Mock Image Processor
    mock_image = MagicMock()
    mock_image.save = MagicMock()
    mock_pipe.image_processor.postprocess.return_value = [mock_image]
    
    # Init Runner
    runner = FluxCoreMLRunner("dummy_model_dir")
    
    # Run Generate
    output_path = tmp_path / "test_output.png"
    # height=64, width=64 for small test
    # steps=1
    runner.generate("test prompt", str(output_path), steps=1, height=64, width=64)
    
    # Verifications
    
    # 1. Pipeline components called
    mock_pipe.encode_prompt.assert_called()
    mock_pipe.vae.decode.assert_called()
    
    # 2. Core ML Predict called
    assert mock_coreml_model.predict.call_count == 1
    
    # 3. Check inputs to Core ML
    call_args = mock_coreml_model.predict.call_args_list[0]
    inputs = call_args[0][0]
    
    # Check shapes
    # hidden_states: Packed latents
    # 8x8 latent -> 16 patches. Shape (1, 16, 64).
    assert inputs["hidden_states"].shape == (1, 16, 64)
    
    # img_ids: (S, 3) -> (16, 3)
    assert inputs["img_ids"].shape == (16, 3)
    
    # txt_ids: (512, 3)
    assert inputs["txt_ids"].shape == (512, 3)
    
    # pooled_projections: (1, 768)
    assert inputs["pooled_projections"].shape == (1, 768)
    
    # Timestep: Should be float
    assert inputs["timestep"].dtype == np.float32
    
    # 4. Save called
    mock_image.save.assert_called_with(str(output_path))
