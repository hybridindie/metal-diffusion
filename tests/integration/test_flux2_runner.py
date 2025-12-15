import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from metal_diffusion.flux_runner import FluxCoreMLRunner

@patch("metal_diffusion.flux_runner.DiffusionPipeline.from_pretrained")
@patch("metal_diffusion.flux_runner.ct.models.MLModel")
def test_flux2_runner_generate_mocked(mock_mlmodel_cls, mock_pipeline_cls, tmp_path):
    """
    Test the Flux Runner generation loop with mocked Flux 2 models.
    """
    # Setup Mocks
    mock_pipe = MagicMock()
    mock_pipeline_cls.return_value = mock_pipe
    # Fix chained .to()
    mock_pipe.to.return_value = mock_pipe
    
    # Scheduler
    mock_pipe.scheduler.timesteps = [torch.tensor(1.0)] 
    mock_pipe.scheduler.step.return_value = (torch.randn(1, 16, 64),) 
    
    # Encode Prompt for Flux 2: returns (prompt_embeds, text_ids) - NO pooled
    mock_pipe.encode_prompt.return_value = (
        torch.randn(1, 512, 4096), 
        torch.zeros(512, 3)
    )
    
    # VAE Config
    mock_pipe.vae.config.latent_channels = 16 
    mock_pipe.vae_scale_factor = 8 
    mock_pipe.text_encoder.dtype = torch.float32
    
    # Core ML Model
    mock_coreml_model = MagicMock()
    mock_mlmodel_cls.return_value = mock_coreml_model
    dummy_output = np.random.randn(1, 16, 64).astype(np.float32)
    mock_coreml_model.predict.return_value = {"sample": dummy_output}
    
    # Mock VAE Decode
    mock_pipe.vae.decode.return_value = [torch.randn(1, 3, 64, 64)]
    
    # Mock Image Processor
    mock_image = MagicMock()
    mock_image.save = MagicMock()
    mock_pipe.image_processor.postprocess.return_value = [mock_image]
    
    # Init Runner
    # Patch Flux2Pipeline to be the type of our mock so isinstance passes
    with patch("metal_diffusion.flux_runner.Flux2Pipeline", type(mock_pipe)):
        runner = FluxCoreMLRunner("dummy_model_dir")
        
        # Run Generate
        output_path = tmp_path / "test_flux2_output.png"
        runner.generate("test prompt", str(output_path), steps=1, height=64, width=64)
    
    # Verifications
    
    # 1. Pipeline components called
    mock_pipe.encode_prompt.assert_called()
    # Check encode_prompt args (no prompt_2)
    # Flux 2: encode_prompt(prompt=..., device=..., num_images_per_prompt=...)
    call_args = mock_pipe.encode_prompt.call_args
    assert "prompt_2" not in call_args.kwargs
    
    # 2. Core ML Predict called
    assert mock_coreml_model.predict.call_count == 1
    
    # 3. Check inputs to Core ML
    call_args = mock_coreml_model.predict.call_args_list[0]
    inputs = call_args[0][0]
    
    # Verify pooled_projections is NOT present
    assert "pooled_projections" not in inputs
    
    # Check other inputs exist
    assert "hidden_states" in inputs
    assert "encoder_hidden_states" in inputs
    assert "timestep" in inputs
    assert "img_ids" in inputs
    assert "txt_ids" in inputs
    
    # 4. Save called
    mock_image.save.assert_called_with(str(output_path))
