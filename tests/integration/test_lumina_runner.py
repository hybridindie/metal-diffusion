import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import coremltools as ct
from alloy.lumina_runner import LuminaCoreMLRunner

@patch("diffusers.Lumina2Pipeline.from_pretrained")
@patch("alloy.lumina_runner.ct.models.MLModel")
def test_lumina_runner_init(mock_mlmodel, mock_pipeline, tmp_path):
    """
    Test the Lumina Runner generation loop with mocked models.
    """
    # Setup Mocks
    mock_pipe = MagicMock()
    mock_pipeline.return_value = mock_pipe
    
    # Mock Scheduler
    mock_pipe.scheduler.timesteps = [torch.tensor(1)] # Single step for test
    # Scheduler step returns tuple(latents, ...) or object with prev_sample
    mock_pipe.scheduler.step.return_value.prev_sample = torch.randn(1, 16, 128, 128)
    
    # Mock Tokenizer
    mock_pipe.tokenizer.return_value.input_ids = torch.randint(0, 100, (1, 256))
    
    # Mock VAE
    mock_pipe.vae.to.return_value = mock_pipe.vae # chained to()
    mock_pipe.vae.decode.return_value.sample = torch.randn(1, 3, 1024, 1024)
    
    # Mock Core ML Models
    mock_text_encoder = MagicMock()
    mock_transformer = MagicMock()
    mock_mlmodel.side_effect = [mock_text_encoder, mock_transformer]
    
    # Text Encoder Predict
    # Last hidden state (1, 256, 2304)
    mock_text_encoder.predict.return_value = {
        "last_hidden_state": np.random.randn(1, 256, 2304).astype(np.float32)
    }
    
    # Transformer Predict
    # Hidden states (1, 16, 128, 128)
    mock_transformer.predict.return_value = {
        "hidden_states": np.random.randn(1, 16, 128, 128).astype(np.float32)
    }
    
    # Init Runner
    runner = LuminaCoreMLRunner("dummy_model_dir")
    
    # Run Generate
    output_path = tmp_path / "test_output.png"
    runner.generate("test prompt", str(output_path), steps=1, height=1024, width=1024)
    
    # Verifications
    
    # 1. Text Encoder called
    mock_text_encoder.predict.assert_called()
    
    # 2. Transformer called (twice per step for CFG usually, or we did standard loop logic)
    # in our runner we did uncond + cond = 2 calls per step.
    assert mock_transformer.predict.call_count == 2
    
    # 3. Check Transformer Inputs
    call_args = mock_transformer.predict.call_args
    inputs = call_args[0][0]
    assert "hidden_states" in inputs
    assert "encoder_hidden_states" in inputs
    assert "timestep" in inputs
