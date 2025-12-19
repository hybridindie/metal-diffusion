"""CoreML runner for Lumina Image 2.0 models."""

import numpy as np
import torch
from diffusers import Lumina2Pipeline
from PIL import Image

from alloy.logging import get_logger
from alloy.runners.core import BaseCoreMLRunner
from alloy.runners.utils import apply_classifier_free_guidance

logger = get_logger(__name__)


class LuminaCoreMLRunner(BaseCoreMLRunner):
    """
    Hybrid runner for Lumina Image 2.0:
    - Tokenizer: PyTorch
    - Text Encoder: Core ML (Gemma2)
    - Transformer: Core ML
    - VAE: PyTorch (CPU/MPS)
    """

    @property
    def model_name(self) -> str:
        return "Lumina"

    @property
    def transformer_filename(self) -> str:
        return "Lumina2_Transformer.mlpackage"

    @property
    def pipeline_class(self):
        return Lumina2Pipeline

    @property
    def default_model_id(self) -> str:
        return "Alpha-VLLM/Lumina-Image-2.0"

    @property
    def supports_single_file(self) -> bool:
        return False

    @property
    def output_key(self) -> str:
        return "hidden_states"  # Lumina uses different output key

    def _load_pipeline(self) -> None:
        """Load Lumina pipeline for scheduler, tokenizer, and VAE."""
        logger.info("Loading generic components from %s...", self.model_id)
        self.pipe = self.pipeline_class.from_pretrained(self.model_id)
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.vae = self.pipe.vae.to(self.device)

    def _load_coreml_models(self) -> None:
        """Load CoreML text encoder and transformer."""
        logger.info("Loading Core ML models...")
        self.text_encoder_model = self._load_coreml_model(
            "Gemma2_TextEncoder.mlpackage",
            description="Text Encoder",
        )
        self.coreml_transformer = self._load_coreml_transformer()

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        Encode prompt using CoreML text encoder.

        Args:
            prompt: Text prompt to encode

        Returns:
            Prompt embeddings tensor
        """
        # Tokenize (Gemma) - max length 256 as per converter
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.numpy().astype(np.int32)

        # Run Core ML Text Encoder
        prediction = self.text_encoder_model.predict({"input_ids": input_ids})
        last_hidden_state = self.from_numpy(prediction["last_hidden_state"])
        return last_hidden_state

    def generate(
        self,
        prompt: str,
        output_path: str,
        height: int = 1024,
        width: int = 1024,
        steps: int = 20,
        guidance_scale: float = 4.0,
    ):
        """
        Generate image with Lumina Image 2.0.

        Args:
            prompt: Text prompt
            output_path: Path to save output image
            height: Output height in pixels
            width: Output width in pixels
            steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            PIL Image
        """
        logger.info("Generating image for prompt: '%s'", prompt)

        # 1. Encode Prompt
        prompt_embeds = self.encode_prompt(prompt)
        neg_prompt_embeds = self.encode_prompt("")  # Unconditional

        # 2. Prepare Latents
        batch_size = 1
        num_channels = self.pipe.transformer.config.in_channels
        h_latent = height // 8
        w_latent = width // 8

        latents = torch.randn(
            (batch_size, num_channels, h_latent, w_latent),
            device=self.device,
            dtype=torch.float32,
        )

        self.scheduler.set_timesteps(steps)

        logger.info("Denoising...")
        for t in self.scheduler.timesteps:
            # A. Unconditional pass
            uncond_inputs = {
                "hidden_states": self.to_numpy(latents),
                "encoder_hidden_states": self.to_numpy(neg_prompt_embeds),
                "timestep": np.array([float(t)], dtype=np.float32),
            }
            noise_pred_uncond = self.predict_coreml(
                self.coreml_transformer, uncond_inputs
            )

            # B. Conditional pass
            cond_inputs = {
                "hidden_states": self.to_numpy(latents),
                "encoder_hidden_states": self.to_numpy(prompt_embeds),
                "timestep": np.array([float(t)], dtype=np.float32),
            }
            noise_pred_text = self.predict_coreml(
                self.coreml_transformer, cond_inputs
            )

            # Apply CFG
            noise_pred = apply_classifier_free_guidance(
                noise_pred_uncond, noise_pred_text, guidance_scale
            )

            # Step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 3. Decode
        logger.info("Decoding...")
        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.vae.decode(latents).sample

        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(image[0])

        pil_image.save(output_path)
        logger.info("Saved to %s", output_path)
        return pil_image
