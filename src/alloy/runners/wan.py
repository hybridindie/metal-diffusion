"""CoreML runner for Wan 2.1 models."""

import numpy as np
import torch
from diffusers import WanPipeline
from PIL import Image

from alloy.logging import get_logger
from alloy.runners.core import BaseCoreMLRunner

logger = get_logger(__name__)

# Latent space constants
LATENT_CHANNELS = 16
VAE_SCALE_FACTOR = 8


class WanCoreMLRunner(BaseCoreMLRunner):
    """
    Hybrid runner for Wan 2.1:
    - Text Encoder: PyTorch (CPU/MPS)
    - Transformer: Core ML
    - VAE: Core ML
    """

    @property
    def model_name(self) -> str:
        return "Wan"

    @property
    def transformer_filename(self) -> str:
        return "Wan2.1_Transformer.mlpackage"

    @property
    def pipeline_class(self):
        return WanPipeline

    @property
    def default_model_id(self) -> str:
        return "Wan-AI/Wan2.1-T2V-14B-Diffusers"

    @property
    def supports_single_file(self) -> bool:
        return False  # Wan doesn't support single-file loading

    def _load_pipeline(self) -> None:
        """Load Wan pipeline with PyTorch text encoder."""
        logger.info("Loading PyTorch components (Text Encoder) from %s...", self.model_id)
        self.pipe = self.pipeline_class.from_pretrained(
            self.model_id,
            torch_dtype=self.default_dtype,
        ).to(self.device)

    def _load_coreml_models(self) -> None:
        """Load CoreML transformer and VAE decoder."""
        self.coreml_transformer = self._load_coreml_transformer()

        logger.info("Loading Core ML VAE...")
        self.coreml_vae = self._load_coreml_model(
            "Wan2.1_VAE_Decoder.mlpackage",
            description="VAE Decoder",
        )

    def generate(
        self,
        prompt: str,
        output_path: str,
        steps: int = 20,
        height: int = 512,
        width: int = 512,
    ) -> None:
        """
        Generate image with Wan 2.1.

        Args:
            prompt: Text prompt
            output_path: Path to save output image
            steps: Number of denoising steps
            height: Output height in pixels
            width: Output width in pixels
        """
        logger.info("Encoding prompt...")
        prompt_embeds = self.pipe.encode_prompt(
            prompt, num_videos_per_prompt=1, do_classifier_free_guidance=True
        )[0]

        # Prepare latents - shape: (B, C, F, H/scale, W/scale)
        latent_height = height // VAE_SCALE_FACTOR
        latent_width = width // VAE_SCALE_FACTOR
        latents = torch.randn(
            1, LATENT_CHANNELS, 1, latent_height, latent_width,
            device=self.device, dtype=self.default_dtype,
        )

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps)

        logger.info("Running Denoising Loop (Core ML)...")
        for t in scheduler.timesteps:
            # Prepare inputs for Core ML
            inputs = {
                "hidden_states": self.to_numpy(latents),
                "encoder_hidden_states": self.to_numpy(prompt_embeds),
                "timestep": np.array([t.item()]).astype(np.int32),
            }

            # Run Core ML Inference
            noise_pred = self.predict_coreml(self.coreml_transformer, inputs)

            # Scheduler Step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        logger.info("Decoding Latents (Core ML)...")
        latents_np = self.to_numpy(latents, dtype=np.float16)
        image_dict = self.coreml_vae.predict({"latents": latents_np})
        out_key = list(image_dict.keys())[0]
        image_np = image_dict[out_key]

        # Post-process: normalize from [-1, 1] to [0, 255]
        image_np = (image_np / 2 + 0.5).clip(0, 1)
        image_np = (image_np * 255).astype(np.uint8)

        # Convert from (1, 3, H, W) to (H, W, 3)
        if image_np.ndim == 4:
            image_np = image_np[0]
        image_np = np.transpose(image_np, (1, 2, 0))

        img = Image.fromarray(image_np)
        img.save(output_path)
        logger.info("Saved to %s", output_path)
