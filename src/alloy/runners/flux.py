"""CoreML runner for Flux.1 models."""

import numpy as np
import os
import torch
from diffusers import DiffusionPipeline, FluxPipeline
from typing import Dict, Optional

try:
    from diffusers import Flux2Pipeline
except ImportError:
    Flux2Pipeline = None

from alloy.logging import get_logger
from alloy.runners.core import BaseCoreMLRunner

logger = get_logger(__name__)


class FluxCoreMLRunner(BaseCoreMLRunner):
    """
    Hybrid runner for Flux.1:
    - Text Encoders (CLIP+T5): PyTorch (CPU/MPS)
    - Transformer: Core ML
    - VAE: PyTorch (CPU/MPS)
    """

    @property
    def model_name(self) -> str:
        return "Flux"

    @property
    def transformer_filename(self) -> str:
        return "Flux_Transformer.mlpackage"

    @property
    def pipeline_class(self):
        return FluxPipeline

    @property
    def default_model_id(self) -> str:
        return "black-forest-labs/FLUX.1-schnell"

    @property
    def default_dtype(self) -> torch.dtype:
        """Use float16 on MPS, float32 on CPU."""
        return torch.float16 if self.device == "mps" else torch.float32

    def _load_pipeline(self) -> None:
        """Load Flux pipeline with Flux2 detection."""
        logger.info("Loading PyTorch components from %s...", self.model_id)

        if self.supports_single_file and os.path.isfile(self.model_id):
            logger.info("Detected single file checkpoint: %s", self.model_id)
            self.pipe = FluxPipeline.from_single_file(
                self.model_id,
                torch_dtype=self.default_dtype,
                transformer=None,
            ).to(self.device)
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.default_dtype,
                transformer=None,
            ).to(self.device)

        # Check if Flux 2
        self.is_flux2 = Flux2Pipeline and isinstance(self.pipe, Flux2Pipeline)
        if self.is_flux2:
            logger.info("Detected Flux.2 Pipeline")

    def _load_coreml_models(self) -> None:
        """Load CoreML transformer."""
        self.coreml_transformer = self._load_coreml_transformer()

    def generate(
        self,
        prompt: str,
        output_path: str,
        steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
        controlnet_residuals: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Generate image with Flux.

        Note: Flux Schnell uses 4 steps and guidance_scale=0.0 by default.
        Flux Dev uses roughly 20-50 steps and guidance 3.5.

        Args:
            prompt: Text prompt
            output_path: Path to save output image
            steps: Number of denoising steps
            height: Output height in pixels
            width: Output width in pixels
            guidance_scale: Guidance scale (0.0 for Schnell)
            seed: Random seed for reproducibility
            controlnet_residuals: Optional ControlNet residual dict
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # 1. Encode Prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
        )

        # 2. Prepare Latents & IDs
        num_channels_latents = self.pipe.vae.config.latent_channels  # 16

        # Adjust height/width for packing
        latent_height = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))

        logger.info("Generating latents for %dx%d...", latent_height, latent_width)

        # Init Random Latents
        latents = torch.randn(
            1, num_channels_latents, latent_height, latent_width,
            device=self.device, dtype=self.pipe.text_encoder.dtype, generator=generator,
        )

        # Pack
        latents = self._pack_latents(latents, 1, num_channels_latents, latent_height, latent_width)

        # Image IDs
        img_ids = self._prepare_latent_image_ids(
            1, latent_height // 2, latent_width // 2, self.device, self.pipe.text_encoder.dtype
        )

        # 3. Timesteps
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps, device=self.device)

        # 4. Denoising Loop
        logger.info("Running Denoising Loop (Core ML)...")

        # Guidance Tensor
        guidance_input = np.array([guidance_scale]).astype(np.float32)

        # Pre-convert constant inputs
        encoder_hidden_states_np = self.to_numpy(prompt_embeds)
        txt_ids_np = self.to_numpy(text_ids)
        img_ids_np = self.to_numpy(img_ids)

        # Pooled Projections (Flux 1 only)
        if pooled_prompt_embeds is not None:
            pooled_projections_np = self.to_numpy(pooled_prompt_embeds)

        for t in scheduler.timesteps:
            t_input = np.array([t.item()]).astype(np.float32)
            latents_input = self.to_numpy(latents)

            inputs = {
                "hidden_states": latents_input,
                "encoder_hidden_states": encoder_hidden_states_np,
                "timestep": t_input,
                "img_ids": img_ids_np,
                "txt_ids": txt_ids_np,
                "guidance": guidance_input,
            }

            if not self.is_flux2:
                inputs["pooled_projections"] = pooled_projections_np

            # Add ControlNet Residuals
            if controlnet_residuals:
                inputs.update(controlnet_residuals)

            # Predict
            noise_pred = self.predict_coreml(self.coreml_transformer, inputs)
            noise_pred = noise_pred.to(latents.dtype)

            # Step
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        logger.info("Decoding...")
        # Unpack
        latents = self._unpack_latents(latents, height, width, self.pipe.vae_scale_factor)

        # Decode
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        latents = latents.to(self.pipe.vae.dtype)
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")
        image[0].save(output_path)
        logger.info("Saved to %s", output_path)

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Pack 2D latents for Flux transformer input."""
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        height: int,
        width: int,
        vae_scale_factor: int,
    ) -> torch.Tensor:
        """Unpack Flux transformer output to 2D latents."""
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int,
        height: int,
        width: int,
        device: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare image position IDs for Flux attention."""
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids.to(device=device, dtype=dtype)
