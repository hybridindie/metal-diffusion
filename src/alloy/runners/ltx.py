"""CoreML runner for LTX-Video models."""

import numpy as np
import os
import torch
from diffusers import LTXPipeline

from alloy.logging import get_logger
from alloy.runners.core import BaseCoreMLRunner
from alloy.runners.utils import apply_classifier_free_guidance

logger = get_logger(__name__)


class LTXCoreMLRunner(BaseCoreMLRunner):
    """
    Hybrid runner for LTX-Video:
    - Text Encoder: PyTorch (CPU/MPS)
    - Transformer: Core ML
    - VAE: PyTorch (CPU/MPS)
    """

    @property
    def model_name(self) -> str:
        return "LTX-Video"

    @property
    def transformer_filename(self) -> str:
        return "LTXVideo_Transformer.mlpackage"

    @property
    def pipeline_class(self):
        return LTXPipeline

    @property
    def default_model_id(self) -> str:
        return "Lightricks/LTX-Video"

    def _load_pipeline(self) -> None:
        """Load LTX pipeline with single-file support."""
        logger.info("Loading PyTorch components from %s...", self.model_id)

        if self.supports_single_file and os.path.isfile(self.model_id):
            logger.info("Detected single file checkpoint: %s", self.model_id)
            self.pipe = self.pipeline_class.from_single_file(
                self.model_id,
                torch_dtype=self.default_dtype,
            ).to(self.device)
        else:
            self.pipe = self.pipeline_class.from_pretrained(
                self.model_id,
                torch_dtype=self.default_dtype,
            ).to(self.device)

    def _load_coreml_models(self) -> None:
        """Load CoreML transformer."""
        self.coreml_transformer = self._load_coreml_transformer()

    def generate(
        self,
        prompt: str,
        output_path: str,
        steps: int = 20,
        height: int = 512,
        width: int = 512,
        num_frames: int = 8,
        guidance_scale: float = 3.0,
    ) -> None:
        """
        Generate video frame with LTX-Video.

        Args:
            prompt: Text prompt
            output_path: Path to save output image
            steps: Number of denoising steps
            height: Output height in pixels
            width: Output width in pixels
            num_frames: Number of video frames
            guidance_scale: Classifier-free guidance scale
        """
        logger.info("Encoding prompt...")
        prompt_embeds, prompt_attention_mask = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=True,
        )  # (2, L, 4096), (2, L)

        num_channels = 128
        patch_size = 1
        patch_size_t = 1

        vae_spatial_compression = 32
        vae_temporal_compression = 8

        latent_height = height // vae_spatial_compression
        latent_width = width // vae_spatial_compression
        latent_frames = (num_frames - 1) // vae_temporal_compression + 1

        # 1. Random Init (B=1)
        latents = torch.randn(
            1, 128, latent_frames, latent_height, latent_width,
            device=self.device, dtype=self.default_dtype,
        )

        # 2. Pack
        latents = self._pack_latents(latents, patch_size, patch_size_t)

        # 3. Timesteps
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps)

        logger.info("Running Denoising Loop (Core ML)...")
        for t in scheduler.timesteps:
            # Input Prep
            latents_input = self.to_numpy(latents)
            timestep_np = np.array([t.item()]).astype(np.int32)

            # 1. Uncond Pass
            inputs_uncond = {
                "hidden_states": latents_input,
                "encoder_hidden_states": self.to_numpy(prompt_embeds[0:1]),
                "timestep": timestep_np,
                "encoder_attention_mask": self.to_numpy(prompt_attention_mask[0:1], dtype=np.int64),
                "num_frames": np.array([latent_frames]).astype(np.int32),
                "height": np.array([latent_height]).astype(np.int32),
                "width": np.array([latent_width]).astype(np.int32),
            }
            noise_uncond = self.predict_coreml(
                self.coreml_transformer, inputs_uncond
            ).to(dtype=self.default_dtype)

            # 2. Text Pass
            inputs_text = {
                "hidden_states": latents_input,
                "encoder_hidden_states": self.to_numpy(prompt_embeds[1:2]),
                "timestep": timestep_np,
                "encoder_attention_mask": self.to_numpy(prompt_attention_mask[1:2], dtype=np.int64),
                "num_frames": np.array([latent_frames]).astype(np.int32),
                "height": np.array([latent_height]).astype(np.int32),
                "width": np.array([latent_width]).astype(np.int32),
            }
            noise_text = self.predict_coreml(
                self.coreml_transformer, inputs_text
            ).to(dtype=self.default_dtype)

            # Apply CFG
            noise_pred = apply_classifier_free_guidance(
                noise_uncond, noise_text, guidance_scale
            )

            # Step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        logger.info("Decoding (PyTorch)...")
        # Unpack
        latents = self._unpack_latents(
            latents, latent_frames, latent_height, latent_width, patch_size, patch_size_t
        )

        # Denormalize
        latents = self._denormalize_latents(
            latents,
            self.pipe.vae.latents_mean,
            self.pipe.vae.latents_std,
            self.pipe.vae.config.scaling_factor,
        )

        # Decode
        latents = latents.to(dtype=self.default_dtype)
        with torch.no_grad():
            video = self.pipe.vae.decode(latents, return_dict=False)[0]

        # Post process video -> Image (first frame)
        video = self.pipe.video_processor.postprocess_video(video, output_type="pil")
        video[0][0].save(output_path)
        logger.info("Saved to %s", output_path)

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        """Pack video latents for transformer input."""
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        """Unpack transformer output to video latents."""
        batch_size = latents.size(0)
        latents = latents.reshape(
            batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size
        )
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        """Denormalize latents for VAE decoding."""
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents
