"""CoreML runner for HunyuanVideo models."""

import numpy as np
import torch
from diffusers import HunyuanVideoPipeline

from alloy.logging import get_logger
from alloy.runners.core import BaseCoreMLRunner

logger = get_logger(__name__)


class HunyuanCoreMLRunner(BaseCoreMLRunner):
    """
    Hybrid runner for HunyuanVideo:
    - Text Encoder: PyTorch (CPU/MPS)
    - Transformer: Core ML
    - VAE: PyTorch (CPU/MPS)
    """

    @property
    def model_name(self) -> str:
        return "Hunyuan"

    @property
    def transformer_filename(self) -> str:
        return "HunyuanVideo_Transformer.mlpackage"

    @property
    def pipeline_class(self):
        return HunyuanVideoPipeline

    @property
    def default_model_id(self) -> str:
        return "hunyuanvideo-community/HunyuanVideo"

    @property
    def supports_single_file(self) -> bool:
        return False  # Hunyuan doesn't support single-file loading

    def _load_pipeline(self) -> None:
        """Load Hunyuan pipeline."""
        logger.info("Loading PyTorch components from %s...", self.model_id)
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
    ) -> None:
        """
        Generate image with HunyuanVideo.

        Args:
            prompt: Text prompt
            output_path: Path to save output image
            steps: Number of denoising steps
            height: Output height in pixels
            width: Output width in pixels
        """
        logger.info("Encoding prompt...")
        prompt_embeds, pooled_prompt_embeds, attention_mask = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        # Prepare latents
        num_channels = 16
        num_frames = 1
        latents = torch.randn(
            1, num_channels, num_frames, height // 16, width // 16,
            device=self.device, dtype=self.default_dtype,
        )

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps)

        # Guidance
        guidance_scale = 6.0
        guidance = torch.tensor([guidance_scale * 1000]).float()

        logger.info("Running Denoising Loop (Core ML)...")
        for t in scheduler.timesteps:
            # Prepare inputs for Core ML (batch 1 - positive prompt only)
            inputs = {
                "hidden_states": self.to_numpy(latents),
                "timestep": np.array([t.item()]).astype(np.int32),
                "encoder_hidden_states": self.to_numpy(prompt_embeds[1:2]),  # Use positive
                "encoder_attention_mask": self.to_numpy(attention_mask[1:2], dtype=np.int64),
                "pooled_projections": self.to_numpy(pooled_prompt_embeds[1:2]),
                "guidance": self.to_numpy(guidance),
            }

            noise_pred = self.predict_coreml(self.coreml_transformer, inputs)

            # Scheduler Step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        logger.info("Decoding Latents (PyTorch)...")
        latents = latents.to(dtype=self.default_dtype)
        with torch.no_grad():
            image = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor,
                return_dict=False,
            )[0]

        # Post-process - handle video tensor (B, C, F, H, W)
        if image.ndim == 5:
            image = image[:, :, 0, :, :]  # Take first frame

        image = self.pipe.image_processor.postprocess(image, output_type="pil")
        image[0].save(output_path)
        logger.info("Saved to %s", output_path)
