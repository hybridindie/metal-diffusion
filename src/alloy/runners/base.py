import os
import re

import coremltools as ct
import numpy as np
import torch
from diffusers import DiffusionPipeline, WanPipeline
from PIL import Image
from python_coreml_stable_diffusion.pipeline import get_coreml_pipe

from alloy.logging import get_logger

logger = get_logger(__name__)

# Latent space constants
LATENT_CHANNELS = 16  # Number of channels in the latent space
VAE_SCALE_FACTOR = 8  # VAE downscaling factor (height/width divided by this)

# SD Turbo inference defaults
SD_TURBO_STEPS = 2
SD_TURBO_GUIDANCE_SCALE = 2.0


def run_sd_pipeline(model_dir, prompt, output_path, compute_unit="ALL", base_model="stabilityai/sd-turbo"):
    """
    Runs a Stable Diffusion Core ML pipeline.
    """
    logger.info("Loading Base Pipeline (for Tokenizer/Scheduler) from %s...", base_model)
    pytorch_pipe = DiffusionPipeline.from_pretrained(base_model)

    logger.info("Loading Core ML Pipeline from %s...", model_dir)

    # Auto-detect model_version from filenames in model_dir
    # Format: Stable_Diffusion_version_{version}_{component}.mlpackage
    model_version_str = base_model.replace("/", "_")  # fallback

    try:
        files = os.listdir(model_dir)
        unet_files = [f for f in files if "unet.mlpackage" in f]
        if unet_files:
            match = re.search(r"Stable_Diffusion_version_(.*)_unet.mlpackage", unet_files[0])
            if match:
                model_version_str = match.group(1)
                logger.debug("Detected model version string: %s", model_version_str)
    except Exception as e:
        logger.warning("Could not auto-detect model version from files: %s", e)

    pipeline = get_coreml_pipe(
        pytorch_pipe=pytorch_pipe,
        mlpackages_dir=model_dir,
        model_version=model_version_str,
        compute_unit=compute_unit
    )

    logger.info("Generating image for prompt: '%s'", prompt)
    image = pipeline(
        prompt=prompt,
        num_inference_steps=SD_TURBO_STEPS,
        guidance_scale=SD_TURBO_GUIDANCE_SCALE
    )["images"][0]

    image.save(output_path)
    logger.info("Saved to %s", output_path)

class WanCoreMLRunner:
    """
    Hybrid runner for Wan 2.1:
    - Text Encoder: PyTorch (CPU/MPS)
    - Transformer: Core ML
    - VAE: Core ML
    """

    def __init__(self, model_dir, model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers"):
        self.model_dir = model_dir
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        logger.info("Loading PyTorch components (Text Encoder) from %s...", model_id)
        self.pipe = WanPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to(self.device)

        logger.info("Loading Core ML Transformer...")
        self.coreml_transformer = ct.models.MLModel(
            os.path.join(model_dir, "Wan2.1_Transformer.mlpackage")
        )

        logger.info("Loading Core ML VAE...")
        self.coreml_vae = ct.models.MLModel(
            os.path.join(model_dir, "Wan2.1_VAE_Decoder.mlpackage")
        )

    def generate(self, prompt, output_path, steps=20, height=512, width=512):
        logger.info("Encoding prompt...")
        prompt_embeds = self.pipe.encode_prompt(
            prompt, num_videos_per_prompt=1, do_classifier_free_guidance=True
        )[0]

        # Prepare latents - shape: (B, C, F, H/scale, W/scale)
        latent_height = height // VAE_SCALE_FACTOR
        latent_width = width // VAE_SCALE_FACTOR
        latents = torch.randn(
            1, LATENT_CHANNELS, 1, latent_height, latent_width,
            device=self.device, dtype=torch.float16
        )
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps)

        logger.info("Running Denoising Loop (Core ML)...")
        for t in scheduler.timesteps:
            # Prepare inputs for Core ML
            latent_np = latents.cpu().numpy().astype(np.float32)
            timestep_np = np.array([t.item()]).astype(np.int32)
            encoder_hidden_states_np = prompt_embeds.cpu().numpy().astype(np.float32)

            inputs = {
                "hidden_states": latent_np,
                "encoder_hidden_states": encoder_hidden_states_np,
                "timestep": timestep_np
            }

            # Run Core ML Inference
            pred_dict = self.coreml_transformer.predict(inputs)
            noise_pred = torch.from_numpy(pred_dict["sample"]).to(self.device)

            # Scheduler Step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        logger.info("Decoding Latents (Core ML)...")
        latents_np = latents.cpu().numpy().astype(np.float16)
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
