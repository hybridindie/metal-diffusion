"""Legacy SD pipeline runner for backward compatibility."""

import os
import re

from diffusers import DiffusionPipeline

from alloy.logging import get_logger

logger = get_logger(__name__)

# SD Turbo inference defaults
SD_TURBO_STEPS = 2
SD_TURBO_GUIDANCE_SCALE = 2.0


def run_sd_pipeline(model_dir, prompt, output_path, compute_unit="ALL", base_model="stabilityai/sd-turbo"):
    """
    Runs a Stable Diffusion Core ML pipeline.

    Args:
        model_dir: Directory containing CoreML model packages
        prompt: Text prompt for generation
        output_path: Path to save output image
        compute_unit: CoreML compute unit
        base_model: Base model for tokenizer/scheduler
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

    # Lazy import - optional dependency for SD support
    from python_coreml_stable_diffusion.pipeline import get_coreml_pipe

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
