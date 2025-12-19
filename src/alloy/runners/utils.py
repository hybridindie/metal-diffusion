"""Shared utilities for CoreML runners."""

import numpy as np
import torch
from typing import Tuple


def apply_classifier_free_guidance(
    noise_uncond: torch.Tensor,
    noise_text: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Apply classifier-free guidance.

    Args:
        noise_uncond: Unconditional noise prediction
        noise_text: Text-conditioned noise prediction
        guidance_scale: CFG scale factor

    Returns:
        Guided noise prediction
    """
    return noise_uncond + guidance_scale * (noise_text - noise_uncond)


def make_timestep_array(
    timestep: torch.Tensor,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Convert timestep tensor to numpy array for CoreML.

    Args:
        timestep: Scalar timestep tensor
        dtype: Target numpy dtype (float32 for Flux/Lumina, int32 for LTX/Hunyuan)

    Returns:
        1-element numpy array
    """
    return np.array([timestep.item()]).astype(dtype)


def prepare_latents(
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
    vae_scale_factor: int,
    device: str,
    dtype: torch.dtype,
    generator: torch.Generator = None,
    num_frames: int = 1,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Create random latents with correct shape.

    Args:
        batch_size: Batch size (usually 1)
        num_channels: Latent channel count
        height: Output height in pixels
        width: Output width in pixels
        vae_scale_factor: VAE spatial downscale factor
        device: Target device
        dtype: Target dtype
        generator: Optional random generator for reproducibility
        num_frames: Number of video frames (1 for images)

    Returns:
        Tuple of (latents, latent_height, latent_width, latent_frames)
    """
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    latent_frames = num_frames

    if num_frames > 1:
        # Video latent shape: (B, C, F, H, W)
        shape = (batch_size, num_channels, latent_frames, latent_height, latent_width)
    else:
        # Image latent shape: (B, C, H, W)
        shape = (batch_size, num_channels, latent_height, latent_width)

    latents = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return latents, latent_height, latent_width, latent_frames
