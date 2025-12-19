"""Shared utilities for CoreML runners."""

import torch


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
