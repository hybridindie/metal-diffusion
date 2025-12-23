"""Shared utilities for Alloy ComfyUI nodes."""

import os
import folder_paths


# Flux model constants
FLUX_LATENT_CHANNELS = 16
FLUX_HIDDEN_DIM = 64  # FLUX_LATENT_CHANNELS * 4 (from 2x2 packing)
CLIP_L_POOLED_DIM = 768
T5_HIDDEN_DIM = 4096
T5_MAX_SEQ_LEN = 256
CLIP_L_MAX_SEQ_LEN = 77


def find_mlpackage_files(folder_type: str = "unet") -> list:
    """
    Recursively find all .mlpackage files in ComfyUI's model folders.

    Args:
        folder_type: ComfyUI folder type ("unet", "vae", "controlnet", etc.)

    Returns:
        List of relative paths that can be resolved with folder_paths.get_full_path()
        or resolve_model_path().
    """
    mlpackages = []
    try:
        folders = folder_paths.get_folder_paths(folder_type)
        for base_folder in folders:
            if not os.path.exists(base_folder):
                continue
            for root, dirs, files in os.walk(base_folder):
                # Check if this directory itself is an .mlpackage (they're directories)
                for d in dirs:
                    if d.endswith(".mlpackage"):
                        full_path = os.path.join(root, d)
                        # Get relative path from base folder
                        rel_path = os.path.relpath(full_path, base_folder)
                        mlpackages.append(rel_path)
    except Exception:
        # Silently fail if folder_paths not available (e.g., during testing)
        pass
    return mlpackages


def resolve_model_path(folder_type: str, model_name: str) -> str:
    """
    Resolve a model name to its full path.

    Handles both files and directories (.mlpackage), unlike folder_paths.get_full_path()
    which only works for files.

    Args:
        folder_type: ComfyUI folder type ("unet", "vae", "controlnet", etc.)
        model_name: Relative model name/path

    Returns:
        Full path to the model

    Raises:
        ValueError: If the model cannot be found
    """
    # First try the standard folder_paths method (works for files)
    full_path = folder_paths.get_full_path(folder_type, model_name)
    if full_path is not None and os.path.exists(full_path):
        return full_path

    # For directories (.mlpackage), search manually
    for folder in folder_paths.get_folder_paths(folder_type):
        candidate = os.path.join(folder, model_name)
        if os.path.exists(candidate):
            return candidate

    raise ValueError(
        f"Model not found: {model_name}. "
        f"Check that the file exists in ComfyUI's {folder_type} folder."
    )
