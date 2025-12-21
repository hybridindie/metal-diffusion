"""Core ML VAE nodes for ComfyUI - accelerated encode/decode on Apple Silicon."""

import os
import json
import torch
import numpy as np
import coremltools as ct
import folder_paths


class CoreMLVAEWrapper:
    """
    Wrapper for Core ML VAE encoder/decoder pair.
    Holds the models and configuration for encode/decode operations.
    """

    def __init__(self, encoder_path=None, decoder_path=None, config=None):
        """
        Initialize VAE wrapper with Core ML models.

        Args:
            encoder_path: Path to VAE_Encoder.mlpackage (optional)
            decoder_path: Path to VAE_Decoder.mlpackage (optional)
            config: VAE configuration dict (scaling_factor, shift_factor, etc.)
        """
        self.encoder = None
        self.decoder = None
        self.config = config or {}

        if encoder_path and os.path.exists(encoder_path):
            self.encoder = ct.models.MLModel(encoder_path)

        if decoder_path and os.path.exists(decoder_path):
            self.decoder = ct.models.MLModel(decoder_path)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode image pixels to latent space.

        Args:
            pixels: Image tensor (B, C, H, W) in range [-1, 1]

        Returns:
            Latent tensor (B, latent_channels, H/8, W/8)
        """
        if self.encoder is None:
            raise RuntimeError("VAE encoder not loaded")

        # Convert to numpy for Core ML
        pixels_np = pixels.cpu().numpy().astype(np.float32)

        # Run Core ML inference
        outputs = self.encoder.predict({"pixels": pixels_np})

        # Extract output
        if "latents" in outputs:
            latents_np = outputs["latents"]
        else:
            latents_np = list(outputs.values())[0]

        # Convert back to tensor
        latents = torch.from_numpy(latents_np).to(pixels.device, dtype=pixels.dtype)

        # Apply scaling factor if configured
        scaling_factor = self.config.get("scaling_factor", 1.0)
        if scaling_factor != 1.0:
            latents = latents * scaling_factor

        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to image pixels.

        Args:
            latents: Latent tensor (B, latent_channels, H, W)

        Returns:
            Image tensor (B, C, H*8, W*8) in range [-1, 1]
        """
        if self.decoder is None:
            raise RuntimeError("VAE decoder not loaded")

        # Apply inverse scaling if configured
        scaling_factor = self.config.get("scaling_factor", 1.0)
        shift_factor = self.config.get("shift_factor", 0.0)

        if scaling_factor != 1.0 or shift_factor != 0.0:
            latents = (latents / scaling_factor) + shift_factor

        # Convert to numpy for Core ML
        latents_np = latents.cpu().numpy().astype(np.float32)

        # Run Core ML inference
        outputs = self.decoder.predict({"latents": latents_np})

        # Extract output
        if "sample" in outputs:
            sample_np = outputs["sample"]
        else:
            sample_np = list(outputs.values())[0]

        # Convert back to tensor
        sample = torch.from_numpy(sample_np).to(latents.device, dtype=latents.dtype)

        return sample


class CoreMLVAELoader:
    """
    Load Core ML VAE (.mlpackage) for use with encode/decode nodes.
    Supports loading encoder, decoder, or both.
    """

    @classmethod
    def INPUT_TYPES(cls):
        vae_files = folder_paths.get_filename_list("vae")
        vae_options = ["none"] + vae_files

        return {
            "required": {
                "decoder_path": (vae_options, {"default": "none"}),
            },
            "optional": {
                "encoder_path": (vae_options, {"default": "none"}),
                "config_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("COREML_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "Alloy"

    def load_vae(self, decoder_path, encoder_path="none", config_path=""):
        """Load Core ML VAE models and configuration."""
        encoder_full = None
        decoder_full = None
        config = {}

        # Resolve paths
        if decoder_path != "none":
            decoder_full = folder_paths.get_full_path("vae", decoder_path)
            print(f"[CoreMLVAELoader] Loading decoder: {decoder_full}")

        if encoder_path != "none":
            encoder_full = folder_paths.get_full_path("vae", encoder_path)
            print(f"[CoreMLVAELoader] Loading encoder: {encoder_full}")

        # Load config if provided or try to find it
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"[CoreMLVAELoader] Loaded config from: {config_path}")
        else:
            # Try to find config next to decoder or encoder
            for path in [decoder_full, encoder_full]:
                if path:
                    config_candidate = os.path.join(os.path.dirname(path), "vae_config.json")
                    if os.path.exists(config_candidate):
                        with open(config_candidate, "r") as f:
                            config = json.load(f)
                        print(f"[CoreMLVAELoader] Found config: {config_candidate}")
                        break

        # Create wrapper
        wrapper = CoreMLVAEWrapper(
            encoder_path=encoder_full,
            decoder_path=decoder_full,
            config=config,
        )

        print(f"[CoreMLVAELoader] VAE loaded (encoder: {encoder_full is not None}, decoder: {decoder_full is not None})")
        return (wrapper,)


class CoreMLVAEEncode:
    """
    Encode images to latents using Core ML VAE.
    Converts IMAGE (pixel space) to LATENT (compressed latent space).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("COREML_VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "Alloy"

    def encode(self, pixels, vae):
        """
        Encode image to latent space.

        Args:
            pixels: IMAGE tensor (B, H, W, C) with values in [0, 1]
            vae: CoreMLVAEWrapper instance

        Returns:
            LATENT dict with {"samples": tensor}
        """
        # Convert ComfyUI IMAGE format to VAE format
        # IMAGE: (B, H, W, C) [0, 1] -> VAE: (B, C, H, W) [-1, 1]
        x = pixels.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = x * 2.0 - 1.0  # [0, 1] -> [-1, 1]

        # Encode with Core ML VAE
        latents = vae.encode(x)

        return ({"samples": latents},)


class CoreMLVAEDecode:
    """
    Decode latents to images using Core ML VAE.
    Converts LATENT (compressed latent space) to IMAGE (pixel space).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("COREML_VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Alloy"

    def decode(self, samples, vae):
        """
        Decode latents to image.

        Args:
            samples: LATENT dict with {"samples": tensor (B, C, H, W)}
            vae: CoreMLVAEWrapper instance

        Returns:
            IMAGE tensor (B, H, W, C) with values in [0, 1]
        """
        latents = samples["samples"]

        # Decode with Core ML VAE
        decoded = vae.decode(latents)

        # Convert VAE output to ComfyUI IMAGE format
        # VAE: (B, C, H, W) [-1, 1] -> IMAGE: (B, H, W, C) [0, 1]
        image = (decoded / 2.0 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
        image = image.permute(0, 2, 3, 1)  # (B, H, W, C)

        return (image,)


class CoreMLVAETile:
    """
    Decode latents to images using tiled Core ML VAE for large images.
    Helps avoid memory issues with high-resolution outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("COREML_VAE",),
                "tile_size": ("INT", {"default": 64, "min": 32, "max": 128, "step": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_tiled"
    CATEGORY = "Alloy"

    def decode_tiled(self, samples, vae, tile_size):
        """
        Decode latents to image using tiles for memory efficiency.

        Args:
            samples: LATENT dict with {"samples": tensor (B, C, H, W)}
            vae: CoreMLVAEWrapper instance
            tile_size: Size of tiles in latent space

        Returns:
            IMAGE tensor (B, H, W, C) with values in [0, 1]
        """
        latents = samples["samples"]
        B, C, H, W = latents.shape

        # If small enough, just decode directly
        if H <= tile_size and W <= tile_size:
            decoded = vae.decode(latents)
            image = (decoded / 2.0 + 0.5).clamp(0, 1)
            image = image.permute(0, 2, 3, 1)
            return (image,)

        # Tiled decoding
        # VAE typically upscales by 8x
        scale = 8
        output_h = H * scale
        output_w = W * scale
        output = torch.zeros(B, 3, output_h, output_w, device=latents.device, dtype=latents.dtype)

        # Overlap for blending
        overlap = tile_size // 4

        for y in range(0, H, tile_size - overlap):
            for x in range(0, W, tile_size - overlap):
                # Get tile bounds
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                # Extract and decode tile
                tile = latents[:, :, y_start:y_end, x_start:x_end]
                decoded_tile = vae.decode(tile)

                # Place in output (simple overwrite, could add blending)
                out_y_start = y_start * scale
                out_y_end = y_end * scale
                out_x_start = x_start * scale
                out_x_end = x_end * scale
                output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = decoded_tile

        # Convert to ComfyUI format
        image = (output / 2.0 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1)

        return (image,)


NODE_CLASS_MAPPINGS = {
    "CoreMLVAELoader": CoreMLVAELoader,
    "CoreMLVAEEncode": CoreMLVAEEncode,
    "CoreMLVAEDecode": CoreMLVAEDecode,
    "CoreMLVAETile": CoreMLVAETile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLVAELoader": "Core ML VAE Loader",
    "CoreMLVAEEncode": "Core ML VAE Encode",
    "CoreMLVAEDecode": "Core ML VAE Decode",
    "CoreMLVAETile": "Core ML VAE Decode (Tiled)",
}
