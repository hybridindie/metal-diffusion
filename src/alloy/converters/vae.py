"""VAE Converter for Core ML - supports encoder and decoder conversion."""

import os
import json
import torch
import torch.nn as nn
import coremltools as ct
from typing import Optional, List, Dict, Any
from diffusers import AutoencoderKL, DiffusionPipeline
from rich.console import Console

from .base import ModelConverter
from alloy.utils.coreml import safe_quantize_model

console = Console()

# VAE configurations for different model types
VAE_CONFIGS = {
    "flux": {
        "latent_channels": 16,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "sample_size": 64,  # Latent size for tracing
    },
    "sdxl": {
        "latent_channels": 4,
        "scaling_factor": 0.13025,
        "shift_factor": 0.0,
        "sample_size": 64,
    },
    "sd": {
        "latent_channels": 4,
        "scaling_factor": 0.18215,
        "shift_factor": 0.0,
        "sample_size": 64,
    },
    "wan": {
        "latent_channels": 16,
        "scaling_factor": 1.0,
        "shift_factor": 0.0,
        "sample_size": 64,
        "is_video": True,
    },
    "ltx": {
        "latent_channels": 128,
        "scaling_factor": 1.0,
        "shift_factor": 0.0,
        "sample_size": 32,
        "is_video": True,
    },
    "hunyuan": {
        "latent_channels": 16,
        "scaling_factor": 0.18215,
        "shift_factor": 0.0,
        "sample_size": 64,
        "is_video": True,
    },
}


class VAEEncoderWrapper(nn.Module):
    """Wrapper for VAE encoder to return just the latent sample."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        # Encode returns a distribution, we sample from it
        # For tracing, we need deterministic output, so use mean
        encoded = self.vae.encode(x)
        if hasattr(encoded, "latent_dist"):
            return encoded.latent_dist.mean
        return encoded.latents


class VAEDecoderWrapper(nn.Module):
    """Wrapper for VAE decoder to handle return_dict=False."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        decoded = self.vae.decode(z, return_dict=False)
        if isinstance(decoded, tuple):
            return decoded[0]
        return decoded


class VAEConverter(ModelConverter):
    """
    Converter for VAE models (encoder and/or decoder).
    Supports both image (4D) and video (5D) VAEs.
    """

    def __init__(
        self,
        model_id: str,
        output_dir: str,
        quantization: str = "float16",
        vae_type: str = "auto",
        components: Optional[List[str]] = None,
    ):
        """
        Initialize VAE converter.

        Args:
            model_id: HuggingFace repo ID, local path, or pipeline ID containing VAE
            output_dir: Output directory for converted models
            quantization: Quantization type ("float16", "int8", "int4")
            vae_type: VAE architecture type ("flux", "sdxl", "sd", "wan", "ltx", "hunyuan", "auto")
            components: List of components to convert (["encoder", "decoder"])
        """
        super().__init__(model_id, output_dir, quantization)
        self.vae_type = vae_type
        self.components = components or ["encoder", "decoder"]

    def _detect_vae_type(self, vae) -> str:
        """Auto-detect VAE type from model config."""
        latent_channels = getattr(vae.config, "latent_channels", 4)

        # Check for Flux-specific config
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor > 0:
            return "flux"

        # Detect by latent channels
        if latent_channels == 128:
            return "ltx"
        elif latent_channels == 16:
            # Could be Flux, Wan, or Hunyuan
            scaling = getattr(vae.config, "scaling_factor", 0.18215)
            if abs(scaling - 0.3611) < 0.01:
                return "flux"
            return "hunyuan"  # Default for 16-channel
        elif latent_channels == 4:
            scaling = getattr(vae.config, "scaling_factor", 0.18215)
            if abs(scaling - 0.13025) < 0.01:
                return "sdxl"
            return "sd"

        return "sd"  # Fallback

    def _get_vae_config(self, vae, vae_type: str) -> Dict[str, Any]:
        """Get VAE configuration for conversion and runtime."""
        base_config = VAE_CONFIGS.get(vae_type, VAE_CONFIGS["sd"]).copy()

        # Override with actual values from model if available
        if hasattr(vae.config, "latent_channels"):
            base_config["latent_channels"] = vae.config.latent_channels
        if hasattr(vae.config, "scaling_factor"):
            base_config["scaling_factor"] = vae.config.scaling_factor
        if hasattr(vae.config, "shift_factor"):
            base_config["shift_factor"] = vae.config.shift_factor

        return base_config

    def _load_vae(self):
        """Load VAE from model ID or pipeline."""
        console.print(f"[cyan]Loading VAE from:[/cyan] {self.model_id}")

        # Try loading as standalone VAE first
        try:
            vae = AutoencoderKL.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
            )
            console.print("[green]✓[/green] Loaded as standalone VAE")
            return vae
        except Exception:
            pass

        # Try loading VAE subfolder
        try:
            vae = AutoencoderKL.from_pretrained(
                self.model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
            )
            console.print("[green]✓[/green] Loaded VAE from subfolder")
            return vae
        except Exception:
            pass

        # Try loading from full pipeline
        try:
            console.print("Loading from pipeline (this may take a moment)...")
            pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,
            )
            vae = pipe.vae
            del pipe
            console.print("[green]✓[/green] Extracted VAE from pipeline")
            return vae
        except Exception as e:
            raise ValueError(f"Could not load VAE from {self.model_id}: {e}")

    def convert(self, show_progress: bool = True):
        """Convert VAE encoder and/or decoder to Core ML."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Load VAE
        vae = self._load_vae()
        vae.eval()

        # Detect or validate VAE type
        if self.vae_type == "auto":
            self.vae_type = self._detect_vae_type(vae)
            console.print(f"[cyan]Detected VAE type:[/cyan] {self.vae_type}")

        # Get configuration
        vae_config = self._get_vae_config(vae, self.vae_type)
        console.print(f"[dim]Config: {vae_config}[/dim]")

        # Save config for runtime use
        config_path = os.path.join(self.output_dir, "vae_config.json")
        with open(config_path, "w") as f:
            json.dump(vae_config, f, indent=2)
        console.print(f"[dim]Saved config to {config_path}[/dim]")

        # Convert requested components
        if "decoder" in self.components:
            decoder_path = os.path.join(self.output_dir, "VAE_Decoder.mlpackage")
            self._convert_decoder(vae, vae_config, decoder_path)

        if "encoder" in self.components:
            encoder_path = os.path.join(self.output_dir, "VAE_Encoder.mlpackage")
            self._convert_encoder(vae, vae_config, encoder_path)

        console.print(f"\n[green]✓ VAE conversion complete![/green]")
        console.print(f"  Output: {self.output_dir}")

    def _convert_decoder(self, vae, config: Dict[str, Any], output_path: str):
        """Convert VAE decoder (latent -> image)."""
        console.print("\n[cyan]Converting VAE Decoder...[/cyan]")

        latent_channels = config["latent_channels"]
        sample_size = config["sample_size"]
        is_video = config.get("is_video", False)

        # Create sample input
        if is_video:
            # Video VAE: (B, C, F, H, W)
            sample_input = torch.randn(1, latent_channels, 1, sample_size, sample_size).float()
        else:
            # Image VAE: (B, C, H, W)
            sample_input = torch.randn(1, latent_channels, sample_size, sample_size).float()

        console.print(f"  Input shape: {list(sample_input.shape)}")

        # Wrap and trace
        wrapper = VAEDecoderWrapper(vae)
        wrapper.eval()

        console.print("  Tracing decoder...")
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, sample_input, strict=False)

        # Convert to Core ML
        console.print("  Converting to Core ML...")
        ml_inputs = [ct.TensorType(name="latents", shape=sample_input.shape)]
        ml_outputs = [ct.TensorType(name="sample")]

        ml_model = ct.convert(
            traced,
            inputs=ml_inputs,
            outputs=ml_outputs,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14,
        )

        # Quantize if requested
        if self.quantization in ["int4", "4bit", "mixed", "int8", "8bit"]:
            console.print(f"  Quantizing to {self.quantization}...")
            ml_model = safe_quantize_model(ml_model, self.quantization)

        # Save
        ml_model.save(output_path)
        console.print(f"[green]  ✓ Decoder saved:[/green] {output_path}")

    def _convert_encoder(self, vae, config: Dict[str, Any], output_path: str):
        """Convert VAE encoder (image -> latent)."""
        console.print("\n[cyan]Converting VAE Encoder...[/cyan]")

        sample_size = config["sample_size"]
        is_video = config.get("is_video", False)

        # Create sample input (image/video in pixel space)
        # VAE downscales by 8x typically, so input is 8x larger
        pixel_size = sample_size * 8

        if is_video:
            # Video VAE: (B, C, F, H, W)
            sample_input = torch.randn(1, 3, 1, pixel_size, pixel_size).float()
        else:
            # Image VAE: (B, C, H, W)
            sample_input = torch.randn(1, 3, pixel_size, pixel_size).float()

        console.print(f"  Input shape: {list(sample_input.shape)}")

        # Wrap and trace
        wrapper = VAEEncoderWrapper(vae)
        wrapper.eval()

        console.print("  Tracing encoder...")
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, sample_input, strict=False)

        # Convert to Core ML
        console.print("  Converting to Core ML...")
        ml_inputs = [ct.TensorType(name="pixels", shape=sample_input.shape)]
        ml_outputs = [ct.TensorType(name="latents")]

        ml_model = ct.convert(
            traced,
            inputs=ml_inputs,
            outputs=ml_outputs,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14,
        )

        # Quantize if requested
        if self.quantization in ["int4", "4bit", "mixed", "int8", "8bit"]:
            console.print(f"  Quantizing to {self.quantization}...")
            ml_model = safe_quantize_model(ml_model, self.quantization)

        # Save
        ml_model.save(output_path)
        console.print(f"[green]  ✓ Encoder saved:[/green] {output_path}")
