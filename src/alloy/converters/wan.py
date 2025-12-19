import os
from typing import Callable

import torch
import coremltools as ct
from rich.console import Console

from .base import TwoPhaseConverter
from alloy.converters.wan_workers import convert_wan_part1, convert_wan_part2
from alloy.exceptions import UnsupportedModelError

console = Console()


class WanConverter(TwoPhaseConverter):
    """
    Converter for Wan 2.1 models using 2-phase subprocess isolation.
    Splits at the midpoint of 40 transformer blocks.
    """

    @property
    def model_name(self) -> str:
        return "Wan"

    @property
    def output_filename(self) -> str:
        return f"Wan2.1_Transformer_{self.quantization}.mlpackage"

    @property
    def should_download_source(self) -> bool:
        return False  # Wan uses custom download patterns

    def get_part1_worker(self) -> Callable:
        return convert_wan_part1

    def get_part2_worker(self) -> Callable:
        return convert_wan_part2

    def convert(self, show_progress: bool = True):
        """Override to handle single-file check and custom download patterns.

        Args:
            show_progress: Whether to show the Rich progress display (default: True)
        """
        # Single file not supported for Wan
        if os.path.isfile(self.model_id):
            raise UnsupportedModelError(
                "Single file loading not supported. Provide a HuggingFace model ID or local directory.",
                model_name="Wan",
                model_type="single_file"
            )

        # Download with custom patterns before standard conversion
        self.model_id = self.download_source_weights(
            self.model_id,
            self.output_dir,
            allow_patterns=["transformer/*", "vae/*", "text_encoder/*", "config.json", "*.json", "*.safetensors"]
        )

        # Continue with standard 2-phase conversion
        super().convert(show_progress=show_progress)

    def convert_vae(self, vae, output_dir):
        """Convert VAE Decoder (optional, can be done separately)."""
        console.print("Converting VAE Decoder...")
        vae.eval()
        latents = torch.randn(1, 16, 1, 128, 128).half()
        traced_vae = torch.jit.trace(vae.decode, latents)

        model = ct.convert(
            traced_vae,
            inputs=[ct.TensorType(name="latents", shape=latents.shape)],
            minimum_deployment_target=ct.target.macOS14
        )
        model.save(os.path.join(output_dir, "Wan2.1_VAE_Decoder.mlpackage"))

    def convert_text_encoder(self, text_encoder, output_dir):
        """Convert Text Encoder (optional, T5 is large)."""
        console.print("Skipping Text Encoder (T5 is large, use standard if available).")
        pass
