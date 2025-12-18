import os
import logging
from typing import Callable

from .base import TwoPhaseConverter
from alloy.converters.lumina_workers import convert_lumina_part1, convert_lumina_part2

logger = logging.getLogger(__name__)


class LuminaConverter(TwoPhaseConverter):
    """
    Converter for Lumina-Image 2.0 models (Next-Gen DiT).
    Uses 2-phase subprocess isolation (split at midpoint of blocks) to prevent OOM.
    Uses Gemma-2B as text encoder and Lumina2Transformer2DModel.
    """

    def __init__(self, model_id: str, output_dir: str, quantization: str = "float16",
                 img_height: int = 1024, img_width: int = 1024):
        super().__init__(model_id, output_dir, quantization)
        self.img_height = img_height
        self.img_width = img_width

    @property
    def model_name(self) -> str:
        return "Lumina"

    @property
    def output_filename(self) -> str:
        return f"Lumina2_Transformer_{self.quantization}.mlpackage"

    @property
    def should_download_source(self) -> bool:
        return False  # Lumina handles download in convert() with custom logger

    def get_part1_worker(self) -> Callable:
        return convert_lumina_part1

    def get_part2_worker(self) -> Callable:
        return convert_lumina_part2

    def convert(self):
        """Override to handle single-file check and custom download."""
        # Single file not supported for Lumina
        if os.path.isfile(self.model_id):
            logger.error("Single file loading is not supported for Lumina-Image 2.0.")
            return

        # Download source weights with custom logger
        self.model_id = self.download_source_weights(
            self.model_id,
            self.output_dir,
            logger_fn=logger.info
        )

        super().convert()

    def convert_vae(self, vae):
        """VAE conversion (optional, reuse standard VAE converter)."""
        pass
