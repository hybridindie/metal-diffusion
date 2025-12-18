import os
from typing import Callable

from .base import TwoPhaseConverter
from alloy.converters.workers import convert_flux_part1, convert_flux_part2

# Flux Architecture Constants
NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38


class FluxConverter(TwoPhaseConverter):
    """
    Converter for Flux models using 2-phase subprocess isolation.
    """

    def __init__(self, model_id, output_dir, quantization, loras=None, controlnet_compatible=False):
        if "/" not in model_id and not os.path.isfile(model_id):
            model_id = "black-forest-labs/FLUX.1-schnell"
        super().__init__(model_id, output_dir, quantization)
        self.loras = loras or []
        self.controlnet_compatible = controlnet_compatible

    @property
    def model_name(self) -> str:
        return "Flux"

    @property
    def should_download_source(self) -> bool:
        return False  # Flux handles model loading in workers

    def get_part1_worker(self) -> Callable:
        return convert_flux_part1

    def get_part2_worker(self) -> Callable:
        return convert_flux_part2

    def apply_loras(self, pipe):
        """Apply LoRAs to pipeline (note: not used with subprocess workers)."""
        return pipe
