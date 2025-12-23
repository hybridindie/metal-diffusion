import os
from typing import Callable

from .base import TwoPhaseConverter
from alloy.converters.ltx_workers import convert_ltx_part1, convert_ltx_part2


class LTXConverter(TwoPhaseConverter):
    """
    Converter for LTX Video models using 2-phase subprocess isolation.
    Splits at the midpoint of transformer blocks.
    """

    def __init__(self, model_id, output_dir, quantization, hf_token=None):
        if "/" not in model_id and not os.path.isfile(model_id):
            model_id = "Lightricks/LTX-Video"
        super().__init__(model_id, output_dir, quantization, hf_token=hf_token)

    @property
    def model_name(self) -> str:
        return "LTX"

    def get_part1_worker(self) -> Callable:
        return convert_ltx_part1

    def get_part2_worker(self) -> Callable:
        return convert_ltx_part2
