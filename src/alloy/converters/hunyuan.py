from typing import Callable

from .base import TwoPhaseConverter
from alloy.converters.hunyuan_workers import convert_hunyuan_part1, convert_hunyuan_part2


class HunyuanConverter(TwoPhaseConverter):
    """
    Converter for HunyuanVideo models using 2-phase subprocess isolation.
    Splits at dual-stream vs single-stream blocks.
    """

    @property
    def model_name(self) -> str:
        return "Hunyuan"

    @property
    def output_filename(self) -> str:
        return f"HunyuanVideo_Transformer_{self.quantization}.mlpackage"

    @property
    def part1_description(self) -> str:
        return "Part 1 (Dual-Stream Blocks)"

    @property
    def part2_description(self) -> str:
        return "Part 2 (Single-Stream Blocks)"

    def get_part1_worker(self) -> Callable:
        return convert_hunyuan_part1

    def get_part2_worker(self) -> Callable:
        return convert_hunyuan_part2
