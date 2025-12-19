"""
Lumina Image 2.0 conversion workers for subprocess isolation.

These functions run in separate processes to prevent OOM during large model conversion.
Lumina has uniform transformer blocks, so we split at the midpoint.
"""
import torch
import os
import gc
from dataclasses import dataclass
from typing import Optional

import coremltools as ct

from alloy.logging import get_logger
from alloy.workers.base import worker_context, quantize_and_save
from alloy.exceptions import DependencyError

try:
    from diffusers import Lumina2Transformer2DModel
except ImportError:
    Lumina2Transformer2DModel = None

logger = get_logger(__name__)

# Constants
DEFAULT_HEIGHT = 128  # 1024px / 8
DEFAULT_WIDTH = 128
DEFAULT_TEXT_LEN = 256
DEFAULT_BATCH_SIZE = 1


@dataclass
class LuminaInputShapes:
    """Configuration for Lumina model input shapes."""
    batch_size: int = DEFAULT_BATCH_SIZE
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    text_len: int = DEFAULT_TEXT_LEN


class LuminaPart1Wrapper(torch.nn.Module):
    """
    Part 1 of Lumina: Input projection + first half of transformer blocks.
    Output: intermediate hidden_states for Part 2
    """
    def __init__(self, model, num_blocks_part1: int):
        super().__init__()
        self.model = model
        self.num_blocks_part1 = num_blocks_part1

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        # Patchify (convert from [B, C, H, W] to sequence)
        hidden_states = self.model.patch_embed(hidden_states)

        # Time embedding
        temb = self.model.time_embed(timestep)

        # Text embedding
        encoder_hidden_states = self.model.context_embed(encoder_hidden_states)

        # Get rotary embeddings
        image_rotary_emb = self.model.rope(hidden_states)

        # First half of transformer blocks
        for i in range(self.num_blocks_part1):
            hidden_states = self.model.transformer_blocks[i](
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )

        return hidden_states, encoder_hidden_states, temb


class LuminaPart2Wrapper(torch.nn.Module):
    """
    Part 2 of Lumina: Second half of transformer blocks + output projection.
    Input: intermediate hidden_states from Part 1
    Output: final sample
    """
    def __init__(self, model, num_blocks_part1: int):
        super().__init__()
        self.model = model
        self.num_blocks_part1 = num_blocks_part1

    def forward(self, hidden_states, encoder_hidden_states, temb):
        # Get rotary embeddings
        image_rotary_emb = self.model.rope(hidden_states)

        # Second half of transformer blocks
        num_blocks = len(self.model.transformer_blocks)
        for i in range(self.num_blocks_part1, num_blocks):
            hidden_states = self.model.transformer_blocks[i](
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )

        # Final norm and projection
        hidden_states = self.model.norm_out(hidden_states, temb)
        hidden_states = self.model.proj_out(hidden_states)

        return hidden_states


def load_lumina_transformer(model_id_or_path: str):
    """Helper to load just the Lumina transformer efficiently."""
    if Lumina2Transformer2DModel is None:
        raise DependencyError(
            "Lumina2Transformer2DModel not available",
            package_name="diffusers",
            install_command="pip install --upgrade diffusers"
        )

    try:
        logger.debug(
            "[dim]Attempting to load transformer from 'transformer' subfolder...[/dim]",
            extra={"markup": True},
        )
        return Lumina2Transformer2DModel.from_pretrained(
            model_id_or_path,
            subfolder="transformer",
            torch_dtype=torch.float32
        )
    except Exception:
        logger.debug(
            "[dim]Subfolder load failed, trying root...[/dim]",
            extra={"markup": True},
        )
        return Lumina2Transformer2DModel.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32
        )


def convert_lumina_part1(
    model_id: str,
    output_path: str,
    quantization: Optional[str],
    intermediates_dir: Optional[str] = None,
    log_queue=None,
    progress_queue=None,
):
    """Worker function for Part 1 (Input projection + first half of blocks)."""
    with worker_context("Lumina", "Part 1", log_queue, progress_queue, "part1") as reporter:
        if reporter:
            reporter.step_start("load", "Loading Lumina transformer")

        transformer = load_lumina_transformer(model_id)
        transformer.eval()

        if reporter:
            reporter.step_end("load")

        # Split at midpoint
        num_blocks = len(transformer.transformer_blocks)
        num_blocks_part1 = num_blocks // 2
        logger.debug(
            "[dim]Splitting %d blocks: Part 1 has blocks 0-%d[/dim]",
            num_blocks,
            num_blocks_part1 - 1,
            extra={"markup": True},
        )

        # Create dummy inputs
        shapes = LuminaInputShapes()
        in_channels = transformer.config.in_channels  # 16

        hidden_states = torch.randn(
            shapes.batch_size, in_channels, shapes.height, shapes.width
        ).float()
        timestep = torch.tensor([1.0]).float()
        enc_dim = transformer.config.hidden_size  # 2304
        encoder_hidden_states = torch.randn(shapes.batch_size, shapes.text_len, enc_dim).float()

        wrapper = LuminaPart1Wrapper(transformer, num_blocks_part1)
        wrapper.eval()

        # Trace
        if reporter:
            reporter.step_start("trace", "Tracing Part 1")
        logger.debug("[dim]Worker: Tracing Lumina Part 1...[/dim]", extra={"markup": True})
        inputs = [hidden_states, encoder_hidden_states, timestep]
        traced = torch.jit.trace(wrapper, inputs, strict=False)
        if reporter:
            reporter.step_end("trace")

        # Convert to CoreML
        if reporter:
            reporter.step_start("convert", "Converting Part 1 to CoreML")
        logger.debug("[dim]Worker: Converting Lumina Part 1 to Core ML...[/dim]", extra={"markup": True})
        ml_inputs = [
            ct.TensorType(name="hidden_states", shape=hidden_states.shape),
            ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
            ct.TensorType(name="timestep", shape=timestep.shape),
        ]
        ml_outputs = [
            ct.TensorType(name="hidden_states_inter"),
            ct.TensorType(name="encoder_hidden_states_inter"),
            ct.TensorType(name="temb_inter"),
        ]

        model = ct.convert(
            traced,
            inputs=ml_inputs,
            outputs=ml_outputs,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15
        )
        if reporter:
            reporter.step_end("convert")

        # Cleanup
        del traced, wrapper, transformer
        gc.collect()

        # Quantize and save
        quantize_and_save(model, output_path, quantization, intermediates_dir, "lumina_part1", reporter)


def convert_lumina_part2(
    model_id: str,
    output_path: str,
    quantization: Optional[str],
    intermediates_dir: Optional[str] = None,
    log_queue=None,
    progress_queue=None,
):
    """Worker function for Part 2 (Second half of blocks + output projection)."""
    with worker_context("Lumina", "Part 2", log_queue, progress_queue, "part2") as reporter:
        if reporter:
            reporter.step_start("load", "Loading Lumina transformer")

        transformer = load_lumina_transformer(model_id)
        transformer.eval()

        if reporter:
            reporter.step_end("load")

        # Split at midpoint
        num_blocks = len(transformer.transformer_blocks)
        num_blocks_part1 = num_blocks // 2
        logger.debug(
            "[dim]Splitting %d blocks: Part 2 has blocks %d-%d[/dim]",
            num_blocks,
            num_blocks_part1,
            num_blocks - 1,
            extra={"markup": True},
        )

        # Create dummy inputs (intermediate outputs from Part 1)
        shapes = LuminaInputShapes()
        inner_dim = transformer.config.hidden_size  # 2304

        # After patch_embed, shape changes
        seq_len = shapes.height * shapes.width  # Simplified

        hidden_states = torch.randn(shapes.batch_size, seq_len, inner_dim).float()
        encoder_hidden_states = torch.randn(shapes.batch_size, shapes.text_len, inner_dim).float()
        temb = torch.randn(shapes.batch_size, inner_dim).float()

        wrapper = LuminaPart2Wrapper(transformer, num_blocks_part1)
        wrapper.eval()

        # Trace
        if reporter:
            reporter.step_start("trace", "Tracing Part 2")
        logger.debug("[dim]Worker: Tracing Lumina Part 2...[/dim]", extra={"markup": True})
        inputs = [hidden_states, encoder_hidden_states, temb]
        traced = torch.jit.trace(wrapper, inputs, strict=False)
        if reporter:
            reporter.step_end("trace")

        # Convert to CoreML
        if reporter:
            reporter.step_start("convert", "Converting Part 2 to CoreML")
        logger.debug("[dim]Worker: Converting Lumina Part 2 to Core ML...[/dim]", extra={"markup": True})
        ml_inputs = [
            ct.TensorType(name="hidden_states_inter", shape=hidden_states.shape),
            ct.TensorType(name="encoder_hidden_states_inter", shape=encoder_hidden_states.shape),
            ct.TensorType(name="temb_inter", shape=temb.shape),
        ]

        model = ct.convert(
            traced,
            inputs=ml_inputs,
            outputs=[ct.TensorType(name="sample")],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15
        )
        if reporter:
            reporter.step_end("convert")

        # Cleanup
        del traced, wrapper, transformer
        gc.collect()

        # Quantize and save
        quantize_and_save(model, output_path, quantization, intermediates_dir, "lumina_part2", reporter)
