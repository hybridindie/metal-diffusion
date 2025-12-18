"""
LTX Video conversion workers for subprocess isolation.

These functions run in separate processes to prevent OOM during large model conversion.
LTX has uniform transformer blocks, so we split at the midpoint.
"""
import torch
import os
import uuid
import tempfile
import gc
from dataclasses import dataclass
from typing import Optional

import coremltools as ct
from rich.console import Console

from alloy.utils.coreml import safe_quantize_model
from alloy.exceptions import DependencyError

try:
    from diffusers import LTXVideoTransformer3DModel
except ImportError:
    LTXVideoTransformer3DModel = None

console = Console()

# Constants
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_NUM_FRAMES = 1
DEFAULT_TEXT_LEN = 128
DEFAULT_BATCH_SIZE = 1


@dataclass
class LTXInputShapes:
    """Configuration for LTX model input shapes."""
    batch_size: int = DEFAULT_BATCH_SIZE
    num_frames: int = DEFAULT_NUM_FRAMES
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    text_len: int = DEFAULT_TEXT_LEN


def quantize_and_save(
    model,
    output_path: str,
    quantization: Optional[str],
    intermediates_dir: Optional[str],
    part_name: str
) -> None:
    """Apply quantization if needed and save the model."""
    if not quantization:
        model.save(output_path)
        return

    console.print(f"[dim]Worker: Quantizing {part_name} ({quantization})...[/dim]")

    if intermediates_dir:
        fp16_path = os.path.join(intermediates_dir, f"{part_name}_fp16_{uuid.uuid4()}.mlpackage")
        model.save(fp16_path)
        del model
        gc.collect()
        model = safe_quantize_model(fp16_path, quantization, intermediate_dir=intermediates_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            fp16_path = os.path.join(tmp, f"{part_name}_fp16.mlpackage")
            model.save(fp16_path)
            del model
            gc.collect()
            model = safe_quantize_model(fp16_path, quantization)

    console.print(f"[dim]Worker: Saving {part_name} to {output_path}...[/dim]")
    model.save(output_path)


class LTXPart1Wrapper(torch.nn.Module):
    """
    Part 1 of LTX: Input projection + first half of transformer blocks.
    Output: intermediate hidden_states for Part 2
    """
    def __init__(self, model, num_blocks_part1: int):
        super().__init__()
        self.model = model
        self.num_blocks_part1 = num_blocks_part1

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask
    ):
        # Get dimensions info for rope
        num_frames = hidden_states.shape[1]
        height = hidden_states.shape[3]
        width = hidden_states.shape[4]

        # Patchify
        hidden_states = self.model.patchify(hidden_states)

        # Project input
        hidden_states = self.model.proj_in(hidden_states)

        # Project caption
        encoder_hidden_states = self.model.caption_projection(encoder_hidden_states)

        # Get positional embeddings
        image_rotary_emb = self.model.rope(hidden_states, num_frames, height, width)

        # Time embeddings
        temb = self.model.time_embed(timestep)

        # First half of transformer blocks
        for i in range(self.num_blocks_part1):
            hidden_states = self.model.transformer_blocks[i](
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                encoder_attention_mask=encoder_attention_mask,
                image_rotary_emb=image_rotary_emb
            )

        return hidden_states, encoder_hidden_states, temb


class LTXPart2Wrapper(torch.nn.Module):
    """
    Part 2 of LTX: Second half of transformer blocks + output projection.
    Input: intermediate hidden_states from Part 1
    Output: final sample
    """
    def __init__(self, model, num_blocks_part1: int):
        super().__init__()
        self.model = model
        self.num_blocks_part1 = num_blocks_part1

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        encoder_attention_mask,
        num_frames,
        height,
        width
    ):
        # Get positional embeddings (need to recompute)
        image_rotary_emb = self.model.rope(hidden_states, num_frames, height, width)

        # Second half of transformer blocks
        num_blocks = len(self.model.transformer_blocks)
        for i in range(self.num_blocks_part1, num_blocks):
            hidden_states = self.model.transformer_blocks[i](
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                encoder_attention_mask=encoder_attention_mask,
                image_rotary_emb=image_rotary_emb
            )

        # Final norm and projection
        scale_shift_values = self.model.scale_shift_table[None, None] + temb[:, :, None]
        shift, scale = scale_shift_values.chunk(2, dim=1)

        hidden_states = self.model.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.model.proj_out(hidden_states)

        # Unpatchify
        hidden_states = self.model.unpatchify(hidden_states, num_frames, height, width)

        return hidden_states


def load_ltx_transformer(model_id_or_path: str):
    """Helper to load just the LTX transformer efficiently."""
    if LTXVideoTransformer3DModel is None:
        raise DependencyError(
            "LTXVideoTransformer3DModel not available",
            package_name="diffusers",
            install_command="pip install --upgrade diffusers"
        )

    if os.path.isfile(model_id_or_path):
        console.print(f"[dim]Worker loading transformer from single file: {model_id_or_path}[/dim]")
        return LTXVideoTransformer3DModel.from_single_file(
            model_id_or_path,
            torch_dtype=torch.float32,
            local_files_only=True
        )
    else:
        try:
            console.print(f"[dim]Attempting to load transformer from 'transformer' subfolder...[/dim]")
            return LTXVideoTransformer3DModel.from_pretrained(
                model_id_or_path,
                subfolder="transformer",
                torch_dtype=torch.float32
            )
        except Exception:
            console.print(f"[dim]Subfolder load failed, trying root...[/dim]")
            return LTXVideoTransformer3DModel.from_pretrained(
                model_id_or_path,
                torch_dtype=torch.float32
            )


def convert_ltx_part1(model_id: str, output_path: str, quantization: Optional[str], intermediates_dir: Optional[str] = None):
    """Worker function for Part 1 (Input projection + first half of blocks)."""
    console.print(f"[cyan]Worker: Starting LTX Part 1 Conversion (PID: {os.getpid()})[/cyan]")

    transformer = load_ltx_transformer(model_id)
    transformer.eval()

    # Split at midpoint
    num_blocks = len(transformer.transformer_blocks)
    num_blocks_part1 = num_blocks // 2
    console.print(f"[dim]Splitting {num_blocks} blocks: Part 1 has blocks 0-{num_blocks_part1-1}[/dim]")

    # Create dummy inputs
    shapes = LTXInputShapes()
    in_channels = transformer.config.in_channels

    hidden_states = torch.randn(
        shapes.batch_size, shapes.num_frames, in_channels, shapes.height, shapes.width
    ).float()
    timestep = torch.tensor([1.0]).float()
    text_dim = transformer.config.caption_channels
    encoder_hidden_states = torch.randn(shapes.batch_size, shapes.text_len, text_dim).float()
    encoder_attention_mask = torch.ones(shapes.batch_size, shapes.text_len).long()

    wrapper = LTXPart1Wrapper(transformer, num_blocks_part1)
    wrapper.eval()

    # Trace
    console.print("[dim]Worker: Tracing LTX Part 1...[/dim]")
    inputs = [hidden_states, encoder_hidden_states, timestep, encoder_attention_mask]
    traced = torch.jit.trace(wrapper, inputs, strict=False)

    # Convert to CoreML
    console.print("[dim]Worker: Converting LTX Part 1 to Core ML...[/dim]")
    ml_inputs = [
        ct.TensorType(name="hidden_states", shape=hidden_states.shape),
        ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
        ct.TensorType(name="timestep", shape=timestep.shape),
        ct.TensorType(name="encoder_attention_mask", shape=encoder_attention_mask.shape),
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
        minimum_deployment_target=ct.target.macOS14
    )

    # Cleanup
    del traced, wrapper, transformer
    gc.collect()

    # Quantize and save
    quantize_and_save(model, output_path, quantization, intermediates_dir, "ltx_part1")
    console.print("[green]Worker: LTX Part 1 Complete[/green]")


def convert_ltx_part2(model_id: str, output_path: str, quantization: Optional[str], intermediates_dir: Optional[str] = None):
    """Worker function for Part 2 (Second half of blocks + output projection)."""
    console.print(f"[cyan]Worker: Starting LTX Part 2 Conversion (PID: {os.getpid()})[/cyan]")

    transformer = load_ltx_transformer(model_id)
    transformer.eval()

    # Split at midpoint
    num_blocks = len(transformer.transformer_blocks)
    num_blocks_part1 = num_blocks // 2
    console.print(f"[dim]Splitting {num_blocks} blocks: Part 2 has blocks {num_blocks_part1}-{num_blocks-1}[/dim]")

    # Create dummy inputs (intermediate outputs from Part 1)
    shapes = LTXInputShapes()
    inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

    # After patchify and proj_in, sequence length changes
    seq_len = shapes.num_frames * shapes.height * shapes.width  # Simplified

    hidden_states = torch.randn(shapes.batch_size, seq_len, inner_dim).float()
    encoder_hidden_states = torch.randn(shapes.batch_size, shapes.text_len, inner_dim).float()
    temb = torch.randn(shapes.batch_size, inner_dim).float()
    encoder_attention_mask = torch.ones(shapes.batch_size, shapes.text_len).long()

    # Shape info for unpatchify
    num_frames = torch.tensor([shapes.num_frames]).long()
    height = torch.tensor([shapes.height]).long()
    width = torch.tensor([shapes.width]).long()

    wrapper = LTXPart2Wrapper(transformer, num_blocks_part1)
    wrapper.eval()

    # Trace
    console.print("[dim]Worker: Tracing LTX Part 2...[/dim]")
    inputs = [hidden_states, encoder_hidden_states, temb, encoder_attention_mask, num_frames, height, width]
    traced = torch.jit.trace(wrapper, inputs, strict=False)

    # Convert to CoreML
    console.print("[dim]Worker: Converting LTX Part 2 to Core ML...[/dim]")
    ml_inputs = [
        ct.TensorType(name="hidden_states_inter", shape=hidden_states.shape),
        ct.TensorType(name="encoder_hidden_states_inter", shape=encoder_hidden_states.shape),
        ct.TensorType(name="temb_inter", shape=temb.shape),
        ct.TensorType(name="encoder_attention_mask", shape=encoder_attention_mask.shape),
        ct.TensorType(name="num_frames", shape=num_frames.shape),
        ct.TensorType(name="height", shape=height.shape),
        ct.TensorType(name="width", shape=width.shape),
    ]

    model = ct.convert(
        traced,
        inputs=ml_inputs,
        outputs=[ct.TensorType(name="sample")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14
    )

    # Cleanup
    del traced, wrapper, transformer
    gc.collect()

    # Quantize and save
    quantize_and_save(model, output_path, quantization, intermediates_dir, "ltx_part2")
    console.print("[green]Worker: LTX Part 2 Complete[/green]")
