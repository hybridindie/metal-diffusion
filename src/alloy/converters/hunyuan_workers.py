"""
HunyuanVideo conversion workers for subprocess isolation.

These functions run in separate processes to prevent OOM during large model conversion.
HunyuanVideo follows the same dual-stream/single-stream architecture as Flux.
"""
import torch
import torch.nn as nn
import os
import uuid
import tempfile
import gc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import coremltools as ct
from rich.console import Console

from alloy.utils.coreml import safe_quantize_model

try:
    from diffusers import HunyuanVideoTransformer3DModel
except ImportError:
    HunyuanVideoTransformer3DModel = None

console = Console()

# Constants
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_NUM_FRAMES = 1
DEFAULT_TEXT_LEN = 256
DEFAULT_BATCH_SIZE = 1


@dataclass
class HunyuanInputShapes:
    """Configuration for HunyuanVideo model input shapes."""
    batch_size: int = DEFAULT_BATCH_SIZE
    num_frames: int = DEFAULT_NUM_FRAMES
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    text_len: int = DEFAULT_TEXT_LEN


def create_hunyuan_dummy_inputs(
    transformer,
    shapes: HunyuanInputShapes,
    use_hidden_size: bool = False
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Create dummy inputs for HunyuanVideo model tracing.

    Args:
        transformer: The HunyuanVideo transformer model
        shapes: Input shape configuration
        use_hidden_size: If True, use hidden_size for states (Part 2), else use in_channels (Part 1)

    Returns:
        Tuple of (input tensors list, input names list)
    """
    in_channels = transformer.config.in_channels  # 16

    if use_hidden_size:
        # Part 2: uses projected hidden dimension
        hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    else:
        # Part 1: uses raw in_channels
        hidden_dim = in_channels

    # For video: (B, C, F, H, W)
    hidden_states = torch.randn(
        shapes.batch_size, in_channels, shapes.num_frames, shapes.height, shapes.width
    ).float()

    # Timestep
    timestep = torch.tensor([1]).long()

    # Text embeddings
    text_dim = transformer.config.text_embed_dim  # 4096
    encoder_hidden_states = torch.randn(shapes.batch_size, shapes.text_len, text_dim).float()
    encoder_attention_mask = torch.ones(shapes.batch_size, shapes.text_len).long()

    # Pooled projections
    pool_dim = transformer.config.pooled_projection_dim  # 768
    pooled_projections = torch.randn(shapes.batch_size, pool_dim).float()

    # Guidance
    guidance = torch.tensor([1000.0]).float()

    inputs = [
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance
    ]
    names = [
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "pooled_projections",
        "guidance"
    ]

    return inputs, names


def quantize_and_save(
    model,
    output_path: str,
    quantization: Optional[str],
    intermediates_dir: Optional[str],
    part_name: str
) -> None:
    """
    Apply quantization if needed and save the model.

    Args:
        model: CoreML model to quantize
        output_path: Final output path
        quantization: Quantization type (int4, int8, etc.) or None
        intermediates_dir: Directory for intermediate files
        part_name: Name for intermediate file (e.g., "part1", "part2")
    """
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


class HunyuanPart1Wrapper(torch.nn.Module):
    """
    Part 1 of HunyuanVideo: Input embeddings + Dual-Stream Transformer Blocks.
    Output: hidden_states, encoder_hidden_states (intermediate tensors for Part 2)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance
    ):
        # Get config values
        batch_size = hidden_states.shape[0]

        # Patchify and embed hidden states
        hidden_states = self.model.patch_embed(hidden_states)

        # Time embeddings
        timestep = timestep.to(hidden_states.dtype)
        temb = self.model.time_embed(timestep, pooled_projections, guidance)

        # Text embeddings
        encoder_hidden_states = self.model.text_embed(encoder_hidden_states, encoder_attention_mask)

        # Get rotary embeddings
        image_rotary_emb = self.model.rope(hidden_states)

        # Dual-stream transformer blocks
        for block in self.model.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )

        return hidden_states, encoder_hidden_states, temb


class HunyuanPart2Wrapper(torch.nn.Module):
    """
    Part 2 of HunyuanVideo: Single-Stream Transformer Blocks + Final Layer.
    Input: Intermediate hidden_states, encoder_hidden_states from Part 1
    Output: Final sample
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        timestep,
        pooled_projections,
        guidance
    ):
        # Recompute rotary embeddings for Part 2
        image_rotary_emb = self.model.rope(hidden_states)

        # If temb not provided, recompute
        if temb is None:
            timestep = timestep.to(hidden_states.dtype)
            temb = self.model.time_embed(timestep, pooled_projections, guidance)

        # Single-stream transformer blocks
        for block in self.model.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )

        # Final layer
        hidden_states = self.model.norm_out(hidden_states, temb)
        hidden_states = self.model.proj_out(hidden_states)

        return hidden_states


def load_hunyuan_transformer(model_id_or_path: str):
    """Helper to load just the HunyuanVideo transformer efficiently."""
    if HunyuanVideoTransformer3DModel is None:
        raise ImportError("HunyuanVideoTransformer3DModel not available. Please upgrade diffusers.")

    if os.path.isfile(model_id_or_path):
        console.print(f"[dim]Worker loading transformer from single file: {model_id_or_path}[/dim]")
        return HunyuanVideoTransformer3DModel.from_single_file(
            model_id_or_path,
            torch_dtype=torch.float32,
            local_files_only=True
        )
    else:
        try:
            console.print(f"[dim]Attempting to load transformer from 'transformer' subfolder...[/dim]")
            return HunyuanVideoTransformer3DModel.from_pretrained(
                model_id_or_path,
                subfolder="transformer",
                torch_dtype=torch.float32
            )
        except Exception:
            console.print(f"[dim]Subfolder load failed, trying root...[/dim]")
            return HunyuanVideoTransformer3DModel.from_pretrained(
                model_id_or_path,
                torch_dtype=torch.float32
            )


def convert_hunyuan_part1(model_id: str, output_path: str, quantization: Optional[str], intermediates_dir: Optional[str] = None):
    """Worker function for Part 1 (Input Embeddings + Dual-Stream Blocks)."""
    console.print(f"[cyan]Worker: Starting Hunyuan Part 1 Conversion (PID: {os.getpid()})[/cyan]")

    transformer = load_hunyuan_transformer(model_id)
    transformer.eval()

    # Create dummy inputs
    shapes = HunyuanInputShapes()
    inputs, names = create_hunyuan_dummy_inputs(transformer, shapes, use_hidden_size=False)

    wrapper = HunyuanPart1Wrapper(transformer)
    wrapper.eval()

    # Trace
    console.print("[dim]Worker: Tracing Hunyuan Part 1...[/dim]")
    traced = torch.jit.trace(wrapper, inputs, strict=False)

    # Convert to CoreML
    console.print("[dim]Worker: Converting Hunyuan Part 1 to Core ML...[/dim]")
    ml_inputs = [ct.TensorType(name=name, shape=inp.shape) for name, inp in zip(names, inputs)]
    ml_outputs = [
        ct.TensorType(name="hidden_states_inter"),
        ct.TensorType(name="encoder_hidden_states_inter"),
        ct.TensorType(name="temb_inter")
    ]

    model = ct.convert(
        traced,
        inputs=ml_inputs,
        outputs=ml_outputs,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14
    )

    # Cleanup PyTorch resources
    del traced, wrapper, transformer
    gc.collect()

    # Quantize and save
    quantize_and_save(model, output_path, quantization, intermediates_dir, "hunyuan_part1")
    console.print("[green]Worker: Hunyuan Part 1 Complete[/green]")


def convert_hunyuan_part2(model_id: str, output_path: str, quantization: Optional[str], intermediates_dir: Optional[str] = None):
    """Worker function for Part 2 (Single-Stream Blocks + Final Layer)."""
    console.print(f"[cyan]Worker: Starting Hunyuan Part 2 Conversion (PID: {os.getpid()})[/cyan]")

    transformer = load_hunyuan_transformer(model_id)
    transformer.eval()

    # Get dimensions from config
    hidden_size = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    shapes = HunyuanInputShapes()

    # Part 2 inputs are intermediate outputs from Part 1
    # These have been through embedding, so dimensions are different
    batch_size = shapes.batch_size
    seq_len = (shapes.height // 2) * (shapes.width // 2) * shapes.num_frames  # Approximate after patchify

    hidden_states = torch.randn(batch_size, seq_len, hidden_size).float()
    encoder_hidden_states = torch.randn(batch_size, shapes.text_len, hidden_size).float()
    temb = torch.randn(batch_size, hidden_size).float()

    # Also need these for potential recomputation
    timestep = torch.tensor([1]).long()
    pool_dim = transformer.config.pooled_projection_dim
    pooled_projections = torch.randn(batch_size, pool_dim).float()
    guidance = torch.tensor([1000.0]).float()

    wrapper = HunyuanPart2Wrapper(transformer)
    wrapper.eval()

    # Trace
    console.print("[dim]Worker: Tracing Hunyuan Part 2...[/dim]")
    inputs = [hidden_states, encoder_hidden_states, temb, timestep, pooled_projections, guidance]
    traced = torch.jit.trace(wrapper, inputs, strict=False)

    # Convert to CoreML
    console.print("[dim]Worker: Converting Hunyuan Part 2 to Core ML...[/dim]")
    ml_inputs = [
        ct.TensorType(name="hidden_states_inter", shape=hidden_states.shape),
        ct.TensorType(name="encoder_hidden_states_inter", shape=encoder_hidden_states.shape),
        ct.TensorType(name="temb_inter", shape=temb.shape),
        ct.TensorType(name="timestep", shape=timestep.shape),
        ct.TensorType(name="pooled_projections", shape=pooled_projections.shape),
        ct.TensorType(name="guidance", shape=guidance.shape),
    ]

    model = ct.convert(
        traced,
        inputs=ml_inputs,
        outputs=[ct.TensorType(name="sample")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14
    )

    # Cleanup PyTorch resources
    del traced, wrapper, transformer
    gc.collect()

    # Quantize and save
    quantize_and_save(model, output_path, quantization, intermediates_dir, "hunyuan_part2")
    console.print("[green]Worker: Hunyuan Part 2 Complete[/green]")
