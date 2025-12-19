"""
HunyuanVideo conversion workers for subprocess isolation.

These functions run in separate processes to prevent OOM during large model conversion.
HunyuanVideo follows the same dual-stream/single-stream architecture as Flux.
"""
import torch
import os
import gc
from dataclasses import dataclass
from multiprocessing import Queue
from typing import List, Optional, Tuple, Union

import coremltools as ct

from alloy.logging import get_logger
from alloy.workers.base import worker_context, quantize_and_save
from alloy.exceptions import DependencyError

try:
    from diffusers import HunyuanVideoTransformer3DModel
except ImportError:
    HunyuanVideoTransformer3DModel = None

logger = get_logger(__name__)

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
    # Note: use_hidden_size parameter reserved for future Part 2 dimension handling

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
        raise DependencyError(
            "HunyuanVideoTransformer3DModel not available",
            package_name="diffusers",
            install_command="pip install --upgrade diffusers"
        )

    if os.path.isfile(model_id_or_path):
        logger.debug(
            "[dim]Worker loading transformer from single file: %s[/dim]",
            model_id_or_path,
            extra={"markup": True},
        )
        return HunyuanVideoTransformer3DModel.from_single_file(
            model_id_or_path,
            torch_dtype=torch.float32,
            local_files_only=True
        )
    else:
        try:
            logger.debug(
                "[dim]Attempting to load transformer from 'transformer' subfolder...[/dim]",
                extra={"markup": True},
            )
            return HunyuanVideoTransformer3DModel.from_pretrained(
                model_id_or_path,
                subfolder="transformer",
                torch_dtype=torch.float32
            )
        except Exception:
            logger.debug(
                "[dim]Subfolder load failed, trying root...[/dim]",
                extra={"markup": True},
            )
            return HunyuanVideoTransformer3DModel.from_pretrained(
                model_id_or_path,
                torch_dtype=torch.float32
            )


def convert_hunyuan_part1(
    model_id: str,
    output_path: str,
    quantization: Optional[str],
    intermediates_dir: Optional[str] = None,
    log_queue: Optional[Queue] = None,
    progress_queue: Optional[Queue] = None,
) -> None:
    """Worker function for Part 1 (Input Embeddings + Dual-Stream Blocks)."""
    with worker_context("Hunyuan", "Part 1", log_queue, progress_queue, "part1") as reporter:
        if reporter:
            reporter.step_start("load", "Loading Hunyuan transformer")

        transformer = load_hunyuan_transformer(model_id)
        transformer.eval()

        if reporter:
            reporter.step_end("load")

        # Create dummy inputs
        shapes = HunyuanInputShapes()
        inputs, names = create_hunyuan_dummy_inputs(transformer, shapes, use_hidden_size=False)

        wrapper = HunyuanPart1Wrapper(transformer)
        wrapper.eval()

        # Trace
        if reporter:
            reporter.step_start("trace", "Tracing Part 1")
        logger.debug("[dim]Worker: Tracing Hunyuan Part 1...[/dim]", extra={"markup": True})
        traced = torch.jit.trace(wrapper, inputs, strict=False)
        if reporter:
            reporter.step_end("trace")

        # Convert to CoreML
        if reporter:
            reporter.step_start("convert", "Converting Part 1 to CoreML")
        logger.debug("[dim]Worker: Converting Hunyuan Part 1 to Core ML...[/dim]", extra={"markup": True})
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
        if reporter:
            reporter.step_end("convert")

        # Cleanup PyTorch resources
        del traced, wrapper, transformer
        gc.collect()

        # Quantize and save
        quantize_and_save(model, output_path, quantization, intermediates_dir, "hunyuan_part1", reporter)


def convert_hunyuan_part2(
    model_id: str,
    output_path: str,
    quantization: Optional[str],
    intermediates_dir: Optional[str] = None,
    log_queue: Optional[Queue] = None,
    progress_queue: Optional[Queue] = None,
) -> None:
    """Worker function for Part 2 (Single-Stream Blocks + Final Layer)."""
    with worker_context("Hunyuan", "Part 2", log_queue, progress_queue, "part2") as reporter:
        if reporter:
            reporter.step_start("load", "Loading Hunyuan transformer")

        transformer = load_hunyuan_transformer(model_id)
        transformer.eval()

        if reporter:
            reporter.step_end("load")

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
        if reporter:
            reporter.step_start("trace", "Tracing Part 2")
        logger.debug("[dim]Worker: Tracing Hunyuan Part 2...[/dim]", extra={"markup": True})
        inputs = [hidden_states, encoder_hidden_states, temb, timestep, pooled_projections, guidance]
        traced = torch.jit.trace(wrapper, inputs, strict=False)
        if reporter:
            reporter.step_end("trace")

        # Convert to CoreML
        if reporter:
            reporter.step_start("convert", "Converting Part 2 to CoreML")
        logger.debug("[dim]Worker: Converting Hunyuan Part 2 to Core ML...[/dim]", extra={"markup": True})
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
        if reporter:
            reporter.step_end("convert")

        # Cleanup PyTorch resources
        del traced, wrapper, transformer
        gc.collect()

        # Quantize and save
        quantize_and_save(model, output_path, quantization, intermediates_dir, "hunyuan_part2", reporter)
