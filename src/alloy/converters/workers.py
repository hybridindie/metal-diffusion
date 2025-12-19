"""
Flux conversion workers for subprocess isolation.

These functions run in separate processes to prevent OOM during large model conversion.
"""
import torch
import torch.nn as nn
import os
import gc
from dataclasses import dataclass
from multiprocessing import Queue
from typing import List, Optional, Tuple

import coremltools as ct
from diffusers import FluxTransformer2DModel

from alloy.logging import get_logger
from alloy.workers.base import worker_context, quantize_and_save

try:
    from diffusers import Flux2Transformer2DModel
except ImportError:
    Flux2Transformer2DModel = None

logger = get_logger(__name__)

# Constants
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_TEXT_LEN = 256
DEFAULT_BATCH_SIZE = 1


@dataclass
class FluxInputShapes:
    """Configuration for Flux model input shapes."""
    batch_size: int = DEFAULT_BATCH_SIZE
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    text_len: int = DEFAULT_TEXT_LEN

    @property
    def sequence_length(self) -> int:
        return (self.height // 2) * (self.width // 2)


def create_flux_dummy_inputs(
    transformer,
    shapes: FluxInputShapes,
    use_hidden_size: bool = False
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Create dummy inputs for Flux model tracing.

    Args:
        transformer: The Flux transformer model
        shapes: Input shape configuration
        use_hidden_size: If True, use hidden_size for states (Part 2), else use in_channels (Part 1)

    Returns:
        Tuple of (input tensors list, input names list)
    """
    s = shapes.sequence_length

    if use_hidden_size:
        # Part 2: uses projected hidden dimension
        hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    else:
        # Part 1: uses raw in_channels
        hidden_dim = transformer.config.in_channels

    hidden_states = torch.randn(shapes.batch_size, s, hidden_dim).float()

    joint_dim = transformer.config.joint_attention_dim
    encoder_hidden_states = torch.randn(shapes.batch_size, shapes.text_len, joint_dim).float()

    if use_hidden_size:
        # Part 2 encoder states also use hidden_size
        hidden_size = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        encoder_hidden_states = torch.randn(shapes.batch_size, shapes.text_len, hidden_size).float()

    pool_dim = transformer.config.pooled_projection_dim
    pooled_projections = torch.randn(shapes.batch_size, pool_dim).float()
    timestep = torch.tensor([1.0]).float()
    guidance = torch.tensor([1.0]).float()
    img_ids = torch.randn(s, 3).float()
    txt_ids = torch.randn(shapes.text_len, 3).float()

    inputs = [hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance]
    names = ["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep", "img_ids", "txt_ids", "guidance"]

    return inputs, names


class FluxPart1Wrapper(torch.nn.Module):
    """
    Part 1 of Flux: Embeddings + DoubleStream Blocks.
    Output: hidden_states, encoder_hidden_states
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_flux2 = Flux2Transformer2DModel and isinstance(model, Flux2Transformer2DModel)

    def _compute_time_embedding(self, timestep, guidance, pooled_projections):
        """Compute time embedding with optional guidance."""
        if guidance is None:
            return self.model.time_text_embed(timestep, pooled_projections)
        return self.model.time_text_embed(timestep, guidance, pooled_projections)

    def _squeeze_ids(self, ids):
        """Squeeze 3D ids to 2D if necessary."""
        return ids[0] if ids.ndim == 3 else ids

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections=None, timestep=None, img_ids=None, txt_ids=None, guidance=None):
        hidden_states = self.model.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self._compute_time_embedding(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.model.context_embedder(encoder_hidden_states)

        txt_ids = self._squeeze_ids(txt_ids)
        img_ids = self._squeeze_ids(img_ids)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.model.pos_embed(ids)

        for block in self.model.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )

        return hidden_states, encoder_hidden_states

class FluxPart2Wrapper(torch.nn.Module):
    """
    Part 2 of Flux: SingleStream Blocks + Final Layer.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_flux2 = Flux2Transformer2DModel and isinstance(model, Flux2Transformer2DModel)

    def _compute_time_embedding(self, timestep, guidance, pooled_projections):
        """Compute time embedding with optional guidance."""
        if guidance is None:
            return self.model.time_text_embed(timestep, pooled_projections)
        return self.model.time_text_embed(timestep, guidance, pooled_projections)

    def _squeeze_ids(self, ids):
        """Squeeze 3D ids to 2D if necessary."""
        return ids[0] if ids.ndim == 3 else ids

    def forward(self, hidden_states_in, encoder_hidden_states_in, pooled_projections=None, timestep=None, img_ids=None, txt_ids=None, guidance=None):
        dtype = hidden_states_in.dtype
        timestep = timestep.to(dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(dtype) * 1000

        temb = self._compute_time_embedding(timestep, guidance, pooled_projections)

        txt_ids = self._squeeze_ids(txt_ids)
        img_ids = self._squeeze_ids(img_ids)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.model.pos_embed(ids)

        hidden_states = hidden_states_in
        encoder_hidden_states = encoder_hidden_states_in

        for block in self.model.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )

        hidden_states = self.model.norm_out(hidden_states, emb=temb)
        hidden_states = self.model.proj_out(hidden_states)

        return hidden_states

def load_flux_transformer(model_id_or_path):
    """Helper to load just the transformer efficiently."""
    if os.path.isfile(model_id_or_path):
        logger.debug(
            "[dim]Worker loading transformer from %s[/dim]",
            model_id_or_path,
            extra={"markup": True},
        )
        return FluxTransformer2DModel.from_single_file(
            model_id_or_path,
            torch_dtype=torch.float32,
            local_files_only=True
        )
    else:
        # Check if it's a repo and needs subfolder
        try:
            logger.debug(
                "[dim]Attempting to load transformer from 'transformer' subfolder...[/dim]",
                extra={"markup": True},
            )
            return FluxTransformer2DModel.from_pretrained(
                model_id_or_path,
                subfolder="transformer",
                torch_dtype=torch.float32
            )
        except EnvironmentError:
            logger.debug(
                "[dim]Subfolder load failed, trying root...[/dim]",
                extra={"markup": True},
            )
            return FluxTransformer2DModel.from_pretrained(
                model_id_or_path,
                torch_dtype=torch.float32
            )

def convert_flux_part1(
    model_id: str,
    output_path: str,
    quantization: Optional[str],
    intermediates_dir: Optional[str] = None,
    log_queue: Optional[Queue] = None,
    progress_queue: Optional[Queue] = None,
) -> None:
    """Worker function for Part 1 (Embeddings + DoubleStream Blocks)."""
    with worker_context("Flux", "Part 1", log_queue, progress_queue, "part1") as reporter:
        if reporter:
            reporter.step_start("load", "Loading Flux transformer")

        transformer = load_flux_transformer(model_id)
        transformer.eval()

        if reporter:
            reporter.step_end("load")

        # Create dummy inputs using helper
        shapes = FluxInputShapes()
        inputs, names = create_flux_dummy_inputs(transformer, shapes, use_hidden_size=False)

        wrapper = FluxPart1Wrapper(transformer)
        wrapper.eval()

        # Trace
        if reporter:
            reporter.step_start("trace", "Tracing Part 1")
        logger.debug("[dim]Worker: Tracing Part 1...[/dim]", extra={"markup": True})
        traced = torch.jit.trace(wrapper, inputs, strict=False)
        if reporter:
            reporter.step_end("trace")

        # Convert to CoreML
        if reporter:
            reporter.step_start("convert", "Converting Part 1 to CoreML")
        logger.debug("[dim]Worker: Converting Part 1 to Core ML...[/dim]", extra={"markup": True})
        ml_inputs = [ct.TensorType(name=name, shape=inp.shape) for name, inp in zip(names, inputs)]
        ml_outputs = [
            ct.TensorType(name="hidden_states_inter"),
            ct.TensorType(name="encoder_hidden_states_inter")
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

        # Quantize and save using helper
        quantize_and_save(model, output_path, quantization, intermediates_dir, "part1", reporter)
    
def convert_flux_part2(
    model_id: str,
    output_path: str,
    quantization: Optional[str],
    intermediates_dir: Optional[str] = None,
    log_queue: Optional[Queue] = None,
    progress_queue: Optional[Queue] = None,
) -> None:
    """Worker function for Part 2 (SingleStream Blocks + Final Layer)."""
    with worker_context("Flux", "Part 2", log_queue, progress_queue, "part2") as reporter:
        if reporter:
            reporter.step_start("load", "Loading Flux transformer")

        transformer = load_flux_transformer(model_id)
        transformer.eval()

        if reporter:
            reporter.step_end("load")

        # Create dummy inputs using helper (Part 2 uses hidden_size from Part 1 output)
        shapes = FluxInputShapes()
        inputs, names = create_flux_dummy_inputs(transformer, shapes, use_hidden_size=True)

        # Part 2 uses "_inter" suffix for input names (matching Part 1 outputs)
        part2_names = ["hidden_states_inter", "encoder_hidden_states_inter"] + names[2:]

        wrapper = FluxPart2Wrapper(transformer)
        wrapper.eval()

        # Trace
        if reporter:
            reporter.step_start("trace", "Tracing Part 2")
        logger.debug("[dim]Worker: Tracing Part 2...[/dim]", extra={"markup": True})
        traced = torch.jit.trace(wrapper, inputs, strict=False)
        if reporter:
            reporter.step_end("trace")

        # Convert to CoreML
        if reporter:
            reporter.step_start("convert", "Converting Part 2 to CoreML")
        logger.debug("[dim]Worker: Converting Part 2 to Core ML...[/dim]", extra={"markup": True})
        ml_inputs = [ct.TensorType(name=name, shape=inp.shape) for name, inp in zip(part2_names, inputs)]

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

        # Quantize and save using helper
        quantize_and_save(model, output_path, quantization, intermediates_dir, "part2", reporter)
