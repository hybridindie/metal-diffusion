"""
Wan 2.1 conversion workers for subprocess isolation.

These functions run in separate processes to prevent OOM during large model conversion.
Wan has 40 uniform transformer blocks, so we split at the midpoint.
"""
import torch
import torch.nn as nn
import os
import gc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import coremltools as ct
from rich.console import Console

from alloy.workers.base import quantize_and_save
from alloy.exceptions import DependencyError

# Monkey patch for RoPE compatibility must be applied before loading
from diffusers.models.transformers.transformer_wan import (
    WanAttnProcessor,
    _get_qkv_projections,
    _get_added_kv_projections
)
from diffusers.models.attention_dispatch import dispatch_attention_fn

console = Console()

# Constants
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64
DEFAULT_NUM_FRAMES = 1
DEFAULT_TEXT_LEN = 226
DEFAULT_BATCH_SIZE = 1


def _apply_wan_monkey_patch():
    """Apply monkey patch for WanAttnProcessor to fix Core ML RoPE tracing issue."""

    def patched_wan_attn_processor_call(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1 = hidden_states[..., 0::2]
                x2 = hidden_states[..., 1::2]

                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]

                out_real = x1 * cos - x2 * sin
                out_imag = x1 * sin + x2 * cos

                out = torch.stack((out_real, out_imag), dim=-1).flatten(-2)
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    WanAttnProcessor.__call__ = patched_wan_attn_processor_call
    console.print("[dim]Applied monkey patch to WanAttnProcessor for Core ML compatibility.[/dim]")


# Apply patch on import
_apply_wan_monkey_patch()

try:
    from diffusers import WanTransformer3DModel
except ImportError:
    WanTransformer3DModel = None


@dataclass
class WanInputShapes:
    """Configuration for Wan model input shapes."""
    batch_size: int = DEFAULT_BATCH_SIZE
    num_frames: int = DEFAULT_NUM_FRAMES
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    text_len: int = DEFAULT_TEXT_LEN


class WanPart1Wrapper(torch.nn.Module):
    """
    Part 1 of Wan: Input projection + first half of transformer blocks.
    Output: intermediate hidden_states for Part 2
    """
    def __init__(self, model, num_blocks_part1: int):
        super().__init__()
        self.model = model
        self.num_blocks_part1 = num_blocks_part1

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        # Patchify
        hidden_states = self.model.patch_embed(hidden_states)

        # Time embedding
        temb = self.model.time_embed(timestep)

        # Text embedding
        encoder_hidden_states = self.model.text_embed(encoder_hidden_states)

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


class WanPart2Wrapper(torch.nn.Module):
    """
    Part 2 of Wan: Second half of transformer blocks + output projection.
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

        # Unpatchify
        hidden_states = self.model.unpatchify(hidden_states)

        return hidden_states


def load_wan_transformer(model_id_or_path: str):
    """Helper to load just the Wan transformer efficiently."""
    if WanTransformer3DModel is None:
        raise DependencyError(
            "WanTransformer3DModel not available",
            package_name="diffusers",
            install_command="pip install --upgrade diffusers"
        )

    try:
        console.print(f"[dim]Attempting to load transformer from 'transformer' subfolder...[/dim]")
        return WanTransformer3DModel.from_pretrained(
            model_id_or_path,
            subfolder="transformer",
            torch_dtype=torch.float32
        )
    except Exception:
        console.print(f"[dim]Subfolder load failed, trying root...[/dim]")
        return WanTransformer3DModel.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32
        )


def convert_wan_part1(model_id: str, output_path: str, quantization: Optional[str], intermediates_dir: Optional[str] = None):
    """Worker function for Part 1 (Input projection + first half of blocks)."""
    console.print(f"[cyan]Worker: Starting Wan Part 1 Conversion (PID: {os.getpid()})[/cyan]")

    transformer = load_wan_transformer(model_id)
    transformer.eval()

    # Split at midpoint
    num_blocks = len(transformer.transformer_blocks)
    num_blocks_part1 = num_blocks // 2
    console.print(f"[dim]Splitting {num_blocks} blocks: Part 1 has blocks 0-{num_blocks_part1-1}[/dim]")

    # Create dummy inputs
    shapes = WanInputShapes()
    in_channels = int(getattr(transformer.config, "in_channels", 16))

    hidden_states = torch.randn(
        shapes.batch_size, in_channels, shapes.num_frames, shapes.height, shapes.width
    ).float()
    timestep = torch.tensor([1]).long()
    encoder_dim = getattr(transformer.config, "cross_attention_dim", 4096)
    encoder_hidden_states = torch.randn(shapes.batch_size, shapes.text_len, encoder_dim).float()

    wrapper = WanPart1Wrapper(transformer, num_blocks_part1)
    wrapper.eval()

    # Trace
    console.print("[dim]Worker: Tracing Wan Part 1...[/dim]")
    inputs = [hidden_states, timestep, encoder_hidden_states]
    traced = torch.jit.trace(wrapper, inputs, strict=False)

    # Convert to CoreML
    console.print("[dim]Worker: Converting Wan Part 1 to Core ML...[/dim]")
    ml_inputs = [
        ct.TensorType(name="hidden_states", shape=hidden_states.shape),
        ct.TensorType(name="timestep", shape=timestep.shape),
        ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
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
    quantize_and_save(model, output_path, quantization, intermediates_dir, "wan_part1")
    console.print("[green]Worker: Wan Part 1 Complete[/green]")


def convert_wan_part2(model_id: str, output_path: str, quantization: Optional[str], intermediates_dir: Optional[str] = None):
    """Worker function for Part 2 (Second half of blocks + output projection)."""
    console.print(f"[cyan]Worker: Starting Wan Part 2 Conversion (PID: {os.getpid()})[/cyan]")

    transformer = load_wan_transformer(model_id)
    transformer.eval()

    # Split at midpoint
    num_blocks = len(transformer.transformer_blocks)
    num_blocks_part1 = num_blocks // 2
    console.print(f"[dim]Splitting {num_blocks} blocks: Part 2 has blocks {num_blocks_part1}-{num_blocks-1}[/dim]")

    # Create dummy inputs (intermediate outputs from Part 1)
    shapes = WanInputShapes()
    inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

    # After patchify, dimensions change - approximate
    seq_len = shapes.num_frames * (shapes.height // 2) * (shapes.width // 2)

    hidden_states = torch.randn(shapes.batch_size, seq_len, inner_dim).float()
    encoder_hidden_states = torch.randn(shapes.batch_size, shapes.text_len, inner_dim).float()
    temb = torch.randn(shapes.batch_size, inner_dim).float()

    wrapper = WanPart2Wrapper(transformer, num_blocks_part1)
    wrapper.eval()

    # Trace
    console.print("[dim]Worker: Tracing Wan Part 2...[/dim]")
    inputs = [hidden_states, encoder_hidden_states, temb]
    traced = torch.jit.trace(wrapper, inputs, strict=False)

    # Convert to CoreML
    console.print("[dim]Worker: Converting Wan Part 2 to Core ML...[/dim]")
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
        minimum_deployment_target=ct.target.macOS14
    )

    # Cleanup
    del traced, wrapper, transformer
    gc.collect()

    # Quantize and save
    quantize_and_save(model, output_path, quantization, intermediates_dir, "wan_part2")
    console.print("[green]Worker: Wan Part 2 Complete[/green]")
