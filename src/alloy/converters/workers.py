import torch
import torch.nn as nn
import os
import coremltools as ct
from diffusers import FluxTransformer2DModel
try:
    from diffusers import Flux2Transformer2DModel
except ImportError:
    Flux2Transformer2DModel = None
import tempfile
import gc
from alloy.utils.coreml import safe_quantize_model
from rich.console import Console

console = Console()

class FluxPart1Wrapper(torch.nn.Module):
    """
    Part 1 of Flux: Embeddings + DoubleStream Blocks.
    Output: hidden_states, encoder_hidden_states
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_flux2 = Flux2Transformer2DModel and isinstance(model, Flux2Transformer2DModel)

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections=None, timestep=None, img_ids=None, txt_ids=None, guidance=None):
        hidden_states = self.model.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        if self.model.config.guidance_embeds:
             temb = (
                self.model.time_text_embed(timestep, pooled_projections)
                if guidance is None
                else self.model.time_text_embed(timestep, guidance, pooled_projections)
            )
        else:
             temb = (
                self.model.time_text_embed(timestep, pooled_projections)
                if guidance is None
                else self.model.time_text_embed(timestep, guidance, pooled_projections)
            )

        encoder_hidden_states = self.model.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

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

    def forward(self, hidden_states_in, encoder_hidden_states_in, pooled_projections=None, timestep=None, img_ids=None, txt_ids=None, guidance=None):
        dtype = hidden_states_in.dtype
        timestep = timestep.to(dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(dtype) * 1000

        temb = (
            self.model.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.model.time_text_embed(timestep, guidance, pooled_projections)
        )

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.model.pos_embed(ids)
        
        hidden_states = hidden_states_in
        encoder_hidden_states = encoder_hidden_states_in

        # Single Stream Blocks
        for block in self.model.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb
            )
            
        # Final Layern_states = self.model.norm_out(hidden_states, emb=temb)
        hidden_states = self.model.proj_out(hidden_states)
        
        return hidden_states

def load_flux_transformer(model_id_or_path):
    """Helper to load just the transformer efficiently."""
    if os.path.isfile(model_id_or_path):
        console.print(f"[dim]Worker loading transformer from {model_id_or_path}[/dim]")
        return FluxTransformer2DModel.from_single_file(
            model_id_or_path, 
            torch_dtype=torch.float32,
            local_files_only=True
        )
    else:
        # Check if it's a repo and needs subfolder
        try:
            console.print(f"[dim]Attempting to load transformer from 'transformer' subfolder...[/dim]")
            return FluxTransformer2DModel.from_pretrained(
                model_id_or_path, 
                subfolder="transformer", 
                torch_dtype=torch.float32
            )
        except EnvironmentError:
            console.print(f"[dim]Subfolder load failed, trying root...[/dim]")
            return FluxTransformer2DModel.from_pretrained(
                model_id_or_path, 
                torch_dtype=torch.float32
            )

def convert_flux_part1(model_id, output_path, quantization, intermediates_dir=None):
    """Worker function for Part 1"""
    console.print(f"[cyan]Worker: Starting Part 1 Conversion (PID: {os.getpid()})[/cyan]")
    
    transformer = load_flux_transformer(model_id)
    transformer.eval()
    
    # Setup dummy inputs
    in_channels = transformer.config.in_channels
    h, w = 64, 64
    s = (h // 2) * (w // 2)
    batch_size = 1
    
    hidden_states = torch.randn(batch_size, s, in_channels).float()
    text_len = 256 
    joint_dim = transformer.config.joint_attention_dim
    encoder_hidden_states = torch.randn(batch_size, text_len, joint_dim).float()
    pool_dim = transformer.config.pooled_projection_dim
    pooled_projections = torch.randn(batch_size, pool_dim).float()
    timestep = torch.tensor([1.0]).float()
    guidance = torch.tensor([1.0]).float()
    img_ids = torch.randn(s, 3).float()
    txt_ids = torch.randn(text_len, 3).float()
    
    wrapper = FluxPart1Wrapper(transformer)
    wrapper.eval()
    
    # Trace
    console.print("[dim]Worker: Tracing Part 1...[/dim]")
    inputs = [hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance]
    traced = torch.jit.trace(wrapper, inputs, strict=False)
    
    # Convert
    console.print("[dim]Worker: Converting Part 1 to Core ML...[/dim]")
    ml_inputs = [
        ct.TensorType(name="hidden_states", shape=hidden_states.shape),
        ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
        ct.TensorType(name="pooled_projections", shape=pooled_projections.shape),
        ct.TensorType(name="timestep", shape=timestep.shape),
        ct.TensorType(name="img_ids", shape=img_ids.shape),
        ct.TensorType(name="txt_ids", shape=txt_ids.shape),
        ct.TensorType(name="guidance", shape=guidance.shape)
    ]
    
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
    
    # Cleanup PyTorch
    del traced, wrapper, transformer
    gc.collect()
    
    # Quantize
    if quantization:
        console.print(f"[dim]Worker: Quantizing Part 1 ({quantization})...[/dim]")
        
        if intermediates_dir:
            # Save unsqueezed FP16 model to persistent intermediates dir
            import uuid
            fp16_path = os.path.join(intermediates_dir, f"part1_fp16_{uuid.uuid4()}.mlpackage")
            model.save(fp16_path)
            del model
            gc.collect()
            model = safe_quantize_model(fp16_path, quantization, intermediate_dir=intermediates_dir)
            # safe_quantize will load from fp16_path. We verified it handles path input.
            # But wait, safe_quantize_model(path) returns loaded object.
            # If we pass path, it doesn't create intermediate.
            # So passing intermediate_dir is irrelevant if we pass PATH.
            # BUT the fp16_path IS the intermediate.
            # So we just stick with this.
        else:
            with tempfile.TemporaryDirectory() as tmp:
                fp16_path = os.path.join(tmp, "p1_fp16.mlpackage")
                model.save(fp16_path)
                del model
                gc.collect()
                model = safe_quantize_model(fp16_path, quantization)
            
    console.print(f"[dim]Worker: Saving Part 1 to {output_path}...[/dim]")
    model.save(output_path)
    console.print("[green]Worker: Part 1 Complete[/green]")
    
def convert_flux_part2(model_id, output_path, quantization, intermediates_dir=None):
    """Worker function for Part 2"""
    console.print(f"[cyan]Worker: Starting Part 2 Conversion (PID: {os.getpid()})[/cyan]")
    
    transformer = load_flux_transformer(model_id)
    transformer.eval()
    
    # Calculate hidden size for Part 2 inputs (outputs of Part 1)
    # Part 1 projects inputs to hidden_size (num_attention_heads * attention_head_dim)
    hidden_size = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    
    # Setup dummy inputs matching Part 1 Output Shapes
    h, w = 64, 64
    s = (h // 2) * (w // 2)
    batch_size = 1
    
    # Note: Part 1 outputs are in the projected hidden dimension!
    hidden_states = torch.randn(batch_size, s, hidden_size).float()
    
    text_len = 256
    encoder_hidden_states = torch.randn(batch_size, text_len, hidden_size).float()
    
    pool_dim = transformer.config.pooled_projection_dim
    pooled_projections = torch.randn(batch_size, pool_dim).float()
    timestep = torch.tensor([1.0]).float()
    guidance = torch.tensor([1.0]).float()
    img_ids = torch.randn(s, 3).float()
    txt_ids = torch.randn(text_len, 3).float()
    
    wrapper = FluxPart2Wrapper(transformer)
    wrapper.eval()
    
    # Trace
    console.print("[dim]Worker: Tracing Part 2...[/dim]")
    inputs = [hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance]
    traced = torch.jit.trace(wrapper, inputs, strict=False)
    
    # Convert
    console.print("[dim]Worker: Converting Part 2 to Core ML...[/dim]")
    ml_inputs = [
        ct.TensorType(name="hidden_states_inter", shape=hidden_states.shape),
        ct.TensorType(name="encoder_hidden_states_inter", shape=encoder_hidden_states.shape),
        ct.TensorType(name="pooled_projections", shape=pooled_projections.shape),
        ct.TensorType(name="timestep", shape=timestep.shape),
        ct.TensorType(name="img_ids", shape=img_ids.shape),
        ct.TensorType(name="txt_ids", shape=txt_ids.shape),
        ct.TensorType(name="guidance", shape=guidance.shape)
    ]
    
    model = ct.convert(
        traced,
        inputs=ml_inputs,
        outputs=[ct.TensorType(name="sample")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14
    )
    
    # Cleanup PyTorch
    del traced, wrapper, transformer
    gc.collect()
    # Quantize
    if quantization:
        console.print(f"[dim]Worker: Quantizing Part 2 ({quantization})...[/dim]")
        
        if intermediates_dir:
            import uuid
            fp16_path = os.path.join(intermediates_dir, f"part2_fp16_{uuid.uuid4()}.mlpackage")
            model.save(fp16_path)
            del model
            gc.collect()
            model = safe_quantize_model(fp16_path, quantization, intermediate_dir=intermediates_dir)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                fp16_path = os.path.join(tmp, "p2_fp16.mlpackage")
                model.save(fp16_path)
                del model
                gc.collect()
                model = safe_quantize_model(fp16_path, quantization)
            
    console.print(f"[dim]Worker: Saving Part 2 to {output_path}...[/dim]")
    model.save(output_path)
    console.print("[green]Worker: Part 2 Complete[/green]")
