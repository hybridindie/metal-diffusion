import torch
import torch.nn as nn
import coremltools as ct
from diffusers import DiffusionPipeline, FluxTransformer2DModel, FluxPipeline
try:
    from diffusers import Flux2Transformer2DModel
except ImportError:
    Flux2Transformer2DModel = None # Handle older diffusers?
from .converter import ModelConverter
import os
from tqdm import tqdm
from rich.console import Console

console = Console()

class FluxModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.is_flux2 = Flux2Transformer2DModel and isinstance(model, Flux2Transformer2DModel)

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections=None, timestep=None, img_ids=None, txt_ids=None, guidance=None):
        if self.is_flux2:
             # Flux 2 signature: (hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, guidance, ...)
             # No pooled_projections
             return self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False
            )
        
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False
        )

class FluxConverter(ModelConverter):
    def __init__(self, model_id, output_dir, quantization):
        if "/" not in model_id and not os.path.isfile(model_id): 
             model_id = "black-forest-labs/FLUX.1-schnell"
        super().__init__(model_id, output_dir, quantization)
    
    def convert(self):
        """Main conversion entry point"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        with tqdm(total=4, desc="[cyan]Converting Flux Model[/cyan]", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            pbar.set_description("Loading pipeline")
            try:
                if os.path.isfile(self.model_id):
                    console.print(f"[yellow]Detected single file:[/yellow] {self.model_id}")
                    self.pipe = FluxPipeline.from_single_file(self.model_id, torch_dtype=torch.float32)
                else:
                    console.print(f"[cyan]Loading from HF:[/cyan] {self.model_id}")
                    self.pipe = DiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float32)
            except Exception as e:
                console.print(f"[red]Error loading pipeline:[/red] {e}")
                raise e
            pbar.update(1)
            
            # Check model type
            pbar.set_description("Detecting model variant")
            self.is_flux2 = Flux2Transformer2DModel and isinstance(self.pipe.transformer, Flux2Transformer2DModel)
            if self.is_flux2:
                console.print("[green]✓[/green] Detected Flux.2 Model")
            else:
                console.print("[green]✓[/green] Detected Flux.1 Model")
            pbar.update(1)

            transformer = self.pipe.transformer
            transformer.eval()

            ml_model_dir = os.path.join(self.output_dir, "Flux_Transformer.mlpackage")
            if os.path.exists(ml_model_dir):
                console.print(f"[yellow]Model exists, skipping:[/yellow] {ml_model_dir}")
                pbar.update(2)
            else:
                pbar.set_description("Converting transformer")
                self.convert_transformer(transformer, ml_model_dir, pbar)

        console.print(f"[bold green]✓ Conversion complete![/bold green] Saved to {self.output_dir}")

    def convert_transformer(self, transformer, ml_model_dir, pbar=None):
        if pbar:
            pbar.set_description("Preparing model (FP32)")
        else:
            console.print("Converting Transformer (FP32 trace)...")
        transformer = transformer.to(dtype=torch.float32)
        
        # Dimensions based on Flux architecture
        # Default Flux: in_channels=64
        in_channels = transformer.config.in_channels
        
        # Dummy Input Sizes
        # Latents: 64 channels.
        # Resolution 1024x1024 -> VAE 1/8 -> 128x128.
        # Patch Size 1 (on latents?) -> Flux usually patches to 2x2.
        # Wait, config `patch_size`=1.
        # Actually Flux packs latents. Input S = (H/2 * W/2).
        # Let's assume a small resolution for trace.
        h, w = 64, 64 # VAE Latent dims
        s = (h // 2) * (w // 2)
        
        batch_size = 1
        
        hidden_states = torch.randn(batch_size, s, in_channels).float()
        
        # Text Embeddings
        # T5 usually 256 or 512 length for Schnell.
        text_len = 256 
        joint_dim = transformer.config.joint_attention_dim # 4096
        encoder_hidden_states = torch.randn(batch_size, text_len, joint_dim).float()
        
        # Pooled Projections (CLIP)
        pool_dim = transformer.config.pooled_projection_dim # 768
        pooled_projections = torch.randn(batch_size, pool_dim).float()
        
        timestep = torch.tensor([1]).float() # Flux takes float timestep often? Check signature. Signature says LongTensor.
        # However, inside forward: `timestep = timestep.to(hidden_states.dtype) * 1000`.
        # It handles conversion. I'll pass LongTensor to match signature type hint.
        timestep = torch.tensor([1.0]).float()

        guidance = torch.tensor([1.0]).float()
        
        # IDs
        # Flux uses 3D rotary embeddings? (h, w, t?) No, usually just (h, w) for image + others?
        # axes_dims_rope=(16, 56, 56) in config.
        # IDs vector size = len(axes_dims_rope) = 3.
        img_ids = torch.randn(s, 3).float() # Using float as they are coordinates?
        # Forward says `ids = torch.cat((txt_ids, img_ids), dim=0)`.
        # `pos_embed(ids)` -> `ids.float()`.
        # So passing float is fine.
        txt_ids = torch.randn(text_len, 3).float()
        
        example_inputs = [
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance
        ]
        
        # Adapt inputs for Flux 2
        if self.is_flux2:
             # Remove pooled_projections (index 2)
             example_inputs.pop(2)

        if pbar:
            pbar.set_description("Tracing PyTorch model")
        else:
            console.print("Tracing model...")
        wrapper = FluxModelWrapper(transformer)
        wrapper.eval()
        
        # Use strict=False just in case
        traced_model = torch.jit.trace(wrapper, example_inputs, strict=False)
        
        if pbar:
            pbar.update(1)
            pbar.set_description("Converting to Core ML")
        else:
            console.print("Converting to Core ML...")
        
        inputs = [
            ct.TensorType(name="hidden_states", shape=hidden_states.shape),
            ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
            ct.TensorType(name="pooled_projections", shape=pooled_projections.shape),
            ct.TensorType(name="timestep", shape=timestep.shape), # Float input?
            ct.TensorType(name="img_ids", shape=img_ids.shape),
            ct.TensorType(name="txt_ids", shape=txt_ids.shape),
            ct.TensorType(name="guidance", shape=guidance.shape)
        ]
        
        if self.is_flux2:
            # Pop pooled_projections (index 2)
            inputs.pop(2)

        ml_model = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=[ct.TensorType(name="sample")],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14
        )
        
        if self.quantization in ["int4", "4bit", "mixed"]:
            if pbar:
                pbar.set_description(f"Quantizing ({self.quantization})")
            else:
                console.print(f"Applying {self.quantization.capitalize()} quantization...")
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                weight_threshold=512
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            ml_model = ct.optimize.coreml.linear_quantize_weights(ml_model, config)
            
        ml_model.save(ml_model_dir)
        if pbar:
            pbar.update(1)
        console.print(f"[green]✓[/green] Saved: {ml_model_dir}")
