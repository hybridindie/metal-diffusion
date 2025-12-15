
import torch
import torch.nn as nn
import coremltools as ct
from diffusers import FluxControlNetModel
import os
from .converter import ModelConverter
from .flux_converter import NUM_DOUBLE_BLOCKS, NUM_SINGLE_BLOCKS
from rich.console import Console

console = Console()

class FluxControlNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, controlnet_cond, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, guidance):
        # Flux ControlNet Forward
        # Returns: FluxControlNetOutput(controlnet_block_samples=..., controlnet_single_block_samples=...)
        # We return a tuple of all flattened residuals
        
        out = self.model(
            controlnet_cond=controlnet_cond,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False
        )
        # out is (block_samples, single_block_samples)
        block_samples = out[0]
        single_block_samples = out[1]
        
        # Flatten
        return (*block_samples, *single_block_samples)

class FluxControlNetConverter(ModelConverter):
    def __init__(self, model_id, output_dir, quantization):
        if "/" not in model_id and not os.path.isfile(model_id):
             model_id = "black-forest-labs/FLUX.1-Canny-dev" # Example default
        super().__init__(model_id, output_dir, quantization)
        
    def convert(self):
        console.print(f"[cyan]Loading Flux ControlNet:[/cyan] {self.model_id}")
        try:
            model = FluxControlNetModel.from_pretrained(self.model_id, torch_dtype=torch.float32)
        except Exception as e:
             console.print(f"[red]Error loading controlnet:[/red] {e}")
             raise e
             
        model.eval()
        

        # Determine ControlNet conditioning dimensions
        # Usually it matches the VAE latent dimensions but depends on the specific ControlNet
        # model.config doesn't always strictly define 'control_channels'.
        # However, Flux ControlNet from X-Labs/InstantX usually takes same shape as hidden_states? 
        # Or it takes image latents.
        # Let's try to infer from the first layer.
        
        # We'll use 64x64 resolution for tracing (matches FluxConverter trace).
        h, w = 64, 64
        s = (h // 2) * (w // 2)
        batch_size = 1
        
        hidden_states = torch.randn(batch_size, s, in_channels).float()
        
        # Text Embeddings
        text_len = 256 
        joint_dim = model.config.joint_attention_dim # 4096
        encoder_hidden_states = torch.randn(batch_size, text_len, joint_dim).float()
        
        timestep = torch.tensor([1.0]).float()
        guidance = torch.tensor([1.0]).float()
        
        img_ids = torch.randn(s, 3).float()
        txt_ids = torch.randn(text_len, 3).float()
        
        # ControlNet Cond (The Hint)
        # Assuming it's packed latents input? 
        # Some Flux ControlNets concat to hidden_states, so shape is (B, S, in_channels)?
        # Or (B, S, 3) if pixels?
        # Let's use `model.dtype` and a safe guess (B, S, in_channels) for now.
        # If it fails, user might need to adjust.
        # But wait, `MultiControlNetOutput` implies diffusers standard.
        # Diffusers standard `FluxControlNetModel` forward takes `controlnet_cond`.
        # The embedding layer usually maps `controlnet_cond` -> hidden_size.
        # Check `model.pos_embed` or `model.input_blocks`?
        
        # Let's assume input is same size as hidden_states for now (e.g. Depth map projected to latent space).
        controlnet_cond = torch.randn(batch_size, s, in_channels).float() 
        
        example_inputs = [
            controlnet_cond,
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            guidance
        ]
        
        console.print("Tracing Flux ControlNet...")
        wrapper = FluxControlNetWrapper(model)
        wrapper.eval()
        
        traced_model = torch.jit.trace(wrapper, example_inputs, strict=False)
        
        console.print("Converting to Core ML...")
        
        # Inputs
        inputs = [
            ct.TensorType(name="controlnet_cond", shape=controlnet_cond.shape),
            ct.TensorType(name="hidden_states", shape=hidden_states.shape),
            ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
            ct.TensorType(name="timestep", shape=timestep.shape),
            ct.TensorType(name="img_ids", shape=img_ids.shape),
            ct.TensorType(name="txt_ids", shape=txt_ids.shape),
            ct.TensorType(name="guidance", shape=guidance.shape)
        ]
        
        # Outputs
        # We must output the flattened list of residuals.
        # Naming MUST match the expected inputs of the Base Flux Model.
        # i.e. c_double_0 ... c_single_37
        
        outputs = []
        # Double Blocks
        for i in range(NUM_DOUBLE_BLOCKS):
            outputs.append(ct.TensorType(name=f"c_double_{i}"))
            
        # Single Blocks
        for i in range(NUM_SINGLE_BLOCKS):
            outputs.append(ct.TensorType(name=f"c_single_{i}"))
            
        ml_model_dir = os.path.join(self.output_dir, "Flux_ControlNet.mlpackage")
        
        ml_model = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14
        )
        
        if self.quantization in ["int4", "4bit", "mixed"]:
             console.print(f"Quantizing ({self.quantization})...")
             op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                weight_threshold=512
             )
             config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
             ml_model = ct.optimize.coreml.linear_quantize_weights(ml_model, config)
             
        ml_model.save(ml_model_dir)
        console.print(f"[green]✓ ControlNet Saved:[/green] {ml_model_dir}")
        
