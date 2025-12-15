import torch
import coremltools as ct
from diffusers import LTXVideoTransformer3DModel, LTXPipeline
from .converter import ModelConverter
import os
import shutil
from typing import Optional, Dict, Any

class LTXModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(
        self, 
        hidden_states, 
        encoder_hidden_states,
        timestep, 
        encoder_attention_mask,
        num_frames,
        height,
        width
    ):
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            return_dict=False
        )

class LTXConverter(ModelConverter):
    def __init__(self, model_id, output_dir, quantization):
        # Allow user to specify Lightricks or other repo
        if "/" not in model_id and not os.path.isfile(model_id): 
             model_id = "Lightricks/LTX-Video"
        super().__init__(model_id, output_dir, quantization)
    
    def convert(self):
        print(f"Loading LTX-Video pipeline: {self.model_id}...")
        try:
            if os.path.isfile(self.model_id):
                print(f"Detected single file checkpoint: {self.model_id}")
                pipe = LTXPipeline.from_single_file(self.model_id, torch_dtype=torch.float32)
            else:
                pipe = LTXPipeline.from_pretrained(self.model_id)
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise e

        transformer = pipe.transformer
        transformer.eval()

        ml_model_dir = os.path.join(self.output_dir, "LTXVideo_Transformer.mlpackage")
        if os.path.exists(ml_model_dir):
            print(f"Model already exists at {ml_model_dir}, skipping.")
        else:
            self.convert_transformer(transformer, ml_model_dir)

        print(f"LTX conversion complete. Model saved to {self.output_dir}")

    def convert_transformer(self, transformer, ml_model_dir):
        print("Converting Transformer (FP32 trace)...")
        transformer = transformer.to(dtype=torch.float32)
        
        # Dimensions based on config
        # in_channels = 128
        in_channels = transformer.config.in_channels
        
        # Dummy Sizes
        # Latent space is compressed. 
        # For 1024x1024 input? 
        # Usually LTX behaves like SD3/Flux: (B, S, C).
        # S = (H/patch) * (W/patch) * (F/patch_t)
        # Config: patch_size=1, patch_size_t=1.
        # But this is on LATENTS.
        # VAE compression depends on VAE. Assuming VAE compression 32 (standard for LTX?)
        # Let's assume a small trace size.
        
        # Let's verify input shapes for trace.
        # Example: 1 frame, 64x64 latent.
        
        batch_size = 1
        latent_height = 32
        latent_width = 32
        latent_frames = 8
        seq_len = latent_height * latent_width * latent_frames 
        
        hidden_states = torch.randn(batch_size, seq_len, in_channels).float()
        
        # Text Encoder
        # T5: 4096 dim.
        text_seq_len = 128
        encoder_hidden_states = torch.randn(batch_size, text_seq_len, 4096).float()
        encoder_attention_mask = torch.ones(batch_size, text_seq_len).long() 
        # Note: LTX attention mask might be int64 or boolean. 
        # Source code: (1 - mask) * -10000. So it expects 1 for keep, 0 for mask.
        
        timestep = torch.tensor([1]).long()
        
        # Scalars
        num_frames = torch.tensor([latent_frames]).long()
        height = torch.tensor([latent_height]).long()
        width = torch.tensor([latent_width]).long()
        
        example_inputs = [
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
            num_frames,
            height,
            width
        ]
        
        wrapper = LTXModelWrapper(transformer)
        wrapper.eval()
        
        print("Tracing model...")
        traced_model = torch.jit.trace(wrapper, example_inputs, strict=False)
        
        print("Converting to Core ML...")
        # Define flexible shapes where possible
        
        # Range of resolutions?
        # Fixed shapes are safer for first pass, but users want flexibility.
        # S = H*W*F. This is hard to enforce as single symbolic dim if H,W,F vary independently.
        # But we can make S flexible.
        
        model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="hidden_states", shape=hidden_states.shape),
                ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
                ct.TensorType(name="timestep", shape=timestep.shape),
                ct.TensorType(name="encoder_attention_mask", shape=encoder_attention_mask.shape),
                ct.TensorType(name="num_frames", shape=num_frames.shape), # Scalar
                ct.TensorType(name="height", shape=height.shape), # Scalar
                ct.TensorType(name="width", shape=width.shape), # Scalar
            ],
            outputs=[ct.TensorType(name="sample")],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14
        )
        
        if self.quantization in ["int4", "4bit", "mixed"]:
            print("Applying Int4 quantization to Transformer...")
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                weight_threshold=512
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            model = ct.optimize.coreml.linear_quantize_weights(model, config)
            
        model.save(ml_model_dir)
        print(f"Transformer converted: {ml_model_dir}")
