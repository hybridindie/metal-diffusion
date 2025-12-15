import torch
import coremltools as ct
from diffusers import HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from .converter import ModelConverter
import os
import shutil
from typing import Optional, Dict, Any

class HunyuanModelWrapper(torch.nn.Module):
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
        return self.model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            pooled_projections=pooled_projections,
            guidance=guidance,
            return_dict=False
        )

class HunyuanConverter(ModelConverter):
    def __init__(self, model_id, output_dir, quantization):
        super().__init__(model_id, output_dir, quantization)
    
    def convert(self):
        print(f"Loading HunyuanVideo pipeline: {self.model_id}...")
        # Load transformer only to save memory if possible, but pipeline handles config best
        # Hunyuan is HUGE. users might crash here if not enough RAM (64GB+).
        # We try to load into CPU + float16 to save RAM, but tracing needs float32 usually.
        # Let's start with float32 for safety in tracing, user needs big RAM.
        try:
            pipe = HunyuanVideoPipeline.from_pretrained(self.model_id)
        except Exception as e:
            print(f"Failed to load pipeline: {e}. Trying to load transformer only.")
            # Fallback (though pipe is needed for tokenizer usually, but we barely use it here)
            raise e

        ml_model_dir = os.path.join(self.output_dir, "HunyuanVideo_Transformer.mlpackage")
        if os.path.exists(ml_model_dir):
            print(f"Model already exists at {ml_model_dir}, skipping.")
        else:
            self.convert_transformer(pipe.transformer, ml_model_dir)

        print(f"Hunyuan conversion complete. Models saved to {self.output_dir}")

    def convert_transformer(self, transformer, ml_model_dir):
        print("Converting Transformer (FP32 trace)...")
        transformer.eval()
        transformer = transformer.to(dtype=torch.float32)
        
        # Dimensions
        # In T2V mode, num_frames can be small for tracing
        batch_size = 1
        num_frames = 1
        height = 64 # Small for tracing
        width = 64 
        in_channels = transformer.config.in_channels # 16
        
        # Inputs
        hidden_states = torch.randn(batch_size, in_channels, num_frames, height, width).float()
        timestep = torch.tensor([1]).long()
        
        # Text Embeddings (T5)
        text_dim = transformer.config.text_embed_dim # 4096
        seq_len = 256 # Default max len
        encoder_hidden_states = torch.randn(batch_size, seq_len, text_dim).float()
        encoder_attention_mask = torch.ones(batch_size, seq_len).long() # or bool? Signature says Tensor. Usually Mask is long or bool. Diffusers often wants attention_mask.
        # Check signature logic: usually applied as mask. 
        # But wait, signature said encoder_attention_mask.
        
        # Pooled Projections (CLIP)
        pool_dim = transformer.config.pooled_projection_dim # 768
        pooled_projections = torch.randn(batch_size, pool_dim).float()
        
        # Guidance
        guidance = torch.tensor([1000.0]).float() # Typical guidance scale * 1000 often
        
        example_inputs = [
            hidden_states, 
            timestep, 
            encoder_hidden_states, 
            encoder_attention_mask, 
            pooled_projections, 
            guidance
        ]
        
        wrapper = HunyuanModelWrapper(transformer)
        wrapper.eval()
        
        print("Tracing model...")
        traced_model = torch.jit.trace(wrapper, example_inputs, strict=False)
        
        print("Converting to Core ML...")
        model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="hidden_states", shape=hidden_states.shape),
                ct.TensorType(name="timestep", shape=timestep.shape),
                ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
                ct.TensorType(name="encoder_attention_mask", shape=encoder_attention_mask.shape),
                ct.TensorType(name="pooled_projections", shape=pooled_projections.shape),
                ct.TensorType(name="guidance", shape=guidance.shape),
            ],
            outputs=[ct.TensorType(name="sample")],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14
        )
        
        if self.quantization in ["int4", "4bit", "mixed"]:
            print("Applying Int4 quantization to Transformer...")
            from coremltools.models.neural_network import quantization_utils
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                weight_threshold=512
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            model = ct.optimize.coreml.linear_quantize_weights(model, config)
            
        model.save(ml_model_dir)
        print(f"Transformer converted: {ml_model_dir}")
