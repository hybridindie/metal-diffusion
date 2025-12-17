import torch
import coremltools as ct
from diffusers import HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from .base import ModelConverter
import os
from alloy.utils.coreml import safe_quantize_model
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
        
        # Output setup
        ml_model_dir = os.path.join(self.output_dir, "HunyuanVideo_Transformer.mlpackage")
        if os.path.exists(ml_model_dir):
            print(f"Model already exists at {ml_model_dir}, skipping.")
            return

        intermediates_dir = os.path.join(self.output_dir, "intermediates")
        os.makedirs(intermediates_dir, exist_ok=True)
        
        # Download Sources
        if "/" in self.model_id and not os.path.isfile(self.model_id):
            print("Downloading original model weights to output folder...")
            try:
                from huggingface_hub import snapshot_download
                source_dir = os.path.join(self.output_dir, "source")
                snapshot_download(
                    repo_id=self.model_id,
                    local_dir=source_dir,
                    allow_patterns=["transformer/*", "config.json", "*.json", "*.safetensors"],
                    ignore_patterns=["*.msgpack", "*.bin"]
                )
                self.model_id = source_dir
                print(f"Originals saved to: {source_dir}")
            except Exception as e:
                print(f"Warning: Failed to download source originals ({e}). Proceeding with remote load...")

        # Load Pipeline (CPU/Low RAM if possible)
        try:
             pipe = HunyuanVideoPipeline.from_pretrained(self.model_id)
        except Exception as e:
            print(f"Failed to load pipeline: {e}.")
            raise e

        try:
            import gc
            intermediate_model_path = os.path.join(intermediates_dir, "HunyuanVideo_Transformer.mlpackage")
            
            # Resume Check
            if os.path.exists(intermediate_model_path):
                 try:
                     print(f"Checking existing intermediate at {intermediate_model_path}...")
                     ct.models.MLModel(intermediate_model_path, compute_units=ct.ComputeUnit.CPU_ONLY)
                     print("Found valid intermediate. Resuming...")
                 except:
                     print("Invalid intermediate found. Re-converting...")
                     shutil.rmtree(intermediate_model_path)
                     self.convert_transformer(pipe.transformer, intermediate_model_path, intermediates_dir)
            else:
                 self.convert_transformer(pipe.transformer, intermediate_model_path, intermediates_dir)
            
            # Move to final
            print(f"Moving to final location: {ml_model_dir}...")
            shutil.move(intermediate_model_path, ml_model_dir)
            
            # Cleanup
            print("Cleaning up intermediates...")
            del pipe
            gc.collect()
            shutil.rmtree(intermediates_dir)

        except Exception as e:
            print(f"Conversion failed. Intermediates left in {intermediates_dir}")
            raise e

        print(f"Hunyuan conversion complete. Models saved to {self.output_dir}")

    def convert_transformer(self, transformer, ml_model_dir, intermediates_dir):
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
        
        if self.quantization in ["int4", "4bit", "mixed", "int8", "8bit"]:
            model = safe_quantize_model(model, self.quantization, intermediate_dir=intermediates_dir)
            
        model.save(ml_model_dir)
        print(f"Transformer converted: {ml_model_dir}")
