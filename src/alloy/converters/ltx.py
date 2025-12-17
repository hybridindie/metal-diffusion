import torch
import coremltools as ct
from diffusers import LTXVideoTransformer3DModel, LTXPipeline
from .base import ModelConverter
import os
from alloy.utils.coreml import safe_quantize_model
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
        print(f"Loading LTX Pipeline: {self.model_id}...")

        # Output setup
        ml_model_dir = os.path.join(self.output_dir, "LTXVideo_Transformer.mlpackage")
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
        
        try:
             # Try single file format if detected or specified
             if os.path.isfile(self.model_id):
                  pipe = LTXPipeline.from_single_file(self.model_id, torch_dtype=torch.float32)
             else:
                  pipe = LTXPipeline.from_pretrained(self.model_id, torch_dtype=torch.float32)
        except Exception as e:
             # Fallback logic or error
             print(f"Error loading pipeline: {e}")
             raise e
             
        try:
            import gc
            intermediate_model_path = os.path.join(intermediates_dir, "LTXVideo_Transformer.mlpackage")
            
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

        print(f"LTX conversion complete. Models saved to {self.output_dir}")

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
        num_frames = 1
        height = 32 # latents
        width = 32
        in_channels = transformer.config.in_channels
        
        hidden_states = torch.randn(batch_size, num_frames, in_channels, height, width).float()
        timestep = torch.tensor([1]).float() # or long? LTX usually float sigma or timestep
        
        # Encoder Hidden States (T5)
        text_dim = transformer.config.cross_attention_dim
        seq_len = 128 
        encoder_hidden_states = torch.randn(batch_size, seq_len, text_dim).float()
        
        # Attention mask?
        encoder_attention_mask = torch.ones(batch_size, seq_len).long()
        
        # Frame rate / FPS ?
        # LTX transformer forward: hidden_states, encoder_hidden_states, timestep, encoder_attention_mask, num_frames (scalar?)
        # Let's check signature wrapper
        
        example_inputs = [
            hidden_states, 
            encoder_hidden_states,
            timestep,
            encoder_attention_mask
        ]
        
        wrapper = LTXModelWrapper(transformer)
        wrapper.eval()
        
        print("Tracing model...")
        traced_model = torch.jit.trace(wrapper, example_inputs, strict=False)
        
        print("Converting to Core ML...")
        model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="hidden_states", shape=hidden_states.shape),
                ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
                ct.TensorType(name="timestep", shape=timestep.shape),
                ct.TensorType(name="encoder_attention_mask", shape=encoder_attention_mask.shape)
            ],
            outputs=[ct.TensorType(name="sample")],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14
        )
        
        if self.quantization in ["int4", "4bit", "mixed", "int8", "8bit"]:
            model = safe_quantize_model(model, self.quantization, intermediate_dir=intermediates_dir)
            
        model.save(ml_model_dir)
        print(f"Transformer converted: {ml_model_dir}")
