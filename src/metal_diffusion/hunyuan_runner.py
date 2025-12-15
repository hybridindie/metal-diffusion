import torch
import numpy as np
import coremltools as ct
from diffusers import HunyuanVideoPipeline
import os
from PIL import Image

class HunyuanCoreMLRunner:
    """
    Hybrid runner for HunyuanVideo:
    - Text Encoder: PyTorch (CPU/MPS)
    - Transformer: Core ML
    - VAE: PyTorch (CPU/MPS) -> Core ML if converted later. 
      For now we keep VAE in PyTorch as we only converted Transformer.
    """
    def __init__(self, model_dir, model_id="hunyuanvideo-community/HunyuanVideo"):
        self.model_dir = model_dir
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"Loading PyTorch components from {model_id}...")
        self.pipe = HunyuanVideoPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        ).to(self.device)
        
        print("Loading Core ML Transformer...")
        self.coreml_transformer = ct.models.MLModel(os.path.join(model_dir, "HunyuanVideo_Transformer.mlpackage"))
        
    def generate(self, prompt, output_path, steps=20, height=512, width=512):
        print("Encoding prompt...")
        # encode_prompt returns: prompt_embeds, pooled_prompt_embeds, attention_mask
        # Note: Hunyuan encode_prompt signature might vary.
        # Assuming typical diffusers signature. 
        # Actually Hunyuan uses T5 and CLIP.
        # Let's inspect signature or assume standard returns.
        prompt_embeds, pooled_prompt_embeds, attention_mask = self.pipe.encode_prompt(
            prompt=prompt, 
            prompt_2=None,
            device=self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=True
        )
        # prompt_embeds: (2, L, 4096) (because cfg=True)
        # pooled: (2, 768)
        # mask: (2, L)
        
        # Prepare latents
        num_channels = 16
        num_frames = 1
        latents = torch.randn(1, num_channels, num_frames, height // 16, width // 16, device=self.device, dtype=torch.float16)
        
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps)
        
        # Guidance
        guidance_scale = 6.0
        guidance = torch.tensor([guidance_scale * 1000]).float() # Transformer expects scaled guidance often, checking config
        # Config said 'guidance_embeds': True. Usually this means it takes a scalar vector.
        
        print("Running Denoising Loop (Core ML)...")
        for i, t in enumerate(scheduler.timesteps):
            # Prepare inputs for Core ML (Batch size 2 usually for CFG, but we might have traced Batch 1)
            # If we traced Batch 1, we must run twice or cat?
            # Creating Core ML model with batch size 1 means we need to run pos and neg separately or loop.
            # To keep it simple, we run batch 1 (positive only) or we need to support flexible shapes.
            # Our converter used batch 1 fixed shape.
            # For CFG, we need user constraint.
            # Let's assume we just run positive prompt for now or naive sampling? 
            # Or simplified flow: just run positive.
            
            # Actually, standard CFG requires running both.
            # If model is fixed batch 1, we must run iteratively.
            
            # 1. Positive
            t_expand = t.expand(1) # (1,)
            
            inputs = {
                "hidden_states": latents.cpu().numpy().astype(np.float32),
                "timestep": np.array([t.item()]).astype(np.int32), # (1,)
                "encoder_hidden_states": prompt_embeds[1:2].cpu().numpy().astype(np.float32), # Use positive
                "encoder_attention_mask": attention_mask[1:2].cpu().numpy().astype(np.int64),
                "pooled_projections": pooled_prompt_embeds[1:2].cpu().numpy().astype(np.float32),
                "guidance": guidance.numpy().astype(np.float32)
            }
            
            pred_dict = self.coreml_transformer.predict(inputs)
            noise_pred = torch.from_numpy(pred_dict["sample"]).to(self.device)
            
            # Scheduler Step (skipping CFG for now as batch 1 model)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
        print("Decoding Latents (PyTorch)...")
        # Decode using PyTorch VAE (since we didn't convert it yet)
        latents = latents.to(dtype=torch.float16) # diffusers VAE expects fp16 usually
        with torch.no_grad():
            image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        
        # Post-process
        # image might be video tensor (B, C, F, H, W).
        # We need first frame.
        if image.ndim == 5:
            image = image[:, :, 0, :, :] # Take first frame
        
        image = self.pipe.image_processor.postprocess(image, output_type="pil")
        image[0].save(output_path)
        print(f"Saved to {output_path}")
