import torch
import numpy as np
import coremltools as ct
from diffusers import LTXPipeline
import os
from PIL import Image

class LTXCoreMLRunner:
    """
    Hybrid runner for LTX-Video:
    - Text Encoder: PyTorch (CPU/MPS)
    - Transformer: Core ML
    - VAE: PyTorch (CPU/MPS)
    """
    def __init__(self, model_dir, model_id="Lightricks/LTX-Video"):
        self.model_dir = model_dir
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"Loading PyTorch components from {model_id}...")
        if os.path.isfile(model_id):
            print(f"Detected single file checkpoint: {model_id}")
            self.pipe = LTXPipeline.from_single_file(
                model_id, 
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.pipe = LTXPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16
            ).to(self.device)
        
        print("Loading Core ML Transformer...")
        self.coreml_transformer = ct.models.MLModel(os.path.join(model_dir, "LTXVideo_Transformer.mlpackage"))
        
    def generate(self, prompt, output_path, steps=20, height=512, width=512, num_frames=8):
        print("Encoding prompt...")
        prompt_embeds, prompt_attention_mask = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None, 
            device=self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=True
        ) # (2, L, 4096), (2, L)
        
        num_channels = 128
        patch_size = 1
        patch_size_t = 1
        
        vae_spatial_compression = 32
        vae_temporal_compression = 8
        
        latent_height = height // vae_spatial_compression
        latent_width = width // vae_spatial_compression
        latent_frames = (num_frames - 1) // vae_temporal_compression + 1
        
        # 1. Random Init (B=1)
        latents = torch.randn(1, 128, latent_frames, latent_height, latent_width, device=self.device, dtype=torch.float16)
        
        # 2. Pack
        latents = self._pack_latents(latents, patch_size, patch_size_t) # (1, S, 128)
        
        # 3. Timesteps
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps)
        
        # 4. RoPE Scale (Not used in Core ML input map currently, but kept for reference)
        frame_rate = 25
        
        print("Running Denoising Loop (Core ML)...")
        for i, t in enumerate(scheduler.timesteps):
            # Input Prep
            timestep = torch.tensor([t.item()]).long() # (1,)
            latents_input = latents.cpu().numpy().astype(np.float32)
            
            # 1. Uncond Pass
            inputs_uncond = {
                "hidden_states": latents_input,
                "encoder_hidden_states": prompt_embeds[0:1].cpu().numpy().astype(np.float32),
                "timestep": np.array([t.item()]).astype(np.int32),
                "encoder_attention_mask": prompt_attention_mask[0:1].cpu().numpy().astype(np.int64),
                "num_frames": np.array([latent_frames]).astype(np.int32),
                "height": np.array([latent_height]).astype(np.int32),
                "width": np.array([latent_width]).astype(np.int32),
            }
            
            out_uncond = self.coreml_transformer.predict(inputs_uncond)
            noise_uncond = torch.from_numpy(out_uncond["sample"]).to(self.device).to(dtype=torch.float16)
            
            # 2. Text Pass
            inputs_text = {
                "hidden_states": latents_input,
                "encoder_hidden_states": prompt_embeds[1:2].cpu().numpy().astype(np.float32),
                "timestep": np.array([t.item()]).astype(np.int32),
                "encoder_attention_mask": prompt_attention_mask[1:2].cpu().numpy().astype(np.int64),
                "num_frames": np.array([latent_frames]).astype(np.int32),
                "height": np.array([latent_height]).astype(np.int32),
                "width": np.array([latent_width]).astype(np.int32),
            }
            out_text = self.coreml_transformer.predict(inputs_text)
            noise_text = torch.from_numpy(out_text["sample"]).to(self.device).to(dtype=torch.float16)
            
            # Classifier Free Guidance
            guidance_scale = 3.0
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            
            # Step
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
        print("Decoding (PyTorch)...")
        # Unpack
        latents = self._unpack_latents(latents, latent_frames, latent_height, latent_width, patch_size, patch_size_t)
        
        # Denormalize
        latents = self._denormalize_latents(latents, self.pipe.vae.latents_mean, self.pipe.vae.latents_std, self.pipe.vae.config.scaling_factor)
        
        # Decode
        latents = latents.to(dtype=torch.float16)
        with torch.no_grad():
            video = self.pipe.vae.decode(latents, return_dict=False)[0]
            
        # Post process video -> Image (first frame)
        video = self.pipe.video_processor.postprocess_video(video, output_type="pil")
        video[0][0].save(output_path)
        print(f"Saved to {output_path}")

    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents
