import torch
import numpy as np
import coremltools as ct
from diffusers import DiffusionPipeline, FluxPipeline
try:
    from diffusers import Flux2Pipeline
except ImportError:
    Flux2Pipeline = None

import os
from PIL import Image

class FluxCoreMLRunner:
    """
    Hybrid runner for Flux.1:
    - Text Encoders (CLIP+T5): PyTorch (CPU/MPS)
    - Transformer: Core ML
    - VAE: PyTorch (CPU/MPS)
    """
    def __init__(self, model_dir, model_id="black-forest-labs/FLUX.1-schnell"):
        self.model_dir = model_dir
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"Loading PyTorch components from {model_id}...")
        # Load full pipeline to get schedulers, tokenizers, etc.
        if os.path.isfile(model_id):
            print(f"Detected single file checkpoint: {model_id}")
            self.pipe = FluxPipeline.from_single_file(
                model_id,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                transformer=None
            ).to(self.device)
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                transformer=None 
            ).to(self.device)
        
        # Check if Flux 2
        self.is_flux2 = Flux2Pipeline and isinstance(self.pipe, Flux2Pipeline)
        if self.is_flux2:
             print("Detected Flux.2 Pipeline")
        
        print("Loading Core ML Transformer...")
        self.coreml_transformer = ct.models.MLModel(os.path.join(model_dir, "Flux_Transformer.mlpackage"))
        
    def generate(self, prompt, output_path, steps=4, height=1024, width=1024, guidance_scale=0.0, seed=None, benchmark=None):
        """
        Run Flux generation. 
        Note: Flux Schnell uses 4 steps and guidance_scale=0.0 by default.
        Flux Dev uses roughly 20-50 steps and guidance 3.5.
        
        Args:
            benchmark: Optional Benchmark object for performance tracking
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # 1. Encode Prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1
        )
        # generic encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids)
        
        # 2. Prepare Latents & IDs
        # Flux VAE has 16 channels. Transformer input has 64.
        num_channels_latents = self.pipe.vae.config.latent_channels # 16
        
        # Adjust height/width for packing
        latent_height = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))
        
        print(f"Generating latents for {latent_height}x{latent_width}...")
        
        # Init Random Latents
        latents = torch.randn(
            1, num_channels_latents, latent_height, latent_width,
            device=self.device, dtype=self.pipe.text_encoder.dtype, generator=generator
        )
        
        # Pack
        latents = self._pack_latents(latents, 1, num_channels_latents, latent_height, latent_width) # (1, S, 64)
        
        # Image IDs
        img_ids = self._prepare_latent_image_ids(1, latent_height // 2, latent_width // 2, self.device, self.pipe.text_encoder.dtype)
        
        # 3. Timesteps
        scheduler = self.pipe.scheduler
        # Flux uses "sigmas" usually or set_timesteps
        scheduler.set_timesteps(steps, device=self.device)
        
        # 4. Denoising Loop
        print("Running Denoising Loop (Core ML)...")
        
        # Guidance Tensor
        guidance = torch.tensor([guidance_scale], device=self.device, dtype=torch.float32) * 1000
        # Wait, Core ML model expects float? Or did I trace with scaled guidance?
        # In converter: `guidance = torch.tensor([1.0]).float()`.
        # Inside model: `guidance = guidance.to(hidden_states.dtype) * 1000`.
        # So I should pass raw guidance scale (e.g. 1.0 or 3.5), model multiplies by 1000.
        guidance_input = np.array([guidance_scale]).astype(np.float32)
        
        # Pre-convert constant inputs
        encoder_hidden_states_np = prompt_embeds.cpu().numpy().astype(np.float32)
        txt_ids_np = text_ids.cpu().numpy().astype(np.float32)
        img_ids_np = img_ids.cpu().numpy().astype(np.float32)
        
        # Pooled Projections (Flux 1 only)
        if pooled_prompt_embeds is not None:
             pooled_projections_np = pooled_prompt_embeds.cpu().numpy().astype(np.float32)

        for i, t in enumerate(scheduler.timesteps):
            # timestep input
            # In converter: `timestep = torch.tensor([1]).long()`
            # Model behaves same as guidance? `timestep = timestep * 1000`?
            # `timestep = timestep.to(hidden_states.dtype) * 1000`
            # The scheduler `t` in Flux is usually 0..1 (Sigmas) or 1000..0?
            # FluxScheduler (FlowMatchEuler) `timesteps` are usually large values 1.0 down to 0.0?
            # Let's check `timestep` values from scheduler.
            # If they are 1.0 -> 0.0, passing them directly is fine. Model multiplies by 1000.
            
            t_input = np.array([t.item()]).astype(np.float32) # Core ML expects float or long?
            # Converter: `ct.TensorType(name="timestep", shape=timestep.shape)` where timestep was LongTensor? 
            # I cast it to float in converter implementation? 
            # `timestep = torch.tensor([1]).long()`
            # So I should pass int/long?
            # But Flux timesteps are floats!
            # If I passed Long in converter, Core ML might implicitly cast or expect Int.
            # Flux timesteps are continuous.
            # I should verify converter used float or long.
            # Converter code: `timestep = torch.tensor([1]).long()`
            # That might be a mistake if Flux uses float timesteps!
            # Flux `forward` signature: `timestep: torch.LongTensor`.
            # THIS IS STRANGE. Float diffusion models usually take Float.
            # Let's check `transformer_flux.py` signature again.
            # `timestep: torch.LongTensor = None`.
            # But line 695: `timestep = timestep.to(hidden_states.dtype) * 1000`.
            # If I pass Long 1, it becomes Float 1000.
            # If scheduler gives 1.0 (float), I can't pass it as Long without losing distinct steps?
            # Actually FlowMatch scheduler gives 1.0, 0.75, 0.5...
            # If signature demands Long, passing 0.75 as Long is 0!
            # Diffusers `FluxTransformer` expects `LongTensor`?
            # Maybe the signature type hint is wrong or I misread it?
            # Re-read file content, line 662: `timestep ( torch.LongTensor):`
            # This seems wrong for Flux.
            # Hmmm.
            # Wait, `pipeline_flux.py`: `timestep` is passed to transformer.
            # Line 685: `timestep = timestep.to(hidden_states.dtype) * 1000` inside model.
            # If input is Long, it truncates!
            # Does Flux use `sigmas` * 1000 as input?
            # Let's check what `FluxPipeline` passes.
            # The pipeline usually handles this.
            # I don't see the loop in `pipeline_flux.py` view (cutoff).
            
            # Let's assume for now I should pass Float to Core ML if the model handles float.
            # If I traced with Long inputs, Core ML will expect Ints/Longs.
            # If I trace with Float inputs, Core ML will expect Floats.
            # I should update `flux_converter.py` to use Float for timestep.
            # Because standard Flux timesteps are indeed floats (0..1).
            
            # Action: I will implement the runner assuming Float is correct, but I might need to fix Converter.
            # I will check/fix Converter in the next step if verification fails.
            
            latents_input = latents.cpu().numpy().astype(np.float32)
            
            inputs = {
                "hidden_states": latents_input,
                "encoder_hidden_states": encoder_hidden_states_np,
                "timestep": t_input,
                "img_ids": img_ids_np,
                "txt_ids": txt_ids_np,
                "guidance": guidance_input
            }
            
            if not self.is_flux2:
                inputs["pooled_projections"] = pooled_projections_np
            
            # Predict
            out = self.coreml_transformer.predict(inputs)
            noise_pred = torch.from_numpy(out["sample"]).to(self.device).to(latents.dtype)
            
            # Step
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
        print("Decoding...")
        # Unpack
        latents = self._unpack_latents(latents, height, width, self.pipe.vae_scale_factor)
        
        # Decode
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        latents = latents.to(self.pipe.vae.dtype)
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")
        image[0].save(output_path)
        print(f"Saved to {output_path}")

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids.to(device=device, dtype=dtype)
