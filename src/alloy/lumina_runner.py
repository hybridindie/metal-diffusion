import torch
import numpy as np
import coremltools as ct
from diffusers import Lumina2Pipeline
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

class LuminaCoreMLRunner:
    def __init__(self, model_dir, model_id="Alpha-VLLM/Lumina-Image-2.0", compute_unit="ALL"):
        self.model_dir = model_dir
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        logger.info(f"Loading generic components from {model_id}...")
        # Load scheduler, tokenizer, vae from generic pipeline
        self.pipe = Lumina2Pipeline.from_pretrained(model_id)
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.vae = self.pipe.vae.to(self.device)
        
        # Load Core ML Models
        logger.info("Loading Core ML models...")
        self.text_encoder_model = ct.models.MLModel(
            os.path.join(model_dir, "Gemma2_TextEncoder.mlpackage"), 
            compute_units=ct.ComputeUnit[compute_unit]
        )
        self.transformer_model = ct.models.MLModel(
            os.path.join(model_dir, "Lumina2_Transformer.mlpackage"), 
            compute_units=ct.ComputeUnit[compute_unit]
        )

    def encode_prompt(self, prompt):
        # Tokenize (Gemma)
        # Max length 256 as per converter
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.numpy().astype(np.int32)
        
        # Run Core ML Text Encoder
        prediction = self.text_encoder_model.predict({"input_ids": input_ids})
        last_hidden_state = torch.from_numpy(prediction["last_hidden_state"]).to(self.device)
        return last_hidden_state

    def generate(self, prompt, output_path, height=1024, width=1024, steps=20, guidance_scale=4.0):
        logger.info(f"Generating image for prompt: '{prompt}'")
        
        # 1. Encode Prompt
        prompt_embeds = self.encode_prompt(prompt)
        
        # Unconditional for guidance (empty string)
        neg_prompt_embeds = self.encode_prompt("")
        
        # Concatenate for classifier-free guidance if needed, 
        # BUT Core ML run usually does uncond/cond separately or batch 2 if API allows.
        # Here we'll do standard loop.
        
        # 2. Prepare Latents
        # VAE downsample factor 8
        batch_size = 1
        num_channels = self.pipe.transformer.config.in_channels
        h_latent = height // 8
        w_latent = width // 8
        
        latents = torch.randn(
            (batch_size, num_channels, h_latent, w_latent),
            device=self.device,
            dtype=torch.float32
        )
        
        self.scheduler.set_timesteps(steps)
        
        logger.info("Denoising...")
        for t in self.scheduler.timesteps:
            # Expand for guidance (2, C, H, W)
            # latents_input = torch.cat([latents] * 2)
            
            # Run Transformer (Core ML)
            # We run twice: once for cond, once for uncond (or batch if possible)
            # Dictionary input for Core ML
            
            # A. Unconditional
            uncond_inputs = {
                "hidden_states": latents.cpu().numpy().astype(np.float32),
                "encoder_hidden_states": neg_prompt_embeds.cpu().numpy().astype(np.float32),
                "timestep": np.array([float(t)], dtype=np.float32)
            }
            noise_pred_uncond = torch.from_numpy(
                self.transformer_model.predict(uncond_inputs)["hidden_states"]
            ).to(self.device)
            
            # B. Conditional
            cond_inputs = {
                "hidden_states": latents.cpu().numpy().astype(np.float32),
                "encoder_hidden_states": prompt_embeds.cpu().numpy().astype(np.float32),
                "timestep": np.array([float(t)], dtype=np.float32)
            }
            noise_pred_text = torch.from_numpy(
                self.transformer_model.predict(cond_inputs)["hidden_states"]
            ).to(self.device)
            
            # Guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # 3. Decode
        logger.info("Decoding...")
        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        
        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(image[0])
        
        pil_image.save(output_path)
        logger.info(f"Saved to {output_path}")
        return pil_image
