import os
import torch
import numpy as np
from PIL import Image
import coremltools as ct
from diffusers import DifftIm2ImPipeline, DiffusionPipeline, WanPipeline
from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline

def run_sd_pipeline(model_dir, prompt, output_path, compute_unit="ALL"):
    """
    Runs a Stable Diffusion Core ML pipeline.
    """
    print(f"Loading SD Pipeline from {model_dir}...")
    pipeline = CoreMLStableDiffusionPipeline(
        proxy_libs_dir=model_dir,
        compute_units=compute_unit
    )
    pipeline.load_resources()
    
    print(f"Generating image for prompt: '{prompt}'")
    image = pipeline(
        prompt=prompt,
        num_inference_steps=20,
    )["images"][0]
    
    image.save(output_path)
    print(f"Saved to {output_path}")

class WanCoreMLRunner:
    """
    Hybrid runner for Wan 2.1:
    - Text Encoder: PyTorch (CPU/MPS)
    - Transformer: Core ML
    - VAE: Core ML
    """
    def __init__(self, model_dir, model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers"):
        self.model_dir = model_dir
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"Loading PyTorch components (Text Encoder) from {model_id}...")
        # Load heavy T5 in 4bit or fp16 if possible to save RAM
        self.pipe = WanPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        ).to(self.device)
        
        print("Loading Core ML Transformer...")
        self.coreml_transformer = ct.models.MLModel(os.path.join(model_dir, "Wan2.1_Transformer.mlpackage"))
        
        print("Loading Core ML VAE...")
        self.coreml_vae = ct.models.MLModel(os.path.join(model_dir, "Wan2.1_VAE_Decoder.mlpackage"))
        
    def generate(self, prompt, output_path, steps=20):
        print("Encoding prompt...")
        # Use underlying PyTorch pipe for text encoding
        # This is simplified; Wan has complex prompting usually
        prompt_embeds = self.pipe.encode_prompt(prompt, num_videos_per_prompt=1, do_classifier_free_guidance=True)[0]
        
        # Prepare latents
        latents = torch.randn(1, 16, 1, 128, 128, device=self.device, dtype=torch.float16)
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps)
        
        print("Running Denoising Loop (Core ML)...")
        for t in scheduler.timesteps:
            # Prepare inputs for Core ML
            # Convert PyTorch tensors to numpy/PIL as expected by Core ML
            latent_np = latents.cpu().numpy().astype(np.float16)
            timestep_np = np.array([t.item()]).astype(np.float16)
            encoder_hidden_states_np = prompt_embeds.cpu().numpy().astype(np.float16) # Simplify shape handling
            
            # Prediction
            inputs = {
                "hidden_states": latent_np,
                "encoder_hidden_states": encoder_hidden_states_np,
                "timestep": timestep_np
            }
            
            # Run Core ML Inference
            pred_dict = self.coreml_transformer.predict(inputs)
            noise_pred = torch.from_numpy(pred_dict["sample"]).to(self.device)
            
            # Scheduler Step
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
        print("Decoding Latents (Core ML)...")
        # Decode
        latents_np = latents.cpu().numpy().astype(np.float16)
        image_dict = self.coreml_vae.predict({"latents": latents_np})
        # Assuming output is named 'var_xxxx' or similar, usually first output
        # Re-verify VAE output name if possible, or assume dictionary has 1 key
        out_key = list(image_dict.keys())[0]
        image_np = image_dict[out_key] 
        
        # Post-process
        image_np = (image_np / 2 + 0.5).clip(0, 1)
        image_np = (image_np * 255).astype(np.uint8)
        # Expected shape (1, 3, H, W) -> (H, W, 3)
        if image_np.ndim == 4:
            image_np = image_np[0]
        image_np = np.transpose(image_np, (1, 2, 0))
        
        img = Image.fromarray(image_np)
        img.save(output_path)
        print(f"Saved to {output_path}")
