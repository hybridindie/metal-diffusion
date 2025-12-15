import os
import torch
import numpy as np
from PIL import Image
import coremltools as ct
from diffusers import DiffusionPipeline, WanPipeline
from python_coreml_stable_diffusion.pipeline import get_coreml_pipe

def run_sd_pipeline(model_dir, prompt, output_path, compute_unit="ALL", base_model="stabilityai/sd-turbo"):
    """
    Runs a Stable Diffusion Core ML pipeline.
    """
    print(f"Loading Base Pipeline (for Tokenizer/Scheduler) from {base_model}...")
    pytorch_pipe = DiffusionPipeline.from_pretrained(base_model)
    
    print(f"Loading Core ML Pipeline from {model_dir}...")
    # model_version needs to match the folder structure if strict, 
    # but usually get_coreml_pipe expects the parent of "compiled_models" or similar.
    # Actually get_coreml_pipe logic is complex. 
    # Let's try to manually instantiate if get_coreml_pipe is too rigid, 
    # or ensure model_dir is pointing to the right place.
    
    # We will use get_coreml_pipe by passing the directory containing the .mlpackages.
    # Note: get_coreml_pipe expects 'mlpackages_dir' to contain the model files directly.
    
    # Auto-detect model_version from filenames in model_dir
    import os
    import re
    
    # helper to find version string
    # Format: Stable_Diffusion_version_{version}_{component}.mlpackage
    # We look for unet
    model_version_str = base_model.replace("/", "_") # fallback
    
    try:
        files = os.listdir(model_dir)
        unet_files = [f for f in files if "unet.mlpackage" in f]
        if unet_files:
            # Example: Stable_Diffusion_version_models_sd-turbo_unet.mlpackage
            # prefix: Stable_Diffusion_version_
            # suffix: _unet.mlpackage
            match = re.search(r"Stable_Diffusion_version_(.*)_unet.mlpackage", unet_files[0])
            if match:
                model_version_str = match.group(1)
                print(f"Detected model version string: {model_version_str}")
    except Exception as e:
        print(f"Warning: Could not auto-detect model version from files: {e}")

    pipeline = get_coreml_pipe(
        pytorch_pipe=pytorch_pipe,
        mlpackages_dir=model_dir,
        model_version=model_version_str, # Use detected string
        compute_unit=compute_unit
    )
    
    print(f"Generating image for prompt: '{prompt}'")
    image = pipeline(
        prompt=prompt,
        num_inference_steps=2, # Turbo needs few steps
        guidance_scale=2.0 # Force standard flow to match compiled input shape (N=2)
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
        
    def generate(self, prompt, output_path, steps=20, height=512, width=512):
        print("Encoding prompt...")
        # Use underlying PyTorch pipe for text encoding
        prompt_embeds = self.pipe.encode_prompt(prompt, num_videos_per_prompt=1, do_classifier_free_guidance=True)[0]
        
        # Prepare latents
        # shape: (B, C, F, H/8, W/8)
        latents = torch.randn(1, 16, 1, height // 8, width // 8, device=self.device, dtype=torch.float16)
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps)
        
        print("Running Denoising Loop (Core ML)...")
        for t in scheduler.timesteps:
            # Prepare inputs for Core ML
            # Convert PyTorch tensors to numpy/PIL as expected by Core ML
            latent_np = latents.cpu().numpy().astype(np.float32)
            timestep_np = np.array([t.item()]).astype(np.int32)
            encoder_hidden_states_np = prompt_embeds.cpu().numpy().astype(np.float32) # Simplify shape handling
            
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
