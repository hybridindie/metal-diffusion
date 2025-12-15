import torch
import numpy as np
import coremltools as ct
import folder_paths
import comfy.model_management
import comfy.model_patcher
import comfy.lora
import comfy.utils

from diffusers import FluxTransformer2DModel, LTXVideoTransformer3DModel, WanTransformer3DModel

from alloy.flux_runner import FluxCoreMLRunner
from .video_wrappers import CoreMLLTXVideoWrapper, CoreMLWanVideoWrapper

class CoreMLFluxLoader:
    """Flux Image Generation - Core ML Accelerated"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model_path": (folder_paths.get_filename_list("unet"),)
        }}

    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Alloy"

    def load_coreml_model(self, model_path):
        base_path = folder_paths.get_full_path("unet", model_path)
        print(f"Loading Flux Core ML Model from: {base_path}")
        
        wrapper = CoreMLFluxWrapper(base_path)
        return (comfy.model_patcher.ModelPatcher(wrapper, load_device="cpu", offload_device="cpu"),)


class CoreMLLTXVideoLoader:
    """LTX-Video Generation - Core ML Accelerated"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model_path": (folder_paths.get_filename_list("unet"),),
            "num_frames": ("INT", {"default": 25, "min": 1, "max": 257, "step": 1})
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_coreml_model"
    CATEGORY = "Alloy/Video"

    def load_coreml_model(self, model_path, num_frames):
        base_path = folder_paths.get_full_path("unet", model_path)
        print(f"Loading LTX-Video Core ML Model from: {base_path}")
        
        wrapper = CoreMLLTXVideoWrapper(base_path, num_frames)
        return (comfy.model_patcher.ModelPatcher(wrapper, load_device="cpu", offload_device="cpu"),)


class CoreMLWanVideoLoader:
    """Wan Video Generation - Core ML Accelerated"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model_path": (folder_paths.get_filename_list("unet"),),
            "num_frames": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1})
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_coreml_model"
    CATEGORY = "Alloy/Video"

    def load_coreml_model(self, model_path, num_frames):
        base_path = folder_paths.get_full_path("unet", model_path)
        print(f"Loading Wan Core ML Model from: {base_path}")
        
        wrapper = CoreMLWanVideoWrapper(base_path, num_frames)
        return (comfy.model_patcher.ModelPatcher(wrapper, load_device="cpu", offload_device="cpu"),)


class CoreMLFluxWrapper(torch.nn.Module):
    """Adapts Flux Core ML model to ComfyUI's sampling interface"""
    def __init__(self, model_path):
        super().__init__()
        # Load Core ML model
        self.coreml_model = ct.models.MLModel(model_path)
        
        # Configuration (minimal needed for samplers)
        self.latent_format = comfy.latent_formats.SDXL() # Dummy default
        self.adm_channels = 0
        
    def forward(self, x, timestep, **kwargs):
        """
        Adapts standard UNet-style inputs to Flux Core ML packed inputs.
        x: Latents (B, C, H, W)
        timestep: Tensor (B,)
        kwargs: "transformer_options", "context", etc.
        """
        # Report progress if available
        transformer_options = kwargs.get("transformer_options", {})
        if hasattr(comfy.utils, 'ProgressBar'):
            # Try to update ComfyUI progress
            try:
                import comfy.model_management as mm
                if hasattr(mm, 'throw_exception_if_processing_interrupted'):
                    mm.throw_exception_if_processing_interrupted()
            except:
                pass
        
        return self._forward_flux(x, timestep, **kwargs)

    def _forward_flux(self, latents, timestep, **kwargs):
        """
        Adapts standard UNet-style inputs (Latents, Timestep, Context) 
        to Flux Core ML packed inputs.
        Reuse logic from FluxCoreMLRunner.
        """
        transformer_options = kwargs.get("transformer_options", {})
        cond = transformer_options.get("cond_or_uncond", [])
        
        # Flux expects packed latents.
        # Check input shape
        B, C, H, W = latents.shape
        
        # Prepare Flux Inputs
        # 1. Pack Latents
        # FluxCoreMLRunner._pack_latents is static input: (latents, bs, ch, h, w)
        # We need access to it.
        packed_latents = FluxCoreMLRunner._pack_latents(latents, B, C, H, W)
        packed_latents_np = packed_latents.cpu().numpy().astype(np.float32)

        # 2. Prepare IDs (Img, Txt)
        # Usually these come from Conditioning? 
        # ComfyUI passes 'c' or 'cond' in kwargs usually contains 'crossattn'?
        # For Flux, 'context' is the text embedding.
        context = kwargs.get("context", None) # (B, Seq, Dim)
        if context is None:
            # Fallback/Debug
             context = torch.zeros(B, 256, 4096)
        
        context_np = context.cpu().numpy().astype(np.float32)
             
        # Text IDs?
        # FluxCoreMLRunner generates simplified IDs.
        # We can regenerate them on CPU. 
        # Txt IDs: (SeqLen, 3)
        seq_len = context.shape[1]
        txt_ids = torch.zeros(seq_len, 3).float()
        txt_ids_np = txt_ids.cpu().numpy().astype(np.float32)
        
        # Img IDs
        img_ids = FluxCoreMLRunner._prepare_latent_image_ids(B, H // 2, W // 2, "cpu", torch.float32)
        img_ids_np = img_ids.cpu().numpy().astype(np.float32)
        
        # 3. Timestep
        # Comfy passes timestep as tensor.
        t_input = np.array([timestep[0].item()]).astype(np.float32)
        
        # 4. Guidance
        # Flux guidances? Usually handled by sampler, but Flux expects it as input.
        # transformer_options['cond'] might have 'guidance'?
        guidance_scale = 1.0 # Default
        guidance_input = np.array([guidance_scale]).astype(np.float32)
        
        # 5. Pooled Projections
        # Comfy usually passes this in 'y' or similar?
        pooled_projections = kwargs.get("y", None)
        inputs = {
            "hidden_states": packed_latents_np,
            "encoder_hidden_states": context_np,
            "timestep": t_input,
            "img_ids": img_ids_np,
            "txt_ids": txt_ids_np,
            "guidance": guidance_input
        }
        
        if pooled_projections is not None:
             inputs["pooled_projections"] = pooled_projections.cpu().numpy().astype(np.float32)
             
        # Run Core ML
        out = self.coreml_model.predict(inputs)
        
        # Unpack
        noise_pred = torch.from_numpy(out["sample"]).to(latents.device)
        
        # Unpack dimensions
        # FluxCoreMLRunner._unpack_latents(latents, height, width, scale_factor)
        # Scale factor=8 usually for Flux? 
        vae_scale_factor = 8 
        unpacked = FluxCoreMLRunner._unpack_latents(noise_pred, H*2*vae_scale_factor, W*2*vae_scale_factor, vae_scale_factor=1) 
        # H, W in unpack are original image dims? 
        # Wait, FluxCoreMLRunner._unpack_latents logic:
        # height = 2 * (int(height) // (vae_scale_factor * 2))
        # It expects "image height/width".
        # Comfy passes Latent Height/Width.
        # We need to reverse logic.
        
        # Re-implementing simplified unpack for arbitrary latent size:
        # Packed shape: (B, Patches, Channels)
        # Patches = (H/2 * W/2). Channels = 64 (16*4).
        # We want (B, 16, H, W).
        unpacked = noise_pred.view(B, H//2, W//2, C//4, 2, 2)
        unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
        unpacked = unpacked.reshape(B, C, H, W)
        
        return unpacked

NODE_CLASS_MAPPINGS = {
    "CoreMLFluxLoader": CoreMLFluxLoader,
    "CoreMLLTXVideoLoader": CoreMLLTXVideoLoader,
    "CoreMLWanVideoLoader": CoreMLWanVideoLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLFluxLoader": "Core ML Flux Loader (Image)",
    "CoreMLLTXVideoLoader": "Core ML LTX-Video Loader",
    "CoreMLWanVideoLoader": "Core ML Wan Video Loader"
}
