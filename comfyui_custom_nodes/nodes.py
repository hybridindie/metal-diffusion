import torch
import numpy as np
import coremltools as ct
import folder_paths
import comfy.model_management
import comfy.model_patcher
import comfy.lora
import comfy.utils

from diffusers import FluxTransformer2DModel, LTXVideoTransformer3DModel, WanTransformer3DModel

from alloy.runners.flux import FluxCoreMLRunner
from .video_wrappers import (
    CoreMLLTXVideoWrapper,
    CoreMLWanVideoWrapper,
    CoreMLHunyuanVideoWrapper,
    CoreMLLuminaWrapper,
)

class CoreMLFluxLoader:
    """Flux Image Generation - Core ML Accelerated"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model_path": (folder_paths.get_filename_list("unet"),)
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_coreml_model"
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
    def __init__(self, model_path, coreml_model=None):
        super().__init__()
        # Load Core ML model
        if coreml_model:
            self.coreml_model = coreml_model
        else:
            self.coreml_model = ct.models.MLModel(model_path)
        
        # Configuration
        self.latent_format = comfy.latent_formats.SDXL() # Dummy default
        self.adm_channels = 0
        self.controlnet_residuals = None
        self.active_controlnets = [] # List of dicts: {model, image, strength}
        
    def clone_with_residuals(self, residuals):
        """Create a shallow copy"""
        new_wrapper = CoreMLFluxWrapper(None, coreml_model=self.coreml_model)
        new_wrapper.latent_format = self.latent_format
        new_wrapper.adm_channels = self.adm_channels
        new_wrapper.controlnet_residuals = residuals
        new_wrapper.active_controlnets = list(self.active_controlnets) # Shallow copy list
        return new_wrapper

    def forward(self, x, timestep, **kwargs):
        """
        Adapts standard UNet-style inputs to Flux Core ML packed inputs.
        """
        # Report progress logic...
        transformer_options = kwargs.get("transformer_options", {})
        if hasattr(comfy.utils, 'ProgressBar'):
             try:
                import comfy.model_management as mm
                if hasattr(mm, 'throw_exception_if_processing_interrupted'):
                    mm.throw_exception_if_processing_interrupted()
             except Exception:
                # Ignore errors interacting with ComfyUI progress bar (e.g. if not running in Comfy)
                pass
        
        return self._forward_flux(x, timestep, **kwargs)

    def _forward_flux(self, latents, timestep, **kwargs):
        # 1. Prepare Inputs (Same as before)
        B, C, H, W = latents.shape
        packed_latents = FluxCoreMLRunner._pack_latents(latents, B, C, H, W)
        packed_latents_np = packed_latents.cpu().numpy().astype(np.float32)

        context = kwargs.get("context", None)
        if context is None:
             context = torch.zeros(B, 256, 4096)
        context_np = context.cpu().numpy().astype(np.float32)
             
        txt_ids = torch.zeros(context.shape[1], 3).float()
        txt_ids_np = txt_ids.cpu().numpy().astype(np.float32)
        
        img_ids = FluxCoreMLRunner._prepare_latent_image_ids(B, H // 2, W // 2, "cpu", torch.float32)
        img_ids_np = img_ids.cpu().numpy().astype(np.float32)
        
        t_input = np.array([timestep[0].item()]).astype(np.float32)
        
        guidance_scale = 1.0 # Default
        guidance_input = np.array([guidance_scale]).astype(np.float32)
        
        inputs = {
            "hidden_states": packed_latents_np,
            "encoder_hidden_states": context_np,
            "timestep": t_input,
            "img_ids": img_ids_np,
            "txt_ids": txt_ids_np,
            "guidance": guidance_input
        }
        
        pooled_projections = kwargs.get("y", None)
        if pooled_projections is not None:
             inputs["pooled_projections"] = pooled_projections.cpu().numpy().astype(np.float32)

        # 2. Run Active ControlNets
        accumulated_residuals = {}
        
        for cn in self.active_controlnets:
            cn_model = cn["model"] # ct.models.MLModel
            cn_image = cn["image"] # Tensor (B, ...)
            strength = cn["strength"]
            
            # Prepare CN Inputs
            # Assumptions: 
            # - CN Image is already packed/compatible shape?
            # - CN needs same IDs/Time/etc as Flux
            # We reuse the `inputs` dict but replace relevant parts if needed?
            # CN expects "controlnet_cond" instead of "hidden_states" (or in addition?)
            # CN inputs: controlnet_cond, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, guidance.
            
            cn_inputs = inputs.copy()
            
            # Prepare controlnet_cond
            # If cn_image is raw (B, H, W, C), check if we need to pack it?
            # If FluxControlNetConverter uses same packing as Flux, we should pack it.
            # Assuming cn_image is (B, H, W, C) from Comfy.
            # But wait, Comfy Image is (B, H, W, C) usually.
            # If it matches Latents H, W?
            # If user passed Latents, it might be (B, C, H, W).
            # Let's assume user passes Latents-compatible tensor for now (B, C, H, W).
            # And we Pack it.
            
            # Checking shape:
            if isinstance(cn_image, torch.Tensor):
                 # Convert to numpy and pack if needed.
                 # If shapes match latents:
                 if cn_image.shape[-2:] == (H, W): # Matches H, W
                      # Pack it
                      packed_cn = FluxCoreMLRunner._pack_latents(cn_image, B, cn_image.shape[1], H, W)
                      cn_inputs["controlnet_cond"] = packed_cn.cpu().numpy().astype(np.float32)
                 else:
                      # Pass as is (maybe it's already packed?)
                      cn_inputs["controlnet_cond"] = cn_image.cpu().numpy().astype(np.float32)
            else:
                 cn_inputs["controlnet_cond"] = cn_image # Already numpy?

            # Run CN
            cn_out = cn_model.predict(cn_inputs)
            
            # Accumulate Residuals & Scale
            # keys: "c_double_0", ...
            for k, v in cn_out.items():
                if k.startswith("c_"):
                    scaled = v * strength
                    if k in accumulated_residuals:
                        accumulated_residuals[k] += scaled
                    else:
                        accumulated_residuals[k] = scaled
                        
        # Merge accumulated residuals into main inputs
        if accumulated_residuals:
            inputs.update(accumulated_residuals)

        # Inject Pre-existing Residuals (from previous nodes?)
        if self.controlnet_residuals:
            inputs.update(self.controlnet_residuals)
             
        # Run Core ML
        out = self.coreml_model.predict(inputs)
        
        # Unpack
        noise_pred = torch.from_numpy(out["sample"]).to(latents.device)
        
        # Unpack dimensions
        unpacked = noise_pred.view(B, H//2, W//2, C//4, 2, 2)
        unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
        unpacked = unpacked.reshape(B, C, H, W)
        
        return unpacked

class CoreMLHunyuanVideoLoader:
    """HunyuanVideo Generation - Core ML Accelerated"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_path": (folder_paths.get_filename_list("unet"),),
            "num_frames": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1})
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_coreml_model"
    CATEGORY = "Alloy/Video"

    def load_coreml_model(self, model_path, num_frames):
        base_path = folder_paths.get_full_path("unet", model_path)
        print(f"Loading HunyuanVideo Core ML Model from: {base_path}")

        wrapper = CoreMLHunyuanVideoWrapper(base_path, num_frames)
        return (comfy.model_patcher.ModelPatcher(wrapper, load_device="cpu", offload_device="cpu"),)


class CoreMLLuminaLoader:
    """Lumina Image 2.0 - Core ML Accelerated"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_path": (folder_paths.get_filename_list("unet"),),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_coreml_model"
    CATEGORY = "Alloy"

    def load_coreml_model(self, model_path):
        base_path = folder_paths.get_full_path("unet", model_path)
        print(f"Loading Lumina Image 2.0 Core ML Model from: {base_path}")

        wrapper = CoreMLLuminaWrapper(base_path)
        return (comfy.model_patcher.ModelPatcher(wrapper, load_device="cpu", offload_device="cpu"),)


class CoreMLControlNetLoader:
    """Loads a Converted ControlNet Model (.mlpackage)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "controlnet_path": (folder_paths.get_filename_list("controlnet"),)
        }}

    RETURN_TYPES = ("COREML_CONTROLNET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "Alloy"

    def load_controlnet(self, controlnet_path):
        base_path = folder_paths.get_full_path("controlnet", controlnet_path)
        print(f"Loading Core ML ControlNet: {base_path}")
        model = ct.models.MLModel(base_path)
        return (model,)

class CoreMLApplyControlNet:
    """Applies Core ML ControlNet to a Core ML Flux Model"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
            "model": ("MODEL",),
            "controlnet": ("COREML_CONTROLNET",),
            "image": ("IMAGE",), # Expects latent-compatible image? or Tensor
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "Alloy"

    def apply_controlnet(self, model, controlnet, image, strength):
        print("Applying Core ML ControlNet...")
        wrapper = model.model
        if not isinstance(wrapper, CoreMLFluxWrapper):
            print("Error: Model is not a CoreMLFluxWrapper. Skipping ControlNet.")
            return (model,)
            
        new_wrapper = wrapper.clone_with_residuals(wrapper.controlnet_residuals) 
        
        # Pre-process image/latents?
        # If image is Comfy Image (1, H, W, 3) [0..1]
        # and we need (1, 4, H/8, W/8)?
        # For full correctness we need VAE Encode usually.
        # But here we assume user passes the correct tensor (e.g. from VAE Encode node).
        # VAE Encode Node returns LATENT.
        # ApplyControlNet input type "IMAGE" refers to pixels.
        # If we change input type to "LATENT", it would verify.
        # For flexibility, let's keep "IMAGE" but check type.
        # If tensor is (B, C, H, W), store it.
        
        if isinstance(image, dict) and "samples" in image:
             # It's a LATENT dict
             img_tensor = image["samples"]
        else:
             # Checks if it is permuted? Comfy Image is channel last (B, H, W, C).
             # We convert to channel first (B, C, H, W)
             if len(image.shape) == 4 and image.shape[-1] in [1, 3, 4]:
                  img_tensor = image.permute(0, 3, 1, 2)
             else:
                  img_tensor = image
        
        new_wrapper.active_controlnets.append({
            "model": controlnet,
            "image": img_tensor, 
            "strength": strength
        })
        
        return (comfy.model_patcher.ModelPatcher(new_wrapper, load_device="cpu", offload_device="cpu"),)


NODE_CLASS_MAPPINGS = {
    "CoreMLFluxLoader": CoreMLFluxLoader,
    "CoreMLLTXVideoLoader": CoreMLLTXVideoLoader,
    "CoreMLWanVideoLoader": CoreMLWanVideoLoader,
    "CoreMLHunyuanVideoLoader": CoreMLHunyuanVideoLoader,
    "CoreMLLuminaLoader": CoreMLLuminaLoader,
    "CoreMLControlNetLoader": CoreMLControlNetLoader,
    "CoreMLApplyControlNet": CoreMLApplyControlNet
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLFluxLoader": "Core ML Flux Loader (Image)",
    "CoreMLLTXVideoLoader": "Core ML LTX-Video Loader",
    "CoreMLWanVideoLoader": "Core ML Wan Video Loader",
    "CoreMLHunyuanVideoLoader": "Core ML Hunyuan Video Loader",
    "CoreMLLuminaLoader": "Core ML Lumina Image 2.0 Loader",
    "CoreMLControlNetLoader": "Core ML ControlNet Loader",
    "CoreMLApplyControlNet": "Apply Core ML ControlNet"
}
