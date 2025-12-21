import torch
import coremltools as ct
import comfy.latent_formats


class CoreMLHunyuanVideoWrapper(torch.nn.Module):
    """Adapts HunyuanVideo Core ML model to ComfyUI's video sampling interface"""
    def __init__(self, model_path, num_frames):
        super().__init__()
        self.coreml_model = ct.models.MLModel(model_path)
        self.num_frames = num_frames

        # Video-specific configuration
        self.latent_format = comfy.latent_formats.SDXL()  # Placeholder
        self.adm_channels = 0

    def forward(self, x, timestep, **kwargs):
        """
        HunyuanVideo forward pass
        x: Video latents (B, C, F, H, W) - 5D tensor
        timestep: Tensor (B,)
        """
        # TODO: Implement Hunyuan-specific packing/unpacking
        raise NotImplementedError("HunyuanVideo ComfyUI integration coming soon! Use CLI for now.")


class CoreMLLuminaWrapper(torch.nn.Module):
    """Adapts Lumina Image 2.0 Core ML model to ComfyUI's sampling interface"""
    def __init__(self, model_path):
        super().__init__()
        self.coreml_model = ct.models.MLModel(model_path)
        self.model_path = model_path

        # Image-specific configuration
        self.latent_format = comfy.latent_formats.SDXL()  # Placeholder
        self.adm_channels = 0

        # Lumina config
        self.in_channels = 16  # Lumina uses 16 channels
        self.hidden_size = 2304  # From Lumina config

    def forward(self, x, timestep, **kwargs):
        """
        Lumina Image forward pass.

        Args:
            x: Image latents (B, C, H, W) - 4D tensor
            timestep: Tensor (B,) or scalar
            **kwargs: Additional arguments including:
                - context: Text embeddings (B, seq_len, dim)
                - y: Pooled embeddings (unused for Lumina)

        Returns:
            Noise prediction tensor (B, C, H, W)
        """
        import numpy as np

        B, C, H, W = x.shape

        # Prepare hidden_states (latents)
        hidden_states_np = x.cpu().numpy().astype(np.float32)

        # Get text embeddings from context
        context = kwargs.get("context", None)
        if context is None:
            # Default empty conditioning
            context = torch.zeros(B, 256, self.hidden_size, device=x.device)
        encoder_hidden_states_np = context.cpu().numpy().astype(np.float32)

        # Prepare timestep
        if isinstance(timestep, torch.Tensor):
            t_val = timestep[0].item() if timestep.dim() > 0 else timestep.item()
        else:
            t_val = float(timestep)
        timestep_np = np.array([t_val], dtype=np.float32)

        # Build inputs dict
        inputs = {
            "hidden_states": hidden_states_np,
            "encoder_hidden_states": encoder_hidden_states_np,
            "timestep": timestep_np,
        }

        # Run Core ML model
        out = self.coreml_model.predict(inputs)

        # Get output - Lumina uses "hidden_states" as output key
        # But the pipeline model may use "sample" after assembly
        if "sample" in out:
            output_np = out["sample"]
        elif "hidden_states" in out:
            output_np = out["hidden_states"]
        else:
            # Fallback to first output
            output_np = list(out.values())[0]

        # Convert back to tensor
        noise_pred = torch.from_numpy(output_np).to(x.device, dtype=x.dtype)

        # Ensure output shape matches input latent shape
        if noise_pred.shape != x.shape:
            # Lumina may output in sequence format, need to reshape
            # From (B, seq_len, dim) back to (B, C, H, W)
            if len(noise_pred.shape) == 3:
                # Reshape from sequence to spatial
                # seq_len should be H*W, dim should be C
                noise_pred = noise_pred.view(B, H, W, C).permute(0, 3, 1, 2)

        return noise_pred


class CoreMLLTXVideoWrapper(torch.nn.Module):
    """Adapts LTX-Video Core ML model to ComfyUI's video sampling interface"""
    def __init__(self, model_path, num_frames):
        super().__init__()
        self.coreml_model = ct.models.MLModel(model_path)
        self.num_frames = num_frames
        
        # Video-specific configuration
        self.latent_format = comfy.latent_formats.SDXL()  # Placeholder
        self.adm_channels = 0
        
    def forward(self, x, timestep, **kwargs):
        """
        LTX-Video forward pass
        x: Video latents (B, C, F, H, W) - 5D tensor
        timestep: Tensor (B,)
        """
        # TODO: Implement LTX-specific packing/unpacking
        # For now, placeholder that will error gracefully
        raise NotImplementedError("LTX-Video ComfyUI integration coming soon! Use CLI for now.")


class CoreMLWanVideoWrapper(torch.nn.Module):
    """Adapts Wan Core ML model to ComfyUI's video sampling interface"""
    def __init__(self, model_path, num_frames):
        super().__init__()
        self.coreml_model = ct.models.MLModel(model_path)
        self.num_frames = num_frames
        
        # Video-specific configuration
        self.latent_format = comfy.latent_formats.SDXL()  # Placeholder
        self.adm_channels = 0
        
    def forward(self, x, timestep, **kwargs):
        """
        Wan Video forward pass
        x: Video latents (B, C, F, H, W) - 5D tensor
        timestep: Tensor (B,)
        """
        # TODO: Implement Wan-specific packing/unpacking
        # For now, placeholder that will error gracefully
        raise NotImplementedError("Wan Video ComfyUI integration coming soon! Use CLI for now.")
