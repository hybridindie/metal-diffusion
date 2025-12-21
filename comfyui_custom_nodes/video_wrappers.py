import torch
import coremltools as ct
import comfy.latent_formats


class CoreMLHunyuanVideoWrapper(torch.nn.Module):
    """Adapts HunyuanVideo Core ML model to ComfyUI's video sampling interface"""
    def __init__(self, model_path, num_frames):
        super().__init__()
        self.coreml_model = ct.models.MLModel(model_path)
        self.num_frames = num_frames
        self.model_path = model_path

        # Video-specific configuration
        self.latent_format = comfy.latent_formats.SDXL()  # Placeholder
        self.adm_channels = 0

        # Hunyuan config
        self.num_channels = 16
        self.vae_scale_factor = 16  # Hunyuan uses 16x downscaling
        self.default_guidance_scale = 6.0

    def forward(self, x, timestep, **kwargs):
        """
        HunyuanVideo forward pass.

        Args:
            x: Video latents (B, C, F, H, W) - 5D tensor
            timestep: Tensor (B,) or scalar
            **kwargs: Additional arguments including:
                - context: Text embeddings (B, seq_len, dim)
                - attention_mask: Encoder attention mask (B, seq_len)
                - y: Pooled text embeddings (B, dim)
                - guidance_scale: CFG guidance scale (float)

        Returns:
            Noise prediction tensor (B, C, F, H, W)
        """
        import numpy as np

        B = x.shape[0]

        # Hunyuan uses 5D latents directly
        hidden_states_np = x.cpu().numpy().astype(np.float32)

        # Get text embeddings from context
        context = kwargs.get("context", None)
        if context is None:
            # Default empty conditioning
            context = torch.zeros(B, 256, 4096, device=x.device, dtype=x.dtype)
        encoder_hidden_states_np = context.cpu().numpy().astype(np.float32)

        # Get attention mask
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is None:
            seq_len = context.shape[1]
            attention_mask = torch.ones(B, seq_len, device=x.device, dtype=torch.int64)
        encoder_attention_mask_np = attention_mask.cpu().numpy().astype(np.int64)

        # Get pooled projections (y)
        pooled = kwargs.get("y", None)
        if pooled is None:
            pooled = torch.zeros(B, 1024, device=x.device, dtype=x.dtype)
        pooled_projections_np = pooled.cpu().numpy().astype(np.float32)

        # Get guidance scale
        guidance_scale = kwargs.get("guidance_scale", self.default_guidance_scale)
        # Hunyuan uses guidance * 1000 format
        guidance_np = np.array([guidance_scale * 1000]).astype(np.float32)

        # Prepare timestep
        if isinstance(timestep, torch.Tensor):
            t_val = timestep[0].item() if timestep.dim() > 0 else timestep.item()
        else:
            t_val = float(timestep)
        timestep_np = np.array([int(t_val)]).astype(np.int32)

        # Build inputs dict
        inputs = {
            "hidden_states": hidden_states_np,
            "timestep": timestep_np,
            "encoder_hidden_states": encoder_hidden_states_np,
            "encoder_attention_mask": encoder_attention_mask_np,
            "pooled_projections": pooled_projections_np,
            "guidance": guidance_np,
        }

        # Run Core ML model
        out = self.coreml_model.predict(inputs)

        # Get output
        if "sample" in out:
            output_np = out["sample"]
        else:
            output_np = list(out.values())[0]

        # Convert back to tensor
        noise_pred = torch.from_numpy(output_np).to(x.device, dtype=x.dtype)

        # Ensure output shape matches input shape
        if noise_pred.shape != x.shape:
            noise_pred = noise_pred.view(x.shape)

        return noise_pred


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
        self.model_path = model_path

        # Video-specific configuration
        self.latent_format = comfy.latent_formats.SDXL()  # Placeholder
        self.adm_channels = 0

        # LTX config
        self.num_channels = 128
        self.patch_size = 1
        self.patch_size_t = 1
        self.vae_spatial_compression = 32
        self.vae_temporal_compression = 8

    def forward(self, x, timestep, **kwargs):
        """
        LTX-Video forward pass.

        Args:
            x: Video latents (B, C, F, H, W) - 5D tensor
            timestep: Tensor (B,) or scalar
            **kwargs: Additional arguments including:
                - context: Text embeddings (B, seq_len, 4096)
                - attention_mask: Encoder attention mask (B, seq_len)

        Returns:
            Noise prediction tensor (B, C, F, H, W)
        """
        import numpy as np

        B, C, F, H, W = x.shape

        # Pack latents for transformer
        latents_packed = self._pack_latents(x)
        hidden_states_np = latents_packed.cpu().numpy().astype(np.float32)

        # Get text embeddings from context
        context = kwargs.get("context", None)
        if context is None:
            # Default empty conditioning
            context = torch.zeros(B, 256, 4096, device=x.device, dtype=x.dtype)
        encoder_hidden_states_np = context.cpu().numpy().astype(np.float32)

        # Get attention mask
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is None:
            # Default: all tokens attended
            seq_len = context.shape[1]
            attention_mask = torch.ones(B, seq_len, device=x.device, dtype=torch.int64)
        encoder_attention_mask_np = attention_mask.cpu().numpy().astype(np.int64)

        # Prepare timestep
        if isinstance(timestep, torch.Tensor):
            t_val = timestep[0].item() if timestep.dim() > 0 else timestep.item()
        else:
            t_val = float(timestep)
        timestep_np = np.array([int(t_val)]).astype(np.int32)

        # Latent dimensions for transformer
        latent_frames = F
        latent_height = H
        latent_width = W

        # Build inputs dict
        inputs = {
            "hidden_states": hidden_states_np,
            "encoder_hidden_states": encoder_hidden_states_np,
            "timestep": timestep_np,
            "encoder_attention_mask": encoder_attention_mask_np,
            "num_frames": np.array([latent_frames]).astype(np.int32),
            "height": np.array([latent_height]).astype(np.int32),
            "width": np.array([latent_width]).astype(np.int32),
        }

        # Run Core ML model
        out = self.coreml_model.predict(inputs)

        # Get output
        if "sample" in out:
            output_np = out["sample"]
        else:
            output_np = list(out.values())[0]

        # Convert back to tensor (still packed)
        noise_pred_packed = torch.from_numpy(output_np).to(x.device, dtype=x.dtype)

        # Unpack back to (B, C, F, H, W)
        noise_pred = self._unpack_latents(noise_pred_packed, F, H, W)

        return noise_pred

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Pack video latents for transformer input."""
        batch_size, num_channels, num_frames, height, width = latents.shape
        patch_size = self.patch_size
        patch_size_t = self.patch_size_t

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

    def _unpack_latents(
        self,
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Unpack transformer output to video latents."""
        batch_size = latents.size(0)
        patch_size = self.patch_size
        patch_size_t = self.patch_size_t

        latents = latents.reshape(
            batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size
        )
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents


class CoreMLWanVideoWrapper(torch.nn.Module):
    """Adapts Wan Core ML model to ComfyUI's video sampling interface"""
    def __init__(self, model_path, num_frames):
        super().__init__()
        self.coreml_model = ct.models.MLModel(model_path)
        self.num_frames = num_frames
        self.model_path = model_path

        # Video-specific configuration
        self.latent_format = comfy.latent_formats.SDXL()  # Placeholder
        self.adm_channels = 0

        # Wan config
        self.latent_channels = 16
        self.vae_scale_factor = 8

    def forward(self, x, timestep, **kwargs):
        """
        Wan Video forward pass.

        Args:
            x: Video latents (B, C, F, H, W) - 5D tensor
            timestep: Tensor (B,) or scalar
            **kwargs: Additional arguments including:
                - context: Text embeddings (B, seq_len, dim)

        Returns:
            Noise prediction tensor (B, C, F, H, W)
        """
        import numpy as np

        # Wan uses simple 5D latents, no packing needed
        hidden_states_np = x.cpu().numpy().astype(np.float32)

        # Get text embeddings from context
        context = kwargs.get("context", None)
        if context is None:
            B = x.shape[0]
            # Default empty conditioning - Wan uses larger dim
            context = torch.zeros(B, 512, 4096, device=x.device, dtype=x.dtype)
        encoder_hidden_states_np = context.cpu().numpy().astype(np.float32)

        # Prepare timestep
        if isinstance(timestep, torch.Tensor):
            t_val = timestep[0].item() if timestep.dim() > 0 else timestep.item()
        else:
            t_val = float(timestep)
        timestep_np = np.array([int(t_val)]).astype(np.int32)

        # Build inputs dict - Wan uses simple inputs
        inputs = {
            "hidden_states": hidden_states_np,
            "encoder_hidden_states": encoder_hidden_states_np,
            "timestep": timestep_np,
        }

        # Run Core ML model
        out = self.coreml_model.predict(inputs)

        # Get output
        if "sample" in out:
            output_np = out["sample"]
        else:
            output_np = list(out.values())[0]

        # Convert back to tensor
        noise_pred = torch.from_numpy(output_np).to(x.device, dtype=x.dtype)

        # Ensure output shape matches input shape
        if noise_pred.shape != x.shape:
            # Handle potential shape mismatches
            noise_pred = noise_pred.view(x.shape)

        return noise_pred
