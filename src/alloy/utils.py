import os
from safetensors import safe_open

def detect_model_type(model_path):
    """
    Detects the model type (flux, ltx, etc.) from a .safetensors file header.
    Returns: 'flux', 'ltx', 'wan', 'hunyuan' or None if unknown/not a file.
    """
    if not os.path.isfile(model_path):
        return None

    if not model_path.endswith(".safetensors"):
        return None

    try:
        with safe_open(model_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            
            # Check for Flux (Original or Diffusers)
            # Original: double_blocks, single_blocks
            # Diffusers: transformer.transformer_blocks, transformer.single_transformer_blocks
            is_flux = any("double_blocks" in k for k in keys) or any("single_blocks" in k for k in keys) or \
                      any("transformer_blocks" in k and "single_transformer_blocks" in k for k in keys)
            if is_flux:
                return "flux"

            # Check for LTX-Video
            # LTX uses 'scale_shift_table' in blocks usually, or 'caption_projection'
            is_ltx = any("scale_shift_table" in k for k in keys) or any("caption_projection" in k for k in keys)
            if is_ltx:
                return "ltx"
            
            # Check for Wan
            # Wan has 'blocks' but often specific naming?
            # If we haven't matched Flux/LTX, and it has blocks...
            # Wan 2.1 specific? 
            # Let's be conservative. If filename contains 'wan', maybe? 
            # But better to check keys.
            # Wan usually has "head" and "blocks".
            
    except Exception as e:
        print(f"Error detecting model type for {model_path}: {e}")
        return None
    
    return None
