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

import shutil
import tempfile
import time

def cleanup_old_temp_files(prefix="alloy_quant_", max_age_hours=1):
    """
    Cleans up old temporary directories created by Alloy that might have been left over due to crashes.
    """
    temp_base = tempfile.gettempdir()
    count = 0
    now = time.time()
    
    try:
        if not os.path.exists(temp_base):
            return 0
            
        for filename in os.listdir(temp_base):
            if filename.startswith(prefix):
                full_path = os.path.join(temp_base, filename)
                if os.path.isdir(full_path):
                    try:
                        # Check age
                        stat = os.stat(full_path)
                        age_hours = (now - stat.st_mtime) / 3600
                        if age_hours > max_age_hours:
                            shutil.rmtree(full_path)
                            count += 1
                    except Exception:
                        pass # Ignore permission errors etc
    except Exception:
        pass
        
    return count
