import os
from safetensors import safe_open

def detect_model_type(model_path):
    """
    Detects the model type (flux, ltx, etc.) from a .safetensors file header.
    Returns: 'flux', 'ltx', 'wan', 'hunyuan' or None if unknown/not a file.
    """
    
    # 1. Local File Detection
    if os.path.isfile(model_path):
        if not model_path.endswith(".safetensors"):
            return None

        try:
            with safe_open(model_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                
                # Check for Flux
                is_flux = any("double_blocks" in k for k in keys) or any("single_blocks" in k for k in keys) or \
                          any("transformer_blocks" in k and "single_transformer_blocks" in k for k in keys)
                if is_flux:
                    return "flux"

                # Check for LTX-Video
                is_ltx = any("scale_shift_table" in k for k in keys) or any("caption_projection" in k for k in keys)
                if is_ltx:
                    return "ltx"
                
                # Wan check (simple heuristic from keys if possible, or skip)
                
        except Exception as e:
            print(f"Error detecting model type for {model_path}: {e}")
            return None
            
    # 2. Hugging Face Repo Detection
    elif "/" in model_path:
        try:
            from huggingface_hub import hf_hub_download
            import json
            
            # Try to fetch config.json or transformer/config.json
            config_path = None
            try:
                # Try transformer config first as it's more specific for diffusers pipelines
                config_path = hf_hub_download(repo_id=model_path, filename="transformer/config.json")
            except:
                try:
                    config_path = hf_hub_download(repo_id=model_path, filename="config.json")
                except:
                    pass
            
            if config_path and os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    
                # Check specifics
                arch = config.get("_class_name", "") or str(config.get("architectures", []))
                
                if "Flux" in arch:
                    return "flux"
                if "LTX" in arch:
                    return "ltx"
                if "Hunyuan" in arch:
                    return "hunyuan"
                if "Wan" in arch:
                    return "wan"
                if "Lumina" in arch:
                    return "lumina"
                    
        except Exception:
            pass

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
