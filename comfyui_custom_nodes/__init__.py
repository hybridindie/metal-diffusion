# Setup alloy package path
import sys
import os

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# Monkey-patch ComfyUI's model_management to properly support MPS
def _patch_comfyui_mps_support():
    """Patch ComfyUI's get_free_memory to handle MPS devices properly."""
    try:
        import comfy.model_management as mm
        import psutil

        _original_get_free_memory = mm.get_free_memory

        def _patched_get_free_memory(dev=None, torch_free_too=False):
            if dev is None:
                dev = mm.get_torch_device()

            # Check for MPS device more robustly
            is_mps = False
            if hasattr(dev, 'type') and dev.type == 'mps':
                is_mps = True
            elif isinstance(dev, str) and 'mps' in dev.lower():
                is_mps = True
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # If dev is the default device and MPS is available, check cpu_state
                if hasattr(mm, 'cpu_state') and hasattr(mm, 'CPUState'):
                    if mm.cpu_state == mm.CPUState.MPS:
                        is_mps = True

            if is_mps or (hasattr(dev, 'type') and dev.type == 'cpu'):
                mem_free_total = psutil.virtual_memory().available
                mem_free_torch = mem_free_total
                if torch_free_too:
                    return (mem_free_total, mem_free_torch)
                return mem_free_total

            # Fall back to original for CUDA and other devices
            return _original_get_free_memory(dev, torch_free_too)

        mm.get_free_memory = _patched_get_free_memory
        print("[Alloy] Patched ComfyUI MPS memory management")
    except Exception as e:
        print(f"[Alloy] Warning: Could not patch MPS support: {e}")

_patch_comfyui_mps_support()

# Try to import alloy - if it fails, add the src path
try:
    import alloy
except ImportError:
    # Check if we're symlinked or in the alloy repo
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_this_dir)
    _src_path = os.path.join(_parent_dir, "src")

    if os.path.exists(_src_path):
        sys.path.insert(0, _src_path)
    else:
        # We might be copied - try to find alloy in common locations
        raise ImportError(
            "Could not find 'alloy' package. Please either:\n"
            "  1. Install alloy: pip install silicon-alloy\n"
            "  2. Use a symlink: ln -s /path/to/alloy/comfyui_custom_nodes /path/to/ComfyUI/custom_nodes/Alloy"
        )

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .integrated_nodes import NODE_CLASS_MAPPINGS as INTEGRATED_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as INTEGRATED_DISPLAY_MAPPINGS
from .advanced_nodes import NODE_CLASS_MAPPINGS as ADVANCED_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ADVANCED_DISPLAY_MAPPINGS
from .converter_nodes import NODE_CLASS_MAPPINGS as CONVERTER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as CONVERTER_DISPLAY_MAPPINGS
from .lora_nodes import NODE_CLASS_MAPPINGS as LORA_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LORA_DISPLAY_MAPPINGS
from .vae_nodes import NODE_CLASS_MAPPINGS as VAE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as VAE_DISPLAY_MAPPINGS

# Merge all mappings
NODE_CLASS_MAPPINGS.update(INTEGRATED_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ADVANCED_MAPPINGS)
NODE_CLASS_MAPPINGS.update(CONVERTER_MAPPINGS)
NODE_CLASS_MAPPINGS.update(LORA_MAPPINGS)
NODE_CLASS_MAPPINGS.update(VAE_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(INTEGRATED_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ADVANCED_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CONVERTER_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LORA_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(VAE_DISPLAY_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
