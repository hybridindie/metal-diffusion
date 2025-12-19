# Setup alloy package path
import sys
import os

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

# Merge all mappings
NODE_CLASS_MAPPINGS.update(INTEGRATED_MAPPINGS)
NODE_CLASS_MAPPINGS.update(ADVANCED_MAPPINGS)
NODE_CLASS_MAPPINGS.update(CONVERTER_MAPPINGS)
NODE_CLASS_MAPPINGS.update(LORA_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(INTEGRATED_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(ADVANCED_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CONVERTER_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LORA_DISPLAY_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
