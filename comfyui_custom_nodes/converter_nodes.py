"""Conversion node for ComfyUI - intelligently converts models with caching"""

import os
import torch
from pathlib import Path
import folder_paths
import hashlib


class CoreMLConverter:
    """
    Convert models to Core ML directly in ComfyUI.
    Intelligently caches conversions - only converts if needed.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_source": ("STRING", {"default": "black-forest-labs/FLUX.1-schnell", "multiline": False}),
            "model_type": (["flux", "ltx", "wan", "hunyuan", "sd"],),
            "quantization": (["float16", "int4", "int8"],),
            "output_name": ("STRING", {"default": "", "multiline": False}),
            "output_name": ("STRING", {"default": "", "multiline": False}),
            "force_reconvert": ("BOOLEAN", {"default": False})
        },
        "optional": {
            "lora_stack": ("LORA_CONFIG",)
        }}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "convert_model"
    CATEGORY = "Alloy/Conversion"
    OUTPUT_NODE = True
    
    def convert_model(self, model_source, model_type, quantization, output_name, force_reconvert, lora_stack=None):
        """
        Convert model to Core ML, using cache if available.
        
        Returns: Path to the converted .mlpackage
        """
        # Determine output path
        if not output_name:
            # Auto-generate name from source
            source_name = model_source.replace("/", "_").replace(".", "_")
            
            # Append LoRA hash to ensure unique caching for different LoRA combos
            if lora_stack:
                import hashlib
                lora_str = "".join([f"{l['path']}{l['strength_model']}{l['strength_clip']}" for l in lora_stack])
                lora_hash = hashlib.md5(lora_str.encode()).hexdigest()[:8]
                output_name = f"{source_name}_lora{lora_hash}_{quantization}"
            else:
                output_name = f"{source_name}_{quantization}"
        
        # Build output path
        output_base = os.path.join("converted_models", output_name)
        transformer_name = f"{model_type.capitalize()}_Transformer.mlpackage"
        final_path = os.path.join(output_base, transformer_name)
        
        # Check if already converted
        if os.path.exists(final_path) and not force_reconvert:
            print(f"✓ Model already converted: {final_path}")
            print(f"  Set 'force_reconvert' to True to reconvert")
            return (final_path,)
        
        # Need to convert
        print(f"Converting {model_source} to Core ML...")
        print(f"  Type: {model_type}")
        print(f"  Quantization: {quantization}")
        print(f"  Output: {output_base}")
        
        try:
            # Import converters
            from alloy.flux_converter import FluxConverter
            from alloy.ltx_converter import LTXConverter
            from alloy.wan_converter import WanConverter
            from alloy.hunyuan_converter import HunyuanConverter
            from alloy.converter import SDConverter
            
            converter_map = {
                'flux': FluxConverter,
                'ltx': LTXConverter,
                'wan': WanConverter,
                'hunyuan': HunyuanConverter,
                'sd': SDConverter
            }
            
            converter_class = converter_map[model_type]
            # Prepare LoRA args for converter
            kwargs = {}
            if model_type == 'flux':
                 # Convert ComfyUI LoRA stack to CLI format strings "path:str_model:str_clip"
                 if lora_stack:
                     lora_args = []
                     for l in lora_stack:
                         arg = f"{l['path']}:{l['strength_model']}:{l['strength_clip']}"
                         lora_args.append(arg)
                     kwargs['loras'] = lora_args
            
            converter = converter_class(
                model_source,
                output_base,
                quantization,
                **kwargs
            )
            
            # Run conversion
            print("Starting conversion (this may take 5-15 minutes)...")
            converter.convert()
            
            print(f"✓ Conversion complete!")
            print(f"  Saved to: {final_path}")
            
            return (final_path,)
            
        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg)


class CoreMLQuickConverter:
    """
    One-click converter with smart defaults.
    Perfect for common use cases.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "preset": (["Flux Schnell (Fast)", "Flux Dev (Quality)", "LTX Video", "Custom"],),
        },
        "optional": {
            "custom_model": ("STRING", {"default": "", "multiline": False}),
            "custom_type": (["flux", "ltx", "wan"],),
        }}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "quick_convert"
    CATEGORY = "Alloy/Conversion"
    OUTPUT_NODE = True
    
    def quick_convert(self, preset, custom_model="", custom_type="flux"):
        """Convert with presets for common models"""
        
        presets = {
            "Flux Schnell (Fast)": {
                "model": "black-forest-labs/FLUX.1-schnell",
                "type": "flux",
                "quant": "int4"
            },
            "Flux Dev (Quality)": {
                "model": "black-forest-labs/FLUX.1-dev",
                "type": "flux",
                "quant": "int4"
            },
            "LTX Video": {
                "model": "Lightricks/LTX-Video",
                "type": "ltx",
                "quant": "int4"
            },
            "Custom": {
                "model": custom_model,
                "type": custom_type,
                "quant": "int4"
            }
        }
        
        config = presets[preset]
        
        # Use the main converter
        converter_node = CoreMLConverter()
        return converter_node.convert_model(
            config["model"],
            config["type"],
            config["quant"],
            "",  # Auto name
            False  # Don't force reconvert
        )


NODE_CLASS_MAPPINGS = {
    "CoreMLConverter": CoreMLConverter,
    "CoreMLQuickConverter": CoreMLQuickConverter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLConverter": "Core ML Converter",
    "CoreMLQuickConverter": "Core ML Quick Converter"
}
