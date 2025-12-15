"""LoRA configuration nodes for Core ML Conversion"""

import folder_paths

class CoreMLLoraConfig:
    """
    Define a LoRA configuration for baking into a Core ML model.
    Supports chaining to build a list of LoRAs.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
        },
        "optional": {
            "previous_lora": ("LORA_CONFIG",)
        }}
    
    RETURN_TYPES = ("LORA_CONFIG",)
    RETURN_NAMES = ("lora_config",)
    FUNCTION = "configure_lora"
    CATEGORY = "Alloy/Conversion"
    
    def configure_lora(self, lora_name, strength_model, strength_clip, previous_lora=None):
        # build config entry
        # path needs to be full path for the CLI
        lora_path = folder_paths.get_full_path("loras", lora_name)
        
        entry = {
            "path": lora_path,
            "strength_model": strength_model,
            "strength_clip": strength_clip
        }
        
        lora_list = []
        if previous_lora:
            if isinstance(previous_lora, list):
                lora_list.extend(previous_lora)
            else:
                lora_list.append(previous_lora)
        
        lora_list.append(entry)
        
        return (lora_list,)

NODE_CLASS_MAPPINGS = {
    "CoreMLLoraConfig": CoreMLLoraConfig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLLoraConfig": "Core ML LoRA Config"
}
