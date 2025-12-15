"""Advanced ComfyUI nodes for Metal Diffusion"""

import folder_paths
import comfy.utils
import comfy.model_management
import coremltools as ct
from pathlib import Path


class CoreMLModelAnalyzer:
    """
    Analyze and display Core ML model information in ComfyUI.
    Useful for debugging and understanding model properties.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_path": (folder_paths.get_filename_list("unet"),)
        }}
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_model"
    CATEGORY = "Alloy/Utilities"
    OUTPUT_NODE = True
    
    def analyze_model(self, model_path):
        """Analyze Core ML model and return detailed info"""
        full_path = folder_paths.get_full_path("unet", model_path)
        
        try:
            model = ct.models.MLModel(full_path)
            spec = model.get_spec()
            
            # Build analysis report
            report = []
            report.append(f"=== Core ML Model Analysis ===")
            report.append(f"Path: {full_path}")
            report.append(f"Model Type: {spec.WhichOneof('Type')}")
            
            # File size
            if Path(full_path).exists():
                size_mb = sum(f.stat().st_size for f in Path(full_path).rglob('*') if f.is_file()) / (1024 * 1024)
                report.append(f"Size: {size_mb:.2f} MB")
            
            # Inputs
            report.append(f"\nInputs ({len(spec.description.input)}):")
            for inp in spec.description.input:
                input_type = inp.type.WhichOneof('Type')
                report.append(f"  - {inp.name}: {input_type}")
            
            # Outputs
            report.append(f"\nOutputs ({len(spec.description.output)}):")
            for out in spec.description.output:
                output_type = out.type.WhichOneof('Type')
                report.append(f"  - {out.name}: {output_type}")
            
            # Metadata
            if spec.description.metadata:
                report.append(f"\nMetadata:")
                if spec.description.metadata.shortDescription:
                    report.append(f"  Description: {spec.description.metadata.shortDescription}")
                if spec.description.metadata.author:
                    report.append(f"  Author: {spec.description.metadata.author}")
            
            result = "\n".join(report)
            print(result)  # Also print to console
            return (result,)
            
        except Exception as e:
            error_msg = f"Error analyzing model: {str(e)}"
            print(error_msg)
            return (error_msg,)


class CoreMLBatchSampler:
    """
    Generate multiple images in parallel using Core ML.
    Experimental - may require significant memory.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
            "sampler_name": (["euler", "euler_a", "dpmpp_2m", "ddim"],),
            "scheduler": (["normal", "karras", "exponential"],),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            "batch_size": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1})
        }}
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch_sample"
    CATEGORY = "Alloy/Advanced"
    
    def batch_sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise, batch_size):
        """
        Generate multiple samples in batch.
        Currently processes sequentially but prepares batched latents.
        """
        import comfy.sample
        
        # Prepare batched latents
        latent = latent_image.copy()
        
        # Expand latent to batch size
        if "samples" in latent:
            samples = latent["samples"]
            # Repeat samples for batch
            batched_samples = samples.repeat(batch_size, 1, 1, 1)
            latent["samples"] = batched_samples
        
        # For now, use standard sampling
        # TODO: Implement true parallel sampling
        samples = comfy.sample.sample(
            model,
            comfy.sample.prepare_noise(latent["samples"], seed),
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent["samples"],
            denoise=denoise
        )
        
        return ({"samples": samples},)


NODE_CLASS_MAPPINGS = {
    "CoreMLModelAnalyzer": CoreMLModelAnalyzer,
    "CoreMLBatchSampler": CoreMLBatchSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLModelAnalyzer": "Core ML Model Analyzer",
    "CoreMLBatchSampler": "Core ML Batch Sampler"
}
