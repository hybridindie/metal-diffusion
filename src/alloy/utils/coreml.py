import os
import coremltools as ct
import tempfile
import shutil
import gc
from rich.console import Console

console = Console()

def safe_quantize_model(model_or_path, quantization_type, op_config=None, pbar=None, intermediate_dir=None):
    """
    Safely quantizes a Core ML model. Can accept either an MLModel object or a path to one.
    If a path is provided, it allows the caller strict control over memory (e.g. deleting the object before calling).
    
    Args:
        model_or_path: MLModel object OR string path to .mlpackage.
        quantization_type (str): "int8", "int4", "mixed", etc.
        op_config: Optional specific OpLinearQuantizerConfig.
        pbar: Optional tqdm progress bar.
        intermediate_dir (str, optional): If provided, saves intermediate FP16 model here instead of a temp dir.
                                          Useful for debugging or resuming. The caller owns cleanup of this directory.
        
    Returns:
        The quantized MLModel object.
    """
    if quantization_type not in ["int4", "4bit", "mixed", "int8", "8bit"]:
        # If it's a path and we aren't quantizing, we must load it to return an object
        if isinstance(model_or_path, str):
             return ct.models.MLModel(model_or_path)
        return model_or_path

    
    # Setup paths and cleanup needed?
    # If input is object: we handle temp file creation/cleanup.
    # If input is path: we use that path, and DO NOT clean it up (caller owns it).
    
    temp_dir_created = None
    intermediate_path = None
    
    try:
        if isinstance(model_or_path, str):
            intermediate_path = model_or_path
            console.print("[dim]Loading model from disk for quantization (Memory Optimized)...[/dim]")
            # Load with CPU only
            ml_model = ct.models.MLModel(intermediate_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        else:
            # It's an object. We must save it to disk to free RAM (if possible, but we can't free caller's ref)
            if intermediate_dir:
                # Use provided directory (Caller owns cleanup)
                import uuid
                intermediate_path = os.path.join(intermediate_dir, f"inter_quant_{uuid.uuid4()}.mlpackage")
                console.print(f"[dim]Saving intermediate model to {intermediate_path}...[/dim]")
            else:
                # Use temp directory (We own cleanup)
                temp_dir_created = tempfile.mkdtemp(prefix="alloy_quant_")
                intermediate_path = os.path.join(temp_dir_created, "Intermediate_Model.mlpackage")
                console.print(f"[dim]Saving intermediate model to {intermediate_path} to free RAM...[/dim]")
            
            model_or_path.save(intermediate_path)
            
            # Delete local ref
            del model_or_path
            gc.collect()
            
            console.print("[dim]Reloading for quantization...[/dim]")
            ml_model = ct.models.MLModel(intermediate_path, compute_units=ct.ComputeUnit.CPU_ONLY)

        if pbar:
            pbar.set_description(f"Quantizing ({quantization_type})")
        else:
            console.print(f"Applying {quantization_type.capitalize()} quantization...")
            
        nbits = 4 if quantization_type in ["int4", "4bit", "mixed"] else 8
        
        op_config_to_use = op_config
        if op_config_to_use is None:
            op_config_to_use = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8" if nbits == 8 else "int4", 
                weight_threshold=512
            )
            
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config_to_use)
        ml_model = ct.optimize.coreml.linear_quantize_weights(ml_model, config)
        
        # Force cleanup of any lingering internal states
        gc.collect()

    finally:
        # Cleanup ONLY if we created the temp dir (NOT if user provided intermediate_dir)
        if temp_dir_created and os.path.exists(temp_dir_created):
            try:
                shutil.rmtree(temp_dir_created)
            except Exception as e:
                console.print(f"[dim yellow]Warning: Could not clean up temp dir {temp_dir_created}: {e}[/dim yellow]")
        
    return ml_model
