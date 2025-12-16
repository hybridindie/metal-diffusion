import os
import coremltools as ct
import tempfile
import shutil
import gc
from rich.console import Console

console = Console()

def safe_quantize_model(ml_model, quantization_type, op_config=None, pbar=None):
    """
    Safely quantizes a Core ML model by offloading the intermediate FP16 model to disk
    to prevent OOM errors, then reloading and applying quantization.
    
    Args:
        ml_model: The source MLModel object (FP16/FP32).
        quantization_type (str): "int8", "int4", "mixed", etc.
        op_config: Optional specific OpLinearQuantizerConfig. If None, defaults are derived from type.
        pbar: Optional tqdm progress bar to update descriptions.
        
    Returns:
        The quantized MLModel object.
    """
    if quantization_type not in ["int4", "4bit", "mixed", "int8", "8bit"]:
        return ml_model

    # OOM Prevention Strategy:
    # 1. Save unquantized FP16 model to temp disk.
    # 2. Clear RAM.
    # 3. Load from disk.
    # 4. Quantize.
    
    # Use mkdtemp to control cleanup timing explicitly
    temp_dir = tempfile.mkdtemp(prefix="alloy_quant_")
    
    # Define a cleanup function
    def _cleanup():
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                console.print(f"[dim yellow]Warning: Could not clean up temp dir {temp_dir}: {e}[/dim yellow]")
    
    # Signal handling to ensure cleanup on cancellation (e.g. Ctrl+C)
    import signal
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    
    def _signal_handler(signum, frame):
        console.print(f"\n[bold red]Interrupted! Cleaning up temporary files...[/bold red]")
        _cleanup()
        # Restore original handler and forward signal if possible, or exit
        if callable(original_sigint) and signum == signal.SIGINT:
             original_sigint(signum, frame)
        elif callable(original_sigterm) and signum == signal.SIGTERM:
             original_sigterm(signum, frame)
        else:
             import sys
             sys.exit(1)
             
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        intermediate_path = os.path.join(temp_dir, "Intermediate_Model.mlpackage")
        console.print(f"[dim]Saving intermediate model to {intermediate_path} to free RAM...[/dim]")
        ml_model.save(intermediate_path)
        
        # Clear heavy objects from memory
        del ml_model
        gc.collect()
        
        console.print("[dim]Reloading for quantization...[/dim]")
        # Load with CPU only for quantization processing
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
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        # Execute cleanup
        _cleanup()
        
    return ml_model
