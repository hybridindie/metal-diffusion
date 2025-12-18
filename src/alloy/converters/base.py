import subprocess
import os
import shutil
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torch.nn as nn
import coremltools as ct


class BaseModelWrapper(nn.Module):
    """
    Base wrapper class for PyTorch models to ensure consistent CoreML tracing.

    All model wrappers should inherit from this class and implement the forward method
    to map inputs to the underlying model's expected signature with return_dict=False.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        """
        Subclasses must override this to map arguments to model inputs.
        Always call model with return_dict=False to ensure tuple output for tracing.
        """
        raise NotImplementedError("Subclasses must implement forward()")


class ModelConverter(ABC):
    def __init__(self, model_id, output_dir, quantization="float16"):
        self.model_id = model_id
        self.output_dir = output_dir
        self.quantization = quantization

    @abstractmethod
    def convert(self):
        pass

    def validate_or_reconvert(
        self,
        intermediate_path: str,
        convert_fn: Callable[[], None],
        logger_fn: Callable[[str], None] = print
    ) -> bool:
        """
        Check if an intermediate model exists and is valid; reconvert if not.

        Args:
            intermediate_path: Path to the intermediate .mlpackage file
            convert_fn: Function to call if reconversion is needed
            logger_fn: Function for logging messages (default: print)

        Returns:
            True if conversion was performed, False if valid intermediate was found
        """
        if not os.path.exists(intermediate_path):
            convert_fn()
            return True

        try:
            logger_fn(f"Checking existing intermediate at {intermediate_path}...")
            ct.models.MLModel(intermediate_path, compute_units=ct.ComputeUnit.CPU_ONLY)
            logger_fn("Found valid intermediate. Resuming...")
            return False  # No reconversion needed
        except Exception:
            logger_fn("Invalid intermediate found. Re-converting...")
            shutil.rmtree(intermediate_path)
            convert_fn()
            return True

    def download_source_weights(
        self,
        repo_id: str,
        output_dir: str,
        allow_patterns: Optional[list] = None,
        ignore_patterns: Optional[list] = None,
        logger_fn: Callable[[str], None] = print
    ) -> str:
        """
        Download model weights from HuggingFace Hub to a local directory.

        Args:
            repo_id: HuggingFace repository ID (e.g., "black-forest-labs/FLUX.1-schnell")
            output_dir: Base output directory for the conversion
            allow_patterns: List of file patterns to include (default: transformer/*, config files)
            ignore_patterns: List of file patterns to exclude (default: msgpack, bin)
            logger_fn: Function for logging messages

        Returns:
            Path to the downloaded source directory, or original repo_id if download fails
        """
        if "/" not in repo_id or os.path.isfile(repo_id):
            return repo_id

        logger_fn("Downloading original model weights to output folder...")

        if allow_patterns is None:
            allow_patterns = ["transformer/*", "config.json", "*.json", "*.safetensors"]
        if ignore_patterns is None:
            ignore_patterns = ["*.msgpack", "*.bin"]

        try:
            from huggingface_hub import snapshot_download
            source_dir = os.path.join(output_dir, "source")
            snapshot_download(
                repo_id=repo_id,
                local_dir=source_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns
            )
            logger_fn(f"Originals saved to: {source_dir}")
            return source_dir
        except Exception as e:
            logger_fn(f"Warning: Failed to download source originals ({e}). Proceeding with remote load...")
            return repo_id

class SDConverter(ModelConverter):
    def convert(self):
        """
        Converts Stable Diffusion models using python_coreml_stable_diffusion.
        """
        import importlib.util
        if importlib.util.find_spec("python_coreml_stable_diffusion") is None:
            print("Error: 'python_coreml_stable_diffusion' is required for SD conversion but not installed.")
            print("Please install it manually:")
            print("  pip install git+https://github.com/apple/ml-stable-diffusion.git@main")
            return

        print(f"Converting {self.model_id} to Core ML (Quantization: {self.quantization})...")
        
        # Base command
        cmd = [
            "python", "-m", "python_coreml_stable_diffusion.torch2coreml",
            "--convert-unet", "--convert-text-encoder", "--convert-vae-decoder", "--convert-safety-checker",
            "--model-version", self.model_id,
            "-o", self.output_dir,
            "--bundle-resources-for-swift-cli"
        ]

        # Handle SDXL specific flags
        if "xl" in self.model_id.lower():
            print("Detected SDXL model. Enabling SDXL specific flags.")
            cmd.append("--xl-version") # This flag might be needed depending on the library version
            # SDXL typically requires attention slicing or split einsum for memory
            cmd.extend(["--attention-implementation", "SPLIT_EINSUM"])

        # Quantization handling
        if self.quantization == "float16":
             cmd.extend(["--compute-unit", "ALL"]) # coremltools default is float32 usually, need to ensure we output float16 if desired, but torch2coreml might default to float32
             # The Apple script usually has --quantize-nbits. If we want pure float16, we might not set nbits, 
             # but usually for SD on Mac, float16 is standard.
             pass 
        elif self.quantization in ["int8", "8bit"]:
             cmd.extend(["--quantize-nbits", "8"])
        elif self.quantization in ["int4", "4bit", "mixed"]:
             # Mixed bit quantization (palettization)
             cmd.extend(["--quantize-nbits", "4"])

        print(f"Running command: {' '.join(cmd)}")
        try:
            # We run this as a subprocess to isolate the conversion environment
            subprocess.run(cmd, check=True)
            print(f"Conversion of {self.model_id} successful.")
        except subprocess.CalledProcessError as e:
            print(f"Conversion failed: {e}")
            raise

