import subprocess
from abc import ABC, abstractmethod

class ModelConverter(ABC):
    def __init__(self, model_id, output_dir, quantization="float16"):
        self.model_id = model_id
        self.output_dir = output_dir
        self.quantization = quantization

    @abstractmethod
    def convert(self):
        pass

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

class WanConverter(ModelConverter):
    def convert(self):
        """
        Custom conversion logic for Wan 2.1 models.
        """
        print(f"Converting Wan 2.1 model {self.model_id}...")
        raise NotImplementedError("Wan 2.1 conversion logic is not yet implemented.")
