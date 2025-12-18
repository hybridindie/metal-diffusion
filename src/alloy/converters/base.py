import subprocess
import os
import shutil
import multiprocessing
import gc
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch.nn as nn
import coremltools as ct
from rich.console import Console

from alloy.exceptions import WorkerError
from alloy.utils.errors import get_worker_suggestions

console = Console()


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


class TwoPhaseConverter(ModelConverter):
    """
    Base class for converters that use 2-phase subprocess isolation.

    This pattern splits large transformer models at the midpoint to prevent OOM
    during CoreML conversion. Each phase runs in a separate subprocess.

    Subclasses must implement:
        - model_name: Property returning the model name (e.g., 'Flux', 'Hunyuan')
        - get_part1_worker(): Returns the worker function for Part 1
        - get_part2_worker(): Returns the worker function for Part 2

    Optional overrides:
        - should_download_source: Set to False to skip automatic weight download
        - part1_description: Custom description for Part 1 (default: "Part 1")
        - part2_description: Custom description for Part 2 (default: "Part 2")
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name for filenames and messages (e.g., 'Flux', 'Hunyuan')."""
        pass

    @abstractmethod
    def get_part1_worker(self) -> Callable:
        """Return the worker function for Part 1 conversion."""
        pass

    @abstractmethod
    def get_part2_worker(self) -> Callable:
        """Return the worker function for Part 2 conversion."""
        pass

    @property
    def should_download_source(self) -> bool:
        """Override to disable automatic source download. Default: True."""
        return True

    @property
    def part1_description(self) -> str:
        """Override for custom Part 1 description. Default: 'Part 1'."""
        return "Part 1"

    @property
    def part2_description(self) -> str:
        """Override for custom Part 2 description. Default: 'Part 2'."""
        return "Part 2"

    @property
    def output_filename(self) -> str:
        """Filename for the final merged model."""
        return f"{self.model_name}_Transformer_{self.quantization}.mlpackage"

    @property
    def part1_filename(self) -> str:
        """Filename for Part 1 intermediate."""
        return f"{self.model_name}Part1.mlpackage"

    @property
    def part2_filename(self) -> str:
        """Filename for Part 2 intermediate."""
        return f"{self.model_name}Part2.mlpackage"

    def convert(self):
        """Main conversion entry point using 2-phase subprocess pattern."""
        os.makedirs(self.output_dir, exist_ok=True)
        ml_model_path = os.path.join(self.output_dir, self.output_filename)

        if os.path.exists(ml_model_path):
            console.print(f"[yellow]Model exists, skipping:[/yellow] {ml_model_path}")
            return

        intermediates_dir = os.path.join(self.output_dir, "intermediates")
        os.makedirs(intermediates_dir, exist_ok=True)

        if self.should_download_source:
            self.model_id = self.download_source_weights(self.model_id, self.output_dir)

        try:
            part1_path = os.path.join(intermediates_dir, self.part1_filename)
            part2_path = os.path.join(intermediates_dir, self.part2_filename)

            self._convert_part(1, part1_path, self.get_part1_worker(), intermediates_dir)
            self._convert_part(2, part2_path, self.get_part2_worker(), intermediates_dir)
            self._assemble_pipeline(part1_path, part2_path, ml_model_path, intermediates_dir)

        except Exception:
            console.print(f"[yellow]Note: Intermediate files left in {intermediates_dir} for inspection/cleanup.[/yellow]")
            raise

    def _convert_part(
        self,
        part_num: int,
        output_path: str,
        worker_fn: Callable,
        intermediates_dir: str
    ) -> None:
        """Convert a single part with validation and subprocess isolation."""
        description = self.part1_description if part_num == 1 else self.part2_description

        # Check for existing valid intermediate
        if os.path.exists(output_path):
            try:
                console.print(f"[dim]Checking existing {description} at {output_path}...[/dim]")
                ct.models.MLModel(output_path, compute_units=ct.ComputeUnit.CPU_ONLY)
                console.print(f"[green]Found valid {description} intermediate. Resuming...[/green]")
                return  # Skip conversion
            except Exception:
                console.print(f"[yellow]Found invalid/incomplete {description}. Re-converting...[/yellow]")
                shutil.rmtree(output_path)

        # Spawn subprocess for conversion
        console.print(f"\n[bold]Spawning {description} Conversion Process...[/bold]")
        process = multiprocessing.Process(
            target=worker_fn,
            args=(self.model_id, output_path, self.quantization),
            kwargs={"intermediates_dir": intermediates_dir}
        )
        process.start()
        process.join()

        if process.exitcode != 0:
            raise WorkerError(
                "Worker process failed",
                model_name=self.model_name,
                phase=description,
                exit_code=process.exitcode,
                suggestions=get_worker_suggestions(process.exitcode, description),
            )

    def _assemble_pipeline(
        self,
        part1_path: str,
        part2_path: str,
        output_path: str,
        intermediates_dir: str
    ) -> None:
        """Load parts, assemble pipeline, cleanup, and save."""
        console.print("\n[bold]Assembling Pipeline...[/bold]")

        # Load parts with CPU_ONLY to minimize memory
        m1 = ct.models.MLModel(part1_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        m2 = ct.models.MLModel(part2_path, compute_units=ct.ComputeUnit.CPU_ONLY)

        pipeline_model = ct.utils.make_pipeline(m1, m2)

        # Add metadata
        pipeline_model.author = "Alloy"
        pipeline_model.license = "Apache 2.0"
        pipeline_model.short_description = f"{self.model_name} Transformer (Split Pipeline) {self.quantization}"

        # Cleanup before final save
        console.print("[dim]Releasing intermediate memory/disk for final save...[/dim]")
        del m1, m2
        gc.collect()

        try:
            shutil.rmtree(intermediates_dir)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not clear intermediates: {e}[/yellow]")

        console.print(f"[dim]Saving final pipeline to {output_path}...[/dim]")
        pipeline_model.save(output_path)

        console.print(f"[bold green]✓ {self.model_name} conversion complete![/bold green] Saved to {self.output_dir}")


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

