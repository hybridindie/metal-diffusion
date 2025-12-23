"""Conversion node for ComfyUI - intelligently converts models with caching"""

import os
import time
from pathlib import Path
import folder_paths
import hashlib

from alloy.monitor import estimate_memory_requirement, check_memory_warning

# Try to import ComfyUI's progress utilities
try:
    import comfy.utils
    HAS_COMFY_PROGRESS = hasattr(comfy.utils, 'ProgressBar')
except ImportError:
    HAS_COMFY_PROGRESS = False


class ComfyUIProgressDisplay:
    """
    Progress display adapter for ComfyUI that consumes Alloy progress events.
    Updates ComfyUI progress bar silently unless verbose mode is enabled.
    """

    # Phase weights for progress calculation (must sum to 1.0)
    PHASE_WEIGHTS = {
        "download": 0.10,
        "part1": 0.40,
        "part2": 0.35,
        "assembly": 0.15,
    }

    def __init__(self, model_name: str, quantization: str, verbose: bool = False):
        self.model_name = model_name
        self.quantization = quantization
        self.verbose = verbose
        self.start_time = time.time()
        self.current_phase = None
        self.completed_phases = set()
        self.pbar = None

    def start(self):
        """Start progress tracking."""
        self.start_time = time.time()
        print(f"[Alloy] Converting {self.model_name} ({self.quantization})...")

        # Initialize ComfyUI progress bar if available
        if HAS_COMFY_PROGRESS:
            try:
                self.pbar = comfy.utils.ProgressBar(100)
            except Exception:
                self.pbar = None

    def stop(self):
        """Stop progress tracking and show summary."""
        elapsed = time.time() - self.start_time
        print(f"[Alloy] Conversion complete ({self._format_duration(elapsed)})")

    def process_event(self, event):
        """Process a progress event from the conversion worker."""
        if event.event_type == "phase_start":
            self.current_phase = event.phase
            if self.verbose:
                phase_name = self._get_phase_display_name(event.phase)
                print(f"[Alloy]   {phase_name}...")
            self._update_progress_bar()

        elif event.event_type == "phase_end":
            if event.phase:
                self.completed_phases.add(event.phase)
            self._update_progress_bar()

        elif event.event_type in ("step_start", "step_end"):
            self._update_progress_bar()

    def _get_phase_display_name(self, phase: str) -> str:
        """Get human-readable phase name."""
        names = {
            "download": "Downloading",
            "part1": "Part 1",
            "part2": "Part 2",
            "assembly": "Assembling",
        }
        return names.get(phase, phase.title())

    def _update_progress_bar(self):
        """Update ComfyUI progress bar based on current state."""
        if not self.pbar:
            return

        completed_weight = sum(
            self.PHASE_WEIGHTS.get(p, 0) for p in self.completed_phases
        )

        if self.current_phase and self.current_phase not in self.completed_phases:
            phase_weight = self.PHASE_WEIGHTS.get(self.current_phase, 0)
            completed_weight += phase_weight * 0.5

        try:
            self.pbar.update_absolute(int(completed_weight * 100))
        except Exception:
            pass

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


class CoreMLConverter:
    """
    Convert models to Core ML directly in ComfyUI.
    Intelligently caches conversions - only converts if needed.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_source": ("STRING", {"default": "black-forest-labs/FLUX.1-schnell", "multiline": False}),
            "model_type": (["flux", "flux-controlnet", "ltx", "wan", "hunyuan", "lumina", "sd"],),
            "quantization": (["int4", "int8", "float16", "float32"],),
            "output_name": ("STRING", {"default": "", "multiline": False}),
            "force_reconvert": ("BOOLEAN", {"default": False}),
        },
        "optional": {
            "lora_stack": ("LORA_CONFIG",),
            "hf_token": ("STRING", {"default": "", "multiline": False}),
            "verbose": ("BOOLEAN", {"default": False}),
        }}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "convert_model"
    CATEGORY = "Alloy/Conversion"
    
    def convert_model(self, model_source, model_type, quantization, output_name, force_reconvert, lora_stack=None, hf_token=None, verbose=False):
        """
        Convert model to Core ML, using cache if available.

        Args:
            hf_token: Optional HuggingFace token for gated models (or set HF_TOKEN env var)
            verbose: Enable detailed output

        Returns: Path to the converted .mlpackage
        """
        if hf_token == "":
            hf_token = None

        # Determine output path
        if not output_name:
            source_name = model_source.replace("/", "_").replace(".", "_")
            if lora_stack:
                lora_str = "".join([f"{l['path']}{l['strength_model']}{l['strength_clip']}" for l in lora_stack])
                lora_hash = hashlib.md5(lora_str.encode()).hexdigest()[:8]
                output_name = f"{source_name}_lora{lora_hash}_{quantization}"
            else:
                output_name = f"{source_name}_{quantization}"

        # Build output path in ComfyUI's models folder
        if model_type == "flux-controlnet":
            model_folder = folder_paths.get_folder_paths("controlnet")[0]
        else:
            model_folder = folder_paths.get_folder_paths("unet")[0]

        output_base = os.path.join(model_folder, output_name)
        clean_source_name = model_source.split("/")[-1] if "/" in model_source else model_source
        final_path = os.path.join(output_base, f"{clean_source_name}_{quantization}.mlpackage")

        # Check if already converted
        if os.path.exists(final_path) and not force_reconvert:
            print(f"[Alloy] Using cached model: {final_path}")
            return (final_path,)

        # Check memory before starting
        base_type = model_type.replace("-controlnet", "")
        required_gb = estimate_memory_requirement(base_type, quantization)
        warning = check_memory_warning(required_gb)
        if warning:
            print(f"[Alloy] Warning: {warning}")

        try:
            # Import converters
            from alloy.converters.flux import FluxConverter
            from alloy.converters.controlnet import FluxControlNetConverter
            from alloy.converters.ltx import LTXConverter
            from alloy.converters.wan import WanConverter
            from alloy.converters.hunyuan import HunyuanConverter
            from alloy.converters.lumina import LuminaConverter
            from alloy.converters.base import SDConverter

            converter_map = {
                'flux': FluxConverter,
                'flux-controlnet': FluxControlNetConverter,
                'ltx': LTXConverter,
                'wan': WanConverter,
                'hunyuan': HunyuanConverter,
                'lumina': LuminaConverter,
                'sd': SDConverter
            }

            converter_class = converter_map[model_type]
            # Prepare LoRA args for converter
            kwargs = {'hf_token': hf_token}
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

            # Run conversion with ComfyUI progress display
            display = ComfyUIProgressDisplay(model_source, quantization, verbose=verbose)
            display.start()
            try:
                converter.convert(show_progress=False, progress_callback=display.process_event)
            finally:
                display.stop()

            if verbose:
                print(f"[Alloy] Saved to: {final_path}")

            return (final_path,)

        except Exception as e:
            raise RuntimeError(f"Conversion failed: {str(e)}")


class CoreMLQuickConverter:
    """
    One-click converter with smart defaults.
    Perfect for common use cases.

    For presets: just select and run - everything is configured automatically.
    For custom models: select "Custom" and provide a HuggingFace model ID.
    Model type is auto-detected, quantization defaults to int4.
    """

    PRESETS = [
        "Flux Schnell (Fast)",
        "Flux Dev (Quality)",
        "Flux ControlNet (Canny)",
        "Flux ControlNet (Depth)",
        "LTX Video",
        "Hunyuan Video",
        "Wan 2.1 Video",
        "Lumina Image 2.0",
        "Custom"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "preset": (CoreMLQuickConverter.PRESETS,),
            "quantization": (["int8", "int4", "float16"], {"default": "int8"}),
        },
        "optional": {
            "custom_model": ("STRING", {"default": "", "multiline": False,
                "tooltip": "HuggingFace model ID (only used when preset is 'Custom'). Model type is auto-detected."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "quick_convert"
    CATEGORY = "Alloy/Conversion"

    def quick_convert(self, preset, quantization="int8", custom_model=""):
        """Convert with presets for common models"""

        presets = {
            "Flux Schnell (Fast)": {
                "model": "black-forest-labs/FLUX.1-schnell",
                "type": "flux",
            },
            "Flux Dev (Quality)": {
                "model": "black-forest-labs/FLUX.1-dev",
                "type": "flux",
            },
            "Flux ControlNet (Canny)": {
                "model": "InstantX/FLUX.1-dev-Controlnet-Canny",
                "type": "flux-controlnet",
            },
            "Flux ControlNet (Depth)": {
                "model": "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
                "type": "flux-controlnet",
            },
            "LTX Video": {
                "model": "Lightricks/LTX-Video",
                "type": "ltx",
            },
            "Hunyuan Video": {
                "model": "hunyuanvideo-community/HunyuanVideo",
                "type": "hunyuan",
            },
            "Wan 2.1 Video": {
                "model": "Wan-AI/Wan2.1-T2V-14B",
                "type": "wan",
            },
            "Lumina Image 2.0": {
                "model": "Alpha-VLLM/Lumina-Image-2.0",
                "type": "lumina",
            },
        }

        if preset == "Custom":
            if not custom_model:
                raise ValueError("Custom preset requires a HuggingFace model ID in 'custom_model'")

            # Auto-detect model type
            from alloy.utils.general import detect_model_type
            detected_type = detect_model_type(custom_model)

            if not detected_type:
                raise ValueError(
                    f"Could not auto-detect model type for '{custom_model}'. "
                    "Use the full 'Core ML Converter' node for unsupported models."
                )

            model_id = custom_model
            model_type = detected_type
        else:
            config = presets[preset]
            model_id = config["model"]
            model_type = config["type"]

        # Use the main converter
        converter_node = CoreMLConverter()
        return converter_node.convert_model(
            model_id,
            model_type,
            quantization,
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
