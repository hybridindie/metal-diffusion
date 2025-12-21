import argparse
import os
import sys
from alloy.converters.base import SDConverter, ModelConverter
from alloy.logging import (
    setup_logging,
    shutdown_logging,
    Verbosity,
    parse_log_level,
)
from alloy.converters.wan import WanConverter
from alloy.converters.hunyuan import HunyuanConverter
from alloy.converters.ltx import LTXConverter
from alloy.converters.flux import FluxConverter
from alloy.converters.controlnet import FluxControlNetConverter
from alloy.converters.lumina import LuminaConverter
from alloy.converters.vae import VAEConverter
from alloy.exceptions import (
    AlloyError,
    WorkerError,
    UnsupportedModelError,
    ConfigError,
    DependencyError,
    HuggingFaceError,
    ValidationError,
)
from alloy.validation import run_preflight_validation

from alloy.runners.flux import FluxCoreMLRunner
from alloy.runners.ltx import LTXCoreMLRunner
from alloy.runners.hunyuan import HunyuanCoreMLRunner
from alloy.runners.lumina import LuminaCoreMLRunner
from alloy.runners.wan import WanCoreMLRunner
from alloy.runners.base import run_sd_pipeline

from alloy.utils.model import validate_model, show_model_info, list_models, detect_safetensors_precision
from alloy.utils.hf import HFManager
from alloy.utils.general import detect_model_type, cleanup_old_temp_files
from dotenv import load_dotenv
import warnings
from rich.console import Console

# Suppress Torch "device_type='cuda'" warning on non-CUDA systems
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'", category=UserWarning)

load_dotenv()

DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "converted_models")


def ensure_cleanup():
    """Force cleanup of multiprocessing backends to avoid semaphore leaks."""
    try:
        # joblib (used by scikit-learn/coremltools) often leaks semaphores if not explicitly shutdown
        # We try to force a shutdown of the reusable executor.
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except ImportError:
        pass
    except Exception:
        # Ignore errors during cleanup
        pass

def main():
    # Setup console
    console = Console()

    def _print_suggestions(suggestions: list) -> None:
        """Display actionable suggestions to the user."""
        if suggestions:
            console.print("\n[yellow]Suggestions:[/yellow]")
            for suggestion in suggestions:
                console.print(f"  [dim]•[/dim] {suggestion}")

    # Opportunistic cleanup of old temp files from crashes
    cleaned = cleanup_old_temp_files(max_age_hours=0.01)
    if cleaned > 0:
        console.print(f"[dim]Cleaned up {cleaned} old temporary directory(s) from previous runs.[/dim]")
    
    parser = argparse.ArgumentParser(description="Alloy: CLI for Core ML Diffusion Models")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Enable verbose logging (-v for verbose, -vv for debug)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all output except errors")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download Command
    download_parser = subparsers.add_parser("download", help="Download a model from Hugging Face")
    download_parser.add_argument("repo_id", type=str, help="Hugging Face Repo ID")
    download_parser.add_argument("--output-dir", type=str, default=os.path.join("models"), help="Directory to save the model")
    
    # Convert Command
    convert_parser = subparsers.add_parser("convert", help="Convert a model to Core ML")
    convert_parser.add_argument("model_id", type=str, help="Hugging Face model ID or path")
    convert_parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    convert_parser.add_argument("--quantization", "-q", type=str, default=None, choices=["float16", "float32", "int8", "int4"], help="Quantization (defaults to float16, or detects from filename)")
    convert_parser.add_argument("--force-quantization", action="store_true", help="Force quantization even if it appears redundant (e.g. Int8 -> Int8)")
    convert_parser.add_argument("--type", type=str, choices=["sd", "wan", "hunyuan", "ltx", "flux", "flux-controlnet", "lumina", "vae"], help="Type of model (optional if auto-detectable)")
    convert_parser.add_argument("--vae-type", type=str, choices=["flux", "sdxl", "sd", "wan", "ltx", "hunyuan", "auto"], default="auto", help="VAE architecture type (for --type vae)")
    convert_parser.add_argument("--vae-components", type=str, nargs="+", choices=["encoder", "decoder"], default=["encoder", "decoder"], help="Which VAE components to convert")
    convert_parser.add_argument("--lora", action="append", help="LoRA to bake in. Format: path:strength or path:model_str:clip_str")
    convert_parser.add_argument("--controlnet", action="store_true", help="Enable ControlNet inputs (Flux only)")
    convert_parser.add_argument("--skip-validation", action="store_true", help="Skip pre-flight validation checks")
    
    # Upload Command
    upload_parser = subparsers.add_parser("upload", help="Upload converted model to Hugging Face")
    upload_parser.add_argument("local_path", type=str, help="Path to the converted model folder")
    upload_parser.add_argument("repo_id", type=str, help="Target Hugging Face Repo ID")
    
    # Full Pipeline Command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline: Download -> Convert -> Upload")
    pipeline_parser.add_argument("repo_id", type=str, help="Hugging Face Repo ID")
    pipeline_parser.add_argument("--target-repo", type=str, required=True, help="Target HF Repo ID")
    pipeline_parser.add_argument("--type", type=str, choices=["sd", "wan", "hunyuan", "ltx", "flux", "lumina"], required=True, help="Type of model")
    
    # Run Command
    run_parser = subparsers.add_parser("run", help="Run a converted model locally")
    run_parser.add_argument("model_dir", type=str, help="Path to converted model directory")

    # Validate Command
    validate_parser = subparsers.add_parser("validate", help="Validate a converted Core ML model")
    validate_parser.add_argument("model_path", type=str, help="Path to .mlpackage")

    # Info Command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model_path", type=str, help="Path to .mlpackage or model directory")

    # List Models Command
    list_parser = subparsers.add_parser("list-models", help="List all converted models")
    list_parser.add_argument("--dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to scan")
    
    # Batch Convert Command
    batch_parser = subparsers.add_parser("batch-convert", help="Convert multiple models from a batch file")
    batch_parser.add_argument("batch_file", type=str, help="Path to JSON/YAML file with model configs")
    batch_parser.add_argument("--dry-run", action="store_true", help="Show what would be converted without converting")
    batch_parser.add_argument("--parallel", action="store_true", help="Run conversions in parallel (experimental)")
    
    run_parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    run_parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    run_parser.add_argument("--type", type=str, choices=["sd", "wan", "hunyuan", "ltx", "flux", "lumina"], help="Type of model (optional if auto-detectable)")
    run_parser.add_argument("--height", type=int, default=512, help="Height")
    run_parser.add_argument("--width", type=int, default=512, help="Width")
    run_parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    run_parser.add_argument("--base-model", type=str, default="stabilityai/sd-turbo", help="Base HF model ID for tokenizer/scheduler")
    run_parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode (multiple iterations)")
    run_parser.add_argument("--benchmark-runs", type=int, default=3, help="Number of benchmark runs")
    run_parser.add_argument("--benchmark-output", type=str, help="Save benchmark JSON to file")

    args = parser.parse_args()

    # Determine verbosity level
    if args.quiet:
        verbosity = Verbosity.QUIET
    elif args.verbose >= 2:
        verbosity = Verbosity.DEBUG
    elif args.verbose == 1:
        verbosity = Verbosity.VERBOSE
    else:
        verbosity = Verbosity.NORMAL

    # Check for environment variable override
    env_level = os.getenv("ALLOY_LOG_LEVEL")
    if env_level:
        verbosity = parse_log_level(env_level)

    # Setup logging
    log_file = os.getenv("ALLOY_LOG_FILE")
    json_logging = os.getenv("ALLOY_JSON_LOGS", "").lower() in ("1", "true", "yes")
    setup_logging(
        verbosity=verbosity,
        log_file=log_file,
        json_logging=json_logging,
    )

    hf_manager = HFManager()

    try:
        if args.command == "download":
            hf_manager.download_model(args.repo_id, local_dir=os.path.join(args.output_dir, args.repo_id.split("/")[-1]))

        elif args.command == "convert":
            model_type = args.type
            if model_type is None:
                print("Auto-detecting model type...")
                model_type = detect_model_type(args.model_id)
                if model_type:
                    print(f"Detected model type: {model_type}")
                else:
                    print("Could not auto-detect model type.")
                    print("Supported types: flux, ltx, hunyuan, wan, lumina")
                    print("Please specify with --type <type>")
                    sys.exit(1)
            
            # 1. Try robust file inspection
            detected_precision = detect_safetensors_precision(args.model_id)
            
            user_specified_quantization = args.quantization is not None
            
            # Auto-detect quantization if not specified
            if args.quantization is None:
                if detected_precision:
                    print(f"[cyan]Auto-detected quantization: {detected_precision}[/cyan] (from metadata)")
                    # Don't set args.quantization yet, we might want to default to float16 to skip redundant quant
                    # But 'auto-detect' implies we want the model to BE that precision.
                    # However, since input IS that precision, to get output of that precision, we must quantize (since Core ML inflate).
                    # Wait, if I set args.quantization = int8, later logic will trigger.
                    args.quantization = detected_precision
                else:
                    # 2. Fallback to filename heuristic
                    model_lower = args.model_id.lower()
                    if "int8" in model_lower or "fp8" in model_lower: 
                         print("[cyan]Auto-detected quantization: int8[/cyan] (from filename)")
                         args.quantization = "int8"
                         detected_precision = "int8" # Treat as detected
                    elif "int4" in model_lower or "q4" in model_lower:
                         print("[cyan]Auto-detected quantization: int4[/cyan] (from filename)")
                         args.quantization = "int4"
                         detected_precision = "int4" # Treat as detected
                    else:
                         args.quantization = "float16" # Default

            # Guard against redundant quantization (Same Level)
            # If source is Int8 and target is Int8.
            # This re-calcuation degrades quality.
            if detected_precision:
                 target = args.quantization
                 # Aliases
                 if target == "4bit": target = "int4"
                 if target == "8bit": target = "int8"
                 
                 if target == detected_precision and target in ["int8", "int4"]:
                     if not args.force_quantization:
                         print(f"[bold yellow]Note:[/bold yellow] Re-quantizing to {target} to maintain source file size.")
                         print("(This allows Core ML conversion without expanding usage to Float16).")
                     else:
                         print(f"[yellow]Forcing re-quantization ({target} -> {target}) as requested.[/yellow]")

            # Warning for lesser quantization (Upscale) or Double Quant (Int8->Int4)
            if user_specified_quantization and detected_precision:
                # Rank: int4(0) < int8(1) < float16(2) < float32(3)
                
                ranks = {"int4": 0, "int8": 1, "float16": 2, "float32": 3}
                # Handle aliases
                target = args.quantization
                if target == "4bit": target = "int4"
                if target == "8bit": target = "int8"
                
                source_rank = ranks.get(detected_precision, 3)
                target_rank = ranks.get(target, 2)
                
                if target_rank > source_rank:
                     print(f"[bold yellow]Warning:[/bold yellow] Input ({detected_precision}) has lower precision than target ({target}).")
                     print("This will increase file size without restoring quality.")
                
                # Warning for double quantization (Int8 -> Int4)
                # If source is already quantized (rank < 2) and we are going lower.
                elif source_rank < 2 and target_rank < source_rank:
                     print(f"[bold yellow]Warning:[/bold yellow] Input model is already quantized ({detected_precision}).")
                     print(f"Quantizing further to {target} may cause significant quality degradation due to double-quantization artifacts.")
                     print("It is recommended to use an FP16 or FP32 source model for best results.")
                    
            if model_type == "flux":
                converter = FluxConverter(args.model_id, args.output_dir, args.quantization, loras=args.lora, controlnet_compatible=args.controlnet)
            elif model_type == "flux-controlnet":
                converter = FluxControlNetConverter(args.model_id, args.output_dir, args.quantization)
            elif model_type == "ltx":
                converter = LTXConverter(args.model_id, args.output_dir, args.quantization)
            elif model_type == "hunyuan":
                converter = HunyuanConverter(args.model_id, args.output_dir, args.quantization)
            elif model_type == "lumina":
                converter = LuminaConverter(args.model_id, args.output_dir, args.quantization)
            elif model_type == "wan":
                # Wan might need local files
                # local_path = hf_manager.download_model(args.repo_id, local_dir=download_dir)
                converter = WanConverter(args.model_id, args.output_dir, args.quantization)
            elif model_type == "vae":
                converter = VAEConverter(
                    args.model_id,
                    args.output_dir,
                    args.quantization or "float16",
                    vae_type=args.vae_type,
                    components=args.vae_components,
                )
            else:
                # Fallback to SD
                converter = SDConverter(args.model_id, args.output_dir, args.quantization)

            # Run pre-flight validation before conversion.
            # If validation fails, the exception propagates to the handler below
            # and the converter object is garbage collected.
            if not args.skip_validation:
                run_preflight_validation(converter)

            converter.convert()
            
        elif args.command == "run":
            # Imports are already done at top level

            
            model_type = args.type
            if model_type is None:
                print("Auto-detecting model type...")
                model_type = detect_model_type(args.model_dir)
                if model_type:
                    print(f"Detected type: {model_type}")
                else:
                     print("Could not auto-detect model type. Please specify --type.")
                     sys.exit(1)

            if model_type == "sd":
                run_sd_pipeline(args.model_dir, args.prompt, args.output, base_model=args.base_model)
            elif model_type == "flux":
                runner = FluxCoreMLRunner(args.model_dir, model_id=args.base_model or "black-forest-labs/FLUX.1-schnell")
                if args.benchmark:
                    # Benchmark mode - reuse runner instance across runs
                    from alloy.utils.benchmark import Benchmark
                    bench = Benchmark(f"Flux {args.height}x{args.width}, {args.steps} steps")

                    for i in range(args.benchmark_runs):
                        print(f"\n[Benchmark Run {i+1}/{args.benchmark_runs}]")
                        bench.start_run()
                        runner.generate(args.prompt, args.output, steps=args.steps, height=args.height, width=args.width)
                        bench.end_run()

                    # Print results
                    bench.print_results()

                    # Save if requested
                    if args.benchmark_output:
                        bench.save_json(args.benchmark_output)
                else:
                    # Normal mode
                    runner.generate(args.prompt, args.output, steps=args.steps, height=args.height, width=args.width)
            elif model_type == "wan":
                runner = WanCoreMLRunner(args.model_dir, model_id=args.base_model)
                runner.generate(args.prompt, args.output, height=args.height, width=args.width)
            elif model_type == "hunyuan":
                runner = HunyuanCoreMLRunner(args.model_dir, model_id=args.base_model)
                runner.generate(args.prompt, args.output, height=args.height, width=args.width)
            elif model_type == "lumina":
                runner = LuminaCoreMLRunner(args.model_dir, model_id=args.base_model or "Alpha-VLLM/Lumina-Image-2.0")
                runner.generate(args.prompt, args.output, height=args.height, width=args.width)
            else: # ltx
                runner = LTXCoreMLRunner(args.model_dir, model_id=args.base_model)
                runner.generate(args.prompt, args.output, height=args.height, width=args.width)

        elif args.command == "upload":
            if not hf_manager.login_check():
                sys.exit(1)
            hf_manager.upload_model(args.local_path, args.repo_id)

        elif args.command == "pipeline":
            # 1. Download
            print("--- Starting Pipeline ---")
            if args.target_repo and not hf_manager.login_check():
                 sys.exit(1)

            repo_name = args.repo_id.split("/")[-1]
            # Note: download_dir can be used if manual download is needed
            # download_dir = os.path.join("models", repo_name)
            # Check if we assume model_path is a local path or HF ID. 
            # For SD, python_coreml relies on HF ID usually, but can take local. 
            # We'll use the ID directly for SDConverter as it handles download internally mostly, 
            # but for Wan we might need manual download.
            
            # Actually, let's keep it simple: Pass HF ID to converter, let converter decide.
            # But if we want to save locally first:
            # hf_manager.download_model(args.repo_id, local_dir=download_dir)
            
            output_dir = os.path.join("converted_models", repo_name + "_coreml")
            
            if args.type == "sd":
                # SD Converter often prefers the HF ID directly
                converter = SDConverter(args.repo_id, output_dir, args.quantization)
            else:
                # Wan might need local files
                # local_path = hf_manager.download_model(args.repo_id, local_dir=download_dir)
                converter = WanConverter(args.repo_id, output_dir, args.quantization)
                
            converter.convert()
            
            if args.target_repo:
                hf_manager.upload_model(output_dir, args.target_repo)

        elif args.command == "validate":
            from alloy.utils.model import validate_model
            validate_model(args.model_path)

        elif args.command == "info":
            from alloy.utils.model import show_model_info
            show_model_info(args.model_path)

        elif args.command == "list-models":
            from alloy.utils.model import list_models
            list_models(args.dir)

        elif args.command == "batch-convert":
            from alloy.utils.batch import run_batch_conversion
            success = run_batch_conversion(args.batch_file, dry_run=args.dry_run, parallel=args.parallel)
            if not success:
                sys.exit(1)

        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except WorkerError as e:
        console.print(f"[red]Worker failed:[/red] {e}")
        _print_suggestions(e.suggestions)
        sys.exit(e.exit_code or 1)
    except HuggingFaceError as e:
        console.print(f"[red]Download failed:[/red] {e}")
        _print_suggestions(e.suggestions)
        sys.exit(1)
    except UnsupportedModelError as e:
        console.print(f"[red]Unsupported model:[/red] {e}")
        sys.exit(1)
    except ConfigError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        _print_suggestions(e.suggestions)
        sys.exit(1)
    except ValidationError as e:
        console.print(f"[red]Validation failed:[/red] {e}")
        _print_suggestions(e.suggestions)
        sys.exit(1)
    except DependencyError as e:
        console.print(f"[red]Missing dependency:[/red] {e}")
        _print_suggestions(e.suggestions)
        sys.exit(1)
    except AlloyError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print_exception()
        sys.exit(1)
    finally:
        shutdown_logging()
        ensure_cleanup()


if __name__ == "__main__":
    main()
