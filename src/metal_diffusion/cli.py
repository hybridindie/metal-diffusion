import argparse
import os
import sys
import logging
from .converter import SDConverter
from .wan_converter import WanConverter
from .hunyuan_converter import HunyuanConverter
from .ltx_converter import LTXConverter
from .flux_converter import FluxConverter
from metal_diffusion.hf_utils import HFManager
from metal_diffusion.utils import detect_model_type
from dotenv import load_dotenv
import warnings
from rich.console import Console
from rich.logging import RichHandler

# Suppress Torch "device_type='cuda'" warning on non-CUDA systems
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'", category=UserWarning)

load_dotenv()

DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "converted_models")

def main():
    parser = argparse.ArgumentParser(description="Diffusion to Core ML Converter")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress all output except errors")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download Command
    download_parser = subparsers.add_parser("download", help="Download a model from Hugging Face")
    download_parser.add_argument("repo_id", type=str, help="Hugging Face Repo ID")
    download_parser.add_argument("--output-dir", type=str, default=os.path.join("models"), help="Directory to save the model")
    
    # Convert Command
    convert_parser = subparsers.add_parser("convert", help="Convert a model to Core ML")
    convert_parser.add_argument("model_path", type=str, help="Path to local model or HF Repo ID")
    convert_parser.add_argument("--type", type=str, choices=["sd", "wan", "hunyuan", "ltx", "flux"], help="Type of model (optional if auto-detectable)")
    convert_parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    convert_parser.add_argument("--quantization", type=str, default="float16", choices=["float16", "int8", "int4"], help="Quantization level")
    
    # Upload Command
    upload_parser = subparsers.add_parser("upload", help="Upload converted model to Hugging Face")
    upload_parser.add_argument("local_path", type=str, help="Path to the converted model folder")
    upload_parser.add_argument("repo_id", type=str, help="Target Hugging Face Repo ID")
    
    # Full Pipeline Command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline: Download -> Convert -> Upload")
    pipeline_parser.add_argument("repo_id", type=str, help="Hugging Face Repo ID")
    pipeline_parser.add_argument("--target-repo", type=str, required=True, help="Target HF Repo ID")
    pipeline_parser.add_argument("--type", type=str, choices=["sd", "wan", "hunyuan", "ltx", "flux"], required=True, help="Type of model")
    
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
    run_parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    run_parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    run_parser.add_argument("--type", type=str, choices=["sd", "wan", "hunyuan", "ltx", "flux"], help="Type of model (optional if auto-detectable)")
    run_parser.add_argument("--height", type=int, default=512, help="Height")
    run_parser.add_argument("--width", type=int, default=512, help="Width")
    run_parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    run_parser.add_argument("--base-model", type=str, default="stabilityai/sd-turbo", help="Base HF model ID for tokenizer/scheduler")

    args = parser.parse_args()
    
    hf_manager = HFManager()

    if args.command == "download":
        hf_manager.download_model(args.repo_id, local_dir=os.path.join(args.output_dir, args.repo_id.split("/")[-1]))

    elif args.command == "convert":
        model_type = args.type
        if model_type is None:
            print("Auto-detecting model type...")
            model_type = detect_model_type(args.model_path)
            if model_type:
                print(f"Detected type: {model_type}")
            else:
                 print("Could not auto-detect model type. Please specify --type.")
                 sys.exit(1)

        if model_type == "sd":
            converter = SDConverter(args.model_path, args.output_dir, args.quantization)
        elif model_type == "wan":
            converter = WanConverter(args.model_path, args.output_dir, args.quantization)
        elif model_type == "hunyuan":
            converter = HunyuanConverter(args.model_path, args.output_dir, args.quantization)
        elif model_type == "flux":
            converter = FluxConverter(args.model_path, args.output_dir, args.quantization)
        else: # ltx
            converter = LTXConverter(args.model_path, args.output_dir, args.quantization)
        converter.convert()
        
    elif args.command == "run":
        from .runner import run_sd_pipeline, WanCoreMLRunner, HunyuanCoreMLRunner, LTXCoreMLRunner, FluxCoreMLRunner
        
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
        elif model_type == "wan":
            runner = WanCoreMLRunner(args.model_dir, model_id=args.base_model)
            runner.generate(args.prompt, args.output, height=args.height, width=args.width)
        elif model_type == "hunyuan":
            runner = HunyuanCoreMLRunner(args.model_dir, model_id=args.base_model)
            runner.generate(args.prompt, args.output, height=args.height, width=args.width)
        elif model_type == "flux":
            runner = FluxCoreMLRunner(args.model_dir, model_id=args.base_model or "black-forest-labs/FLUX.1-schnell")
            runner.generate(args.prompt, args.output, steps=args.steps, height=args.height, width=args.width)
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
        download_dir = os.path.join("models", repo_name)
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
        from .model_utils import validate_model
        validate_model(args.model_path)

    elif args.command == "info":
        from .model_utils import show_model_info
        show_model_info(args.model_path)

    elif args.command == "list-models":
        from .model_utils import list_models
        list_models(args.dir)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
