import argparse
import os
import sys
from .converter import SDConverter
from .wan_converter import WanConverter
from .hf_utils import HFManager
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "converted_models")

def main():
    parser = argparse.ArgumentParser(description="Diffusion to Core ML Converter")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download Command
    download_parser = subparsers.add_parser("download", help="Download a model from Hugging Face")
    download_parser.add_argument("repo_id", type=str, help="Hugging Face Repo ID")
    download_parser.add_argument("--output-dir", type=str, default=os.path.join("models"), help="Directory to save the model")
    
    # Convert Command
    convert_parser = subparsers.add_parser("convert", help="Convert a model to Core ML")
    convert_parser.add_argument("model_path", type=str, help="Path to local model or HF Repo ID")
    convert_parser.add_argument("--type", type=str, choices=["sd", "wan"], required=True, help="Type of model")
    convert_parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    convert_parser.add_argument("--quantization", type=str, default="float16", choices=["float16", "int8", "int4"], help="Quantization level")
    
    # Upload Command
    upload_parser = subparsers.add_parser("upload", help="Upload converted model to Hugging Face")
    upload_parser.add_argument("local_path", type=str, help="Path to the converted model folder")
    upload_parser.add_argument("repo_id", type=str, help="Target Hugging Face Repo ID")
    
    # Full Pipeline Command
    pipeline_parser = subparsers.add_parser("pipeline", help="Download -> Convert -> Upload")
    pipeline_parser.add_argument("repo_id", type=str, help="Source HF Repo ID")
    pipeline_parser.add_argument("--target-repo", type=str, help="Target HF Repo ID for upload")
    pipeline_parser.add_argument("--type", type=str, choices=["sd", "wan"], required=True)
    pipeline_parser.add_argument("--quantization", type=str, default="float16")

    args = parser.parse_args()
    
    hf_manager = HFManager()

    if args.command == "download":
        hf_manager.download_model(args.repo_id, local_dir=os.path.join(args.output_dir, args.repo_id.split("/")[-1]))

    elif args.command == "convert":
        if args.type == "sd":
            converter = SDConverter(args.model_path, args.output_dir, args.quantization)
        else:
            converter = WanConverter(args.model_path, args.output_dir, args.quantization)
        converter.convert()

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

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
