# Alloy ⚡️
> [!IMPORTANT]
> **🚧 Work in Progress 🚧**
> This project is currently under active development. APIs and CLI commands are subject to change. Use with caution.

A unified toolchain for converting open-source diffusion models (Stable Diffusion, Wan 2.1/2.2) into Apple's **Core ML** format for hardware-accelerated inference on macOS.

## Features

-   **Core ML Conversion**: Optimize models for Apple Silicon (Neural Engine/GPU).
-   **Wan 2.x Support**:
    -   Supports **Wan 2.1** and **Wan 2.2** (Text-to-Video).
    -   Supports **Image-to-Video / Edit** models (automatic 36-channel input detection).
    -   Implements **Int4 Quantization** to run 14B models on consumer Macs (64GB RAM recommended).
-   **Flux ControlNet Support**:
    -   Full support for Flux ControlNet residuals (Base Model + ControlNet Model).
    -   **ComfyUI Nodes**: Dedicated nodes for loading and applying Core ML ControlNets.
-   **Stable Diffusion Support**: Wraps Apple's `python_coreml_stable_diffusion` for SDXL and SD3.
-   **Lumina-Image 2.0 Support**: Implements Next-Gen DiT conversion using Gemma 2B text encoder.
-   **Full Pipeline**: Automates Download -> Convert -> Upload to Hugging Face.
-   **Dependency Management**: Uses `uv` to resolve complex conflicts between legacy Core ML scripts and modern Hugging Face libraries.

## Installation

1.  **Install uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install the Tool**:
    ```bash
    uv sync
    # Or install globally:
    # uv tool install .
    ```

3.  **Hugging Face Login** (Required for Uploads/Gated Models):
    ```bash
    uv run huggingface-cli login
    ```
    Or configure it via `.env` (recommended).

4.  **Configuration**:
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` to set your `HF_TOKEN` and `OUTPUT_DIR`.

## Usage

You can run the tool using `uv run alloy`.

### Convert Stable Diffusion (SDXL/SD3)
```bash
uv run alloy convert stabilityai/stable-diffusion-xl-base-1.0 \
  --type sd \
  --output-dir converted_models/sdxl \
  --quantization float16
```

### Convert Lumina-Image 2.0
```bash
uv run alloy convert Alpha-VLLM/Lumina-Image-2.0 \
  --type lumina \
  --quantization int4
```

### Convert Flux ControlNet (New!)
Support for converting **Flux ControlNet** models (X-Labs, InstantX, etc.) and preparing Base Flux models to accept them.

**1. Convert Base Model (ControlNet Ready)**
You must re-convert your base model with `--controlnet` to inject the residual inputs.
```bash
uv run alloy convert black-forest-labs/FLUX.1-schnell \
  --output-dir converted_models/flux_controlnet \
  --quantization int4 \
  --controlnet
```

**2. Convert ControlNet Model**
```bash
uv run alloy convert x-labs/flux-controlnet-canny \
  --type flux-controlnet \
  --output-dir converted_models/flux_canny \
  --quantization int4
```

### Convert Wan 2.1 / 2.2
Supports both T2V (Text-to-Video) and I2V (Image-to-Video). The converter automatically detects the input channels from the model config.

```bash
# Basic Conversion
uv run alloy convert black-forest-labs/FLUX.1-schnell \
  --output-dir converted_models/flux \
  --quantization int4

# With LoRA Baking
uv run alloy convert black-forest-labs/FLUX.1-schnell \
  --output-dir converted_models/flux_style \
  --quantization int4 \
  --lora "path/to/style.safetensors:0.8:1.0" \
  --lora "path/to/fix.safetensors:1.0"
```

# Image-to-Video (36 channels)
uv run alloy convert Wan-AI/Wan2.1-I2V-14B-720P-Diffusers \
  --type wan \
  --output-dir converted_models/wan_i2v \
  --quantization int4
```

### Civitai / Single-File Models
Support for directly loading `.safetensors` files (e.g., from Civitai) for **Flux** and **LTX-Video**. 

**Auto-Detection**: The CLI automatically detects the model architecture (Flux vs LTX) from the file header, so you can often skip the `--type` argument.

```bash
# Convert a single file checkpoint (Type auto-detected!)
uv run alloy convert /path/to/flux_schnell.safetensors \
  --output-dir converted_models/flux_civiai \
  --quantization int4
```

### Full Pipeline
Downloads a model, converts it, and uploads the Core ML package to your Hugging Face account.

### Run Locally
Verify your converted models by generating an image directly.

```bash
uv run alloy run converted_models/wan2.2  \
  --prompt "A astronaut riding a horse on mars, photorealistic, 4k" \
  --type wan \
  --output result.png
```

### Benchmark Performance
Measure real performance on your hardware:

```bash
uv run alloy run converted_models/flux \
  --prompt "test image" \
  --benchmark \
  --benchmark-runs 5 \
  --benchmark-output benchmarks.json
```

This will run 5 iterations and report:
- **Mean/median generation time**
- **Memory usage**
- **Per-step timing** breakdown
- **Statistical variance**

Results are saved to JSON for further analysis.

## ComfyUI Integration

Alloy includes **custom nodes** for seamless ComfyUI integration with Core ML acceleration!

### Installation

```bash
# 1. Install silicon-alloy
pip install -e .

# 2. Link to ComfyUI
ln -s /path/to/metal-diffusion/comfyui_custom_nodes /path/to/ComfyUI/custom_nodes/alloy

# 3. Restart ComfyUI
```

### Quick Start (ComfyUI)

1. **Install Alloy** via ComfyUI Manager
2. **Convert Model** using the `CoreMLQuickConverter` node
   - Select preset (e.g., "Flux Schnell")
   - Run once to convert (caches automatically)
3. **Load Model** using `CoreMLFluxWithCLIP`
4. **Generate!**

### Example Workflows

Check out `comfyui_custom_nodes/example_workflows/` for ready-to-use examples:
- **Basic Text-to-Image**: Simple Flux workflow
- **All-in-One**: Integrated CLIP/T5/VAE loading
- **Smart Conversion**: Convert models directly in ComfyUI

See the [ComfyUI Node Reference](comfyui_custom_nodes/NODE_REFERENCE.md) for full documentation of all 8 nodes.

## Performance Benchmarks

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for detailed performance data on various Apple Silicon chips.

Key highlights (M2 Max):
- **Flux.1-Schnell (1024x1024)**: ~2.3s
- **LTX-Video (25 frames)**: ~28s
- **Memory Usage**: ~8GB (int4) vs 22GB (PyTorch)

## Utility Commands

### Validate Models
```bash
alloy validate converted_models/flux/Flux_Transformer.mlpackage
```

### Show Model Info
```bash
alloy info converted_models/flux/Flux_Transformer.mlpackage
```

### List All Converted Models
```bash
alloy list-models
# or specify directory
alloy list-models --dir /path/to/models
```


## Architecture

-   **`main.py`**: CLI entry point.
-   **`converter.py`**: Base class and Stable Diffusion conversion logic.
-   **`wan_converter.py`**: Custom implementation for Wan 2.x, featuring lazy imports and dynamic input shaping.
-   **`hf_utils.py`**: Utilities for Hugging Face authentication and file operations.
-   **`pyproject.toml`**: Dependency overrides to force compatibility between `coremltools` and `diffusers`.

## Requirements
-   macOS 14+ (Sonoma) or newer.
-   Python 3.11+.
-   Apple Silicon (M1/M2/M3/M4).
