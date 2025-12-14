# Metal Diffusion Converter

A unified toolchain for converting open-source diffusion models (Stable Diffusion, Wan 2.1/2.2) into Apple's **Core ML** format for hardware-accelerated inference on macOS.

## Features

-   **Core ML Conversion**: Optimize models for Apple Silicon (Neural Engine/GPU).
-   **Wan 2.x Support**:
    -   Supports **Wan 2.1** and **Wan 2.2** (Text-to-Video).
    -   Supports **Image-to-Video / Edit** models (automatic 36-channel input detection).
    -   Implements **Int4 Quantization** to run 14B models on consumer Macs (64GB RAM recommended).
-   **Stable Diffusion Support**: Wraps Apple's `python_coreml_stable_diffusion` for SDXL and SD3.
-   **Full Pipeline**: Automates Download -> Convert -> Upload to Hugging Face.
-   **Dependency Management**: Uses `uv` to resolve complex conflicts between legacy Core ML scripts and modern Hugging Face libraries.

## Installation

This project uses **uv** for dependency management.

1.  **Install uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync Dependencies**:
    ```bash
    uv sync
    ```

3.  **Hugging Face Login** (Required for Uploads/Gated Models):
    
    You can either run:
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
    ```env
    HF_TOKEN=hf_...
    OUTPUT_DIR=converted_models
    ```

## Usage

All commands are run via `uv run main.py`.

### Convert Stable Diffusion (SDXL/SD3)
```bash
uv run main.py convert stabilityai/stable-diffusion-xl-base-1.0 \
  --type sd \
  --output-dir converted_models/sdxl \
  --quantization float16
```

### Convert Wan 2.1 / 2.2
Supports both T2V (Text-to-Video) and I2V (Image-to-Video). The converter automatically detects the input channels from the model config.

```bash
# Text-to-Video (16 channels)
uv run main.py convert Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --type wan \
  --output-dir converted_models/wan2.2 \
  --quantization int4

# Image-to-Video (36 channels)
uv run main.py convert Wan-AI/Wan2.1-I2V-14B-720P-Diffusers \
  --type wan \
  --output-dir converted_models/wan_i2v \
  --quantization int4
```

### Full Pipeline
Downloads a model, converts it, and uploads the Core ML package to your Hugging Face account.

```bash
uv run main.py pipeline stabilityai/sd-turbo \
  --target-repo your-username/sd-turbo-coreml \
  --type sd
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
