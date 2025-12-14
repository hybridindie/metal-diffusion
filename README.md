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

You can run the tool using `uv run metal-diffusion`.

### Convert Stable Diffusion (SDXL/SD3)
```bash
uv run metal-diffusion convert stabilityai/stable-diffusion-xl-base-1.0 \
  --type sd \
  --output-dir converted_models/sdxl \
  --quantization float16
```

### Convert Wan 2.1 / 2.2
Supports both T2V (Text-to-Video) and I2V (Image-to-Video). The converter automatically detects the input channels from the model config.

```bash
# Text-to-Video (16 channels)
uv run metal-diffusion convert Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --type wan \
  --output-dir converted_models/wan2.2 \
  --quantization int4

# Image-to-Video (36 channels)
uv run metal-diffusion convert Wan-AI/Wan2.1-I2V-14B-720P-Diffusers \
  --type wan \
  --output-dir converted_models/wan_i2v \
  --quantization int4
```

### Full Pipeline
Downloads a model, converts it, and uploads the Core ML package to your Hugging Face account.

### Run Locally
Verify your converted models by generating an image directly.

```bash
uv run metal-diffusion run converted_models/wan2.2  \
  --prompt "A astronaut riding a horse on mars, photorealistic, 4k" \
  --type wan \
  --output result.png
```

## ComfyUI Integration

To use these models with **ComfyUI** (specifically with the `ComfyUI-CoreMLSuite` custom nodes):

1.  **Locate the `.mlpackage`**: Inside your output directory (e.g., `converted_models/wan2.2/Wan2.1_Transformer.mlpackage`).
2.  **Move to ComfyUI**:
    *   Copy the `.mlpackage` folder to `ComfyUI/models/unet/` (or `models/coreml/` depending on your node setup).
3.  **Load in ComfyUI**:
    *   Use the **CoreML Unet Loader** node.
    *   Select your model directory.
    *   Connect to a standard VAE and CLIP (since we currently run Text Encoding/VAE in hybrid mode or standard nodes often expect separated components).

*Note: For Wan 2.1, ensure you use the correct input dimensions (T2V=16ch, I2V=36ch) which match the converted model's expectation.*

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
