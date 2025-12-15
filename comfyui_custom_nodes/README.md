# Metal Diffusion - ComfyUI Custom Nodes

Custom nodes to use Core ML-accelerated transformers (Flux, LTX, Wan) in ComfyUI on Apple Silicon.

## Installation

### Option 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "Metal Diffusion"
3. Click Install
4. Restart ComfyUI

### Option 2: Manual Installation

1. **Install metal-diffusion package** (if not already installed):
   ```bash
   cd /path/to/metal-diffusion
   pip install -e .
   ```

2. **Link to ComfyUI**:
   ```bash
   ln -s /path/to/metal-diffusion/comfyui_custom_nodes /path/to/ComfyUI/custom_nodes/metal-diffusion
   ```

3. **Restart ComfyUI** and the nodes will appear in the "MetalDiffusion" category.

## Usage

### 1. Convert Your Model
First, convert a Flux/LTX model to Core ML using the CLI:

```bash
uv run metal-diffusion convert black-forest-labs/FLUX.1-schnell \
  --type flux \
  --output-dir ~/models/flux_coreml \
  --quantization int4
```

### 2. Place the `.mlpackage` in ComfyUI
Copy the converted `.mlpackage` folder to `ComfyUI/models/unet/`:

```bash
cp -r ~/models/flux_coreml/Flux_Transformer.mlpackage ~/ComfyUI/models/unet/
```

### 3. Use in Workflow
- **Use the Core ML nodes** in your workflow:
   - **For Images (Flux)**: Use "Core ML Flux Loader" 
   - **For Video (LTX/Wan)**: Use "Core ML LTX-Video Loader" or "Core ML Wan Video Loader" (coming soon!)
- Select your `.mlpackage` and model type (flux/ltx/wan)
- Connect it like a standard UNet/Transformer
- Use with your favorite KSampler!

## Nodes

### Core ML Flux Loader (Image)
Loads Flux models converted to Core ML for image generation.

**Inputs:**
- `model_path`: Path to the `.mlpackage` (from `models/unet/`)

**Outputs:**
- `MODEL`: A ComfyUI-compatible model ready for KSampler

### Core ML LTX-Video Loader
Loads LTX-Video models for video generation (placeholder - full implementation coming soon).

**Inputs:**
- `model_path`: Path to the `.mlpackage`
- `num_frames`: Number of frames to generate (default: 25)

**Outputs:**
- `MODEL`: Video model ready for sampling

### Core ML Wan Video Loader
Loads Wan models for video generation (placeholder - full implementation coming soon).

**Inputs:**
- `model_path`: Path to the `.mlpackage`
- `num_frames`: Number of frames to generate (default: 16)

**Outputs:**
- `MODEL`: Video model ready for sampling

## Current Status

**✅ Ready to Use:**
- **Flux Image Generation**: Fully functional with Core ML acceleration

**🚧 Coming Soon:**
- **LTX-Video**: Node structure ready, video latent pack/unpack in progress
- **Wan Video**: Node structure ready, implementation pending
- **LoRA Support**: Standard Core ML constraints apply

## Roadmap

- [ ] Full LTX-Video support
- [ ] Wan 2.x support
- [ ] Integrated CLIP/T5 loading for complete end-to-end ComfyUI workflows
- [ ] ControlNet adaptation

## Troubleshooting

**"Module not found: metal_diffusion"**
- Ensure you ran `pip install -e .` from the metal-diffusion root directory.

**"Model path not found"**
- Check that the `.mlpackage` is in `ComfyUI/models/unet/`
- Verify the path in the node dropdown matches your file

**Shape mismatches**
- Ensure your latent size matches the model's expected input (Flux typically uses 1024x1024 or 512x512)
