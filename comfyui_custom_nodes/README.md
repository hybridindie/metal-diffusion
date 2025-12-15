# Metal Diffusion - ComfyUI Custom Nodes

Custom nodes to use Core ML-accelerated transformers (Flux, LTX, Wan) in ComfyUI on Apple Silicon.

## Installation

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
- Add the **"Core ML Transformer Loader"** node
- Select your `.mlpackage` and model type (flux/ltx/wan)
- Connect it like a standard UNet/Transformer
- Use with your favorite KSampler!

## Nodes

### Core ML Transformer Loader
**Inputs:**
- `model_path`: Path to the `.mlpackage` (from `models/unet/`)
- `model_type`: Type of model (`flux`, `ltx`, or `wan`)

**Outputs:**
- `MODEL`: A ComfyUI-compatible model ready for sampling

## Current Limitations

- **Flux Only**: Only Flux models are fully implemented. LTX and Wan support coming soon.
- **No LoRA support yet**: Standard Core ML constraints apply.
- **Text Encoders**: Currently requires separate CLIP/T5 nodes (hybrid execution).

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
