# ComfyUI Example Workflows

This directory contains example workflows that demonstrate how to use Metal Diffusion's Core ML nodes in ComfyUI.

## Available Workflows

### 1. Flux Basic Text-to-Image (`flux_basic_txt2img.json`)

A simple Flux.1 Schnell workflow for text-to-image generation using Core ML acceleration.

**Features:**
- Core ML Transformer (accelerated on Apple Neural Engine)
- Standard CLIP/T5 text encoding (PyTorch)
- VAE decode (PyTorch)
- 4 steps, Euler sampler

**Requirements:**
- Converted Flux Core ML model (`.mlpackage`)
- Flux VAE (`ae.safetensors`)
- Flux CLIP models (`t5xxl_fp16.safetensors`, `clip_l.safetensors`)

**Usage:**
1. Place your `Flux_Transformer.mlpackage` in `ComfyUI/models/unet/`
2. Ensure VAE and CLIP models are in their respective directories
3. Load this workflow in ComfyUI
4. Update the prompt and generate!

**Expected Performance (M2 Max):**
- ~2-3 seconds per image at 1024x1024
- Significantly faster than pure PyTorch on MPS

---

### 2. Flux Image-to-Image (`flux_img2img.json`)

Transform existing images using Flux with Core ML acceleration.

**Features:**
- Load and encode input image with VAE
- Apply artistic styles or modifications
- Adjustable denoise strength (0.7 = moderate changes)
- 20 steps for high quality

**Requirements:**
- Same as basic workflow
- Input image (any resolution, will be resized to latent space)

**Usage:**
1. Load workflow
2. Upload your image in the LoadImage node
3. Adjust the denoise strength in KSampler (0.5-0.9 range)
4. Modify prompt to describe desired changes
5. Generate!

**Denoise Strength Guide:**
- 0.3-0.5: Subtle refinement, keep original structure
- 0.6-0.7: Moderate changes, artistic style transfer
- 0.8-0.9: Major transformation, only rough composition kept

## Customization Tips

### Changing Resolution
Modify the `EmptyLatentImage` node:
- 512x512: Mobile-friendly, faster
- 1024x1024: Standard quality
- 1024x1536: Portrait orientation

### Adjusting Steps
Flux Schnell works best with **4 steps**. For Flux Dev models:
- Increase steps to 20-50
- Set guidance scale to 3.5

### Using Different Schedulers
Try different samplers in the KSampler node:
- `euler` - Fast, good quality (recommended)
- `dpmpp_2m` - Better detail
- `heun` - Smoother results

## Troubleshooting

**"Model not found"**
- Ensure the `.mlpackage` path in the Core ML Transformer Loader matches your file
- Check that it's in `ComfyUI/models/unet/`

**"Shape mismatch"**
- Verify your EmptyLatentImage dimensions are compatible with Flux (multiples of 64)
- Flux typically uses 512, 1024, or 1536 dimensions

**Slow performance**
- Ensure Core ML model is using ANE: check Activity Monitor for "Neural Engine" usage
- Try int4 quantization for faster inference (convert with `--quantization int4`)

## Creating Your Own Workflows

The Core ML Transformer Loader works like a standard model loader. You can:
1. Add ControlNet nodes (if converted to Core ML)
2. Use with LoRA (requires PyTorch model patching)
3. Combine with other ComfyUI extensions

For advanced usage, see the main `README.md` in the parent directory.
