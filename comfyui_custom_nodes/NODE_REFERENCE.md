# Alloy ComfyUI Nodes - Complete Reference

## Node Categories

### Core Loaders
- [CoreMLFluxLoader](#coremlfluxloader) - Flux image generation
- [CoreMLLTXVideoLoader](#coremlltxvideoloader) - LTX video generation
- [CoreMLWanVideoLoader](#coremlwanvideoloader) - Wan video generation
- [CoreMLHunyuanVideoLoader](#coremlhunyuanvideoloader) - Hunyuan video generation
- [CoreMLLuminaLoader](#coremlluminaloader) - Lumina Image 2.0 generation
- [CoreMLControlNetLoader](#coremlcontrolnetloader) - Load ControlNet models
- [CoreMLApplyControlNet](#coremlapplycontrolnet) - Apply ControlNet

### Integrated Loaders
- [CoreMLFluxWithCLIP](#coremlfluxwithclip) - All-in-one Flux loader
- [CoreMLLuminaWithCLIP](#coremlluminawithclip) - All-in-one Lumina loader
- [CoreMLLTXVideoWithCLIP](#coremlltxvideowithclip) - All-in-one LTX Video loader
- [CoreMLWanVideoWithCLIP](#coremlwanvideowithclip) - All-in-one Wan Video loader
- [CoreMLHunyuanVideoWithCLIP](#coremlhunyuanvideowithclip) - All-in-one Hunyuan Video loader

### Conversion
- [CoreMLConverter](#coremlconverter) - Advanced conversion with options
- [CoreMLQuickConverter](#coremlquickconverter) - One-click conversion presets

### Utilities
- [CoreMLModelAnalyzer](#coremlmodelanalyzer) - Inspect model details
- [CoreMLBatchSampler](#coremlbatchsampler) - Parallel batch generation

---

## Node Descriptions

### CoreMLFluxLoader

**Category**: Alloy  
**Purpose**: Load Flux Core ML transformer for image generation

**Inputs**:
- `model_path` (unet dropdown): Path to `.mlpackage` file

**Outputs**:
- `MODEL`: Flux transformer ready for sampling

**Usage**:
```
CoreMLFluxLoader → KSampler
```

**Notes**:
- Requires separate CLIP/VAE loaders
- Supports Flux.1-Schnell and Flux.1-Dev
- Core ML transformer runs on ANE for speed

---

---

### CoreMLFluxWithCLIP

**Category**: Alloy  
**Purpose**: All-in-one loader with integrated text encoders

**Inputs**:
- `transformer_path` (unet dropdown): Core ML transformer
- `clip_model` (dropdown): HF model ID
  - `black-forest-labs/FLUX.1-schnell`
  - `black-forest-labs/FLUX.1-dev`

**Outputs**:
- `MODEL`: Flux transformer
- `CLIP`: Combined CLIP-L + T5 text encoders
- `VAE`: VAE decoder

**Usage**:
```
CoreMLFluxWithCLIP → MODEL+CLIP+VAE → KSampler
```

**Advantages**:
- One node instead of three
- Automatic CLIP/T5 loading

---

### CoreMLLoraConfig

**Category**: Alloy/Conversion  
**Purpose**: Define LoRA configuration for baking (chainable)

**Inputs**:
- `lora_name` (dropdown): Select LoRA from `models/loras/`
- `strength_model` (float): Strength for Transformer/UNet (default 1.0)
- `strength_clip` (float): Strength for Text Encoder (default 1.0)
- `previous_lora` (LORA_CONFIG): Optional input from another LoRA node

**Outputs**:
- `lora_config`: Configuration stack

**Usage**:
```
CoreMLLoraConfig (Style A)
  ↓
CoreMLLoraConfig (Style B)
  ↓
CoreMLConverter
```

---

### CoreMLConverter

**Category**: Alloy/Conversion
**Purpose**: Convert models to Core ML with full control

**Inputs**:
- `model_source` (string): Hugging Face ID (e.g., `black-forest-labs/FLUX.1-schnell`) or local path
- `model_type`: flux, flux-controlnet, ltx, wan, hunyuan, lumina, sd
- `quantization`: int4 (recommended), int8, float16, float32
- `output_name` (string): Optional custom folder name
- `force_reconvert` (bool): If True, overwrites existing conversion
- `skip_validation` (bool): Skip pre-flight validation checks
- `lora_stack` (LORA_CONFIG): Optional LoRA stack (Flux only)

**Outputs**:
- `model_path` (STRING): Path to the converted `.mlpackage`

**Usage**:
```
CoreMLConverter → CoreMLFluxLoader
```

**Notes**:
- ⚠️ **Blocks UI**: Conversion takes 5-15 minutes. Console shows progress.
- **Smart Cache**: Skips conversion if model already exists (unless forced).
- **ControlNet**: Use `flux-controlnet` type for Flux ControlNet models.

---

### CoreMLQuickConverter

**Category**: Alloy/Conversion
**Purpose**: One-click conversion for popular models

**Inputs**:
- `preset`:
  - `Flux Schnell (Fast)` → int4
  - `Flux Dev (Quality)` → int4
  - `Flux ControlNet (Canny)` → int4
  - `Flux ControlNet (Depth)` → int4
  - `LTX Video` → int4
  - `Hunyuan Video` → int4
  - `Wan 2.1 Video` → int4
  - `Lumina Image 2.0` → int4
  - `Custom` → Use optional inputs below
- `custom_model` (optional): HF ID for custom preset
- `custom_type` (optional): Model type for custom preset
- `custom_quantization` (optional): Quantization for custom preset

**Outputs**:
- `model_path` (STRING): Path to converted model

**Usage**:
```
CoreMLQuickConverter (Preset: Flux Schnell) → CoreMLFluxLoader
```

---

### CoreMLLTXVideoLoader

**Category**: Alloy/Video  
**Purpose**: Load LTX-Video model for video generation

**Inputs**:
- `model_path` (unet dropdown): Path to `.mlpackage`
- `num_frames` (int, 1-257, default 25): Number of frames

**Outputs**:
- `MODEL`: LTX video model

**Usage**:
```
CoreMLLTXVideoLoader → VideoKSampler
```

**Notes**:
- Implements latent packing/unpacking for LTX transformer format
- Supports T5 text encoder attention masks
- VAE compression: spatial=32, temporal=8

**Status**: Beta - Forward pass implemented, testing in progress

---

### CoreMLWanVideoLoader

**Category**: Alloy/Video  
**Purpose**: Load Wan model for video generation

**Inputs**:
- `model_path` (unet dropdown): Path to `.mlpackage`
- `num_frames` (int, 1-128, default 16): Number of frames

**Outputs**:
- `MODEL`: Wan video model

**Usage**:
```
CoreMLWanVideoLoader → VideoKSampler
```

**Notes**:
- Supports Wan 2.1 and 2.2 models (T2V and I2V)
- Simple 5D latent format (no packing required)
- Uses Core ML VAE decoder when available

**Status**: Beta - Forward pass implemented, testing in progress

---

### CoreMLHunyuanVideoLoader

**Category**: Alloy/Video
**Purpose**: Load HunyuanVideo model for video generation

**Inputs**:
- `model_path` (unet dropdown): Path to `.mlpackage`
- `num_frames` (int, 1-128, default 16): Number of frames

**Outputs**:
- `MODEL`: Hunyuan video model

**Usage**:
```
CoreMLHunyuanVideoLoader → VideoKSampler
```

**Notes**:
- Supports embedded guidance (guidance_scale * 1000 format)
- Requires pooled projections and attention masks
- VAE scale factor: 16

**Status**: Beta - Forward pass implemented, testing in progress

---

### CoreMLLuminaLoader

**Category**: Alloy
**Purpose**: Load Lumina Image 2.0 model for image generation

**Inputs**:
- `model_path` (unet dropdown): Path to `.mlpackage`

**Outputs**:
- `MODEL`: Lumina image model

**Usage**:
```
CoreMLLuminaLoader → KSampler
```

**Notes**:
- Requires separate text encoder (Gemma 2B) and VAE loaders
- Use `CoreMLLuminaWithCLIP` for simpler workflow

---

### CoreMLLuminaWithCLIP

**Category**: Alloy
**Purpose**: All-in-one Lumina loader with integrated Gemma text encoder

**Inputs**:
- `transformer_path` (unet dropdown): Core ML transformer

**Outputs**:
- `MODEL`: Lumina transformer
- `CLIP`: Gemma 2B text encoder
- `VAE`: VAE decoder

**Usage**:
```
CoreMLLuminaWithCLIP → MODEL+CLIP+VAE → KSampler
```

**Advantages**:
- One node instead of three
- Automatic Gemma 2B and VAE loading from HuggingFace

---

### CoreMLLTXVideoWithCLIP

**Category**: Alloy/Video
**Purpose**: All-in-one LTX-Video loader with integrated T5 text encoder and VAE

**Inputs**:
- `transformer_path` (unet dropdown): Core ML transformer
- `num_frames` (int, 1-257, default 25): Number of video frames

**Outputs**:
- `MODEL`: LTX video transformer
- `CLIP`: T5 text encoder
- `VAE`: Video VAE decoder

**Usage**:
```
CoreMLLTXVideoWithCLIP → MODEL+CLIP+VAE → VideoKSampler
```

**Advantages**:
- One node instead of three
- Automatic T5 and VAE loading from HuggingFace (Lightricks/LTX-Video)
- Configurable frame count

**Status**: Beta

---

### CoreMLWanVideoWithCLIP

**Category**: Alloy/Video
**Purpose**: All-in-one Wan Video loader with integrated text encoder and VAE

**Inputs**:
- `transformer_path` (unet dropdown): Core ML transformer
- `num_frames` (int, 1-128, default 16): Number of video frames
- `model_variant` (dropdown): HuggingFace model variant
  - `Wan-AI/Wan2.1-T2V-14B-Diffusers` (Text-to-Video)
  - `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` (Image-to-Video)

**Outputs**:
- `MODEL`: Wan video transformer
- `CLIP`: Text encoder
- `VAE`: Video VAE decoder

**Usage**:
```
CoreMLWanVideoWithCLIP → MODEL+CLIP+VAE → VideoKSampler
```

**Advantages**:
- One node instead of three
- Supports both T2V and I2V model variants
- Automatic text encoder and VAE loading

**Status**: Beta

---

### CoreMLHunyuanVideoWithCLIP

**Category**: Alloy/Video
**Purpose**: All-in-one HunyuanVideo loader with integrated text encoders and VAE

**Inputs**:
- `transformer_path` (unet dropdown): Core ML transformer
- `num_frames` (int, 1-128, default 16): Number of video frames

**Outputs**:
- `MODEL`: Hunyuan video transformer
- `CLIP`: Dual text encoders (LLAVA + CLIP)
- `VAE`: Video VAE decoder

**Usage**:
```
CoreMLHunyuanVideoWithCLIP → MODEL+CLIP+VAE → VideoKSampler
```

**Advantages**:
- One node instead of three
- Automatic dual text encoder loading (LLAVA + CLIP)
- Automatic VAE loading from HuggingFace

**Status**: Beta

---

### CoreMLControlNetLoader

**Category**: Alloy
**Purpose**: Load a converted ControlNet model (Core ML)

**Inputs**:
- `controlnet_path` (controlnet dropdown): Path to `.mlpackage`

**Outputs**:
- `COREML_CONTROLNET`: ControlNet model object

**Usage**:
```
CoreMLControlNetLoader → CoreMLApplyControlNet
```

---

### CoreMLApplyControlNet

**Category**: Alloy
**Purpose**: Apply a Core ML ControlNet to the Flux model

**Inputs**:
- `model` (MODEL): Core ML Flux model (must be from CoreMLFluxLoader/Wrapper)
- `controlnet` (COREML_CONTROLNET): Loaded ControlNet
- `image` (IMAGE): Control image (Canny/Depth map etc.)
- `strength` (FLOAT, default 1.0): Control strength

**Outputs**:
- `MODEL`: Model with ControlNet attached

**Usage**:
```
CoreMLFluxLoader
      ↓
CoreMLApplyControlNet ← CoreMLControlNetLoader
      ↓
   KSampler
```

**Notes**:
- Multiple ControlNets can be chained.
- Ensure the ControlNet model matches the base model (Flux.1).

---

### CoreMLModelAnalyzer

**Category**: Alloy/Utilities  
**Purpose**: Analyze and display Core ML model information

**Inputs**:
- `model_path` (unet dropdown): Path to `.mlpackage`

**Outputs**:
- `STRING`: Detailed analysis report

**Information Displayed**:
- Model type and size
- Input/output specifications
- Tensor shapes and types
- Metadata (author, description)

**Usage**:
```
CoreMLModelAnalyzer → ShowText
```

**Use Cases**:
- Debugging model issues
- Verifying conversion correctness
- Understanding model architecture
- Checking quantization applied

---

### CoreMLBatchSampler

**Category**: Alloy/Advanced  
**Purpose**: Generate multiple images in one go

**Inputs**:
- `model` (MODEL): The loaded model
- `positive` (CONDITIONING): Positive prompt
- `negative` (CONDITIONING): Negative prompt
- `latent_image` (LATENT): Starting latents
- `seed` (int): Random seed
- `steps` (int, 1-10000, default 20): Sampling steps
- `cfg` (float, 0-100, default 8.0): Guidance scale
- `sampler_name` (dropdown): Sampler algorithm
- `scheduler` (dropdown): Noise schedule
- `denoise` (float, 0-1, default 1.0): Denoise strength
- `batch_size` (int, 1-16, default 4): Number of images

**Outputs**:
- `LATENT`: Batched latent outputs

**Usage**:
```
CoreMLBatchSampler → VAEDecode → SaveImage
```

**Performance**:
- Processes batches more efficiently
- Memory usage scales with batch size
- Recommended: 4-8 batch size on 64GB RAM

**Notes**:
- Currently sequential processing
- Parallel implementation planned
- Monitor memory usage

---

## Workflow Examples

### Basic Flux Image Generation

```
CoreMLFluxLoader 
  ↓ MODEL
KSampler ← CLIP (from DualCLIPLoader)
  ↓ LATENT
VAEDecode ← VAE (from VAELoader)
  ↓ IMAGE
SaveImage
```

### Simplified Flux (All-in-One)

```
CoreMLFluxWithCLIP
  ↓ MODEL + CLIP + VAE
KSampler
  ↓ LATENT
VAEDecode
  ↓ IMAGE
SaveImage
```

### Model Debugging

```
CoreMLFluxLoader
  ↓ model_path
CoreMLModelAnalyzer
  ↓ STRING
ShowText
```

### Batch Generation

```
CoreMLFluxLoader → MODEL
  ↓
CoreMLBatchSampler (batch_size=4)
  ↓ LATENT
VAEDecode
  ↓ IMAGE (4 images)
SaveImage
```

---

## Tips & Best Practices

### Memory Management
- Close other applications during conversion
- Use int4 quantization for large models
- Monitor Activity Monitor → Memory tab

### Performance Optimization
- **Flux Schnell**: 4 steps optimal
- **Flux Dev**: 20-50 steps recommended
- Use int4 quantization for best ANE utilization
- Batch size 4-8 for best throughput

### Resolution Guidelines
- Flux: Multiples of 64 (512, 1024, 1536)
- LTX: 512x512 or 768x768
- Higher resolution = slower but better quality

### Common Issues

**"Model not found"**
- Check `.mlpackage` is in `ComfyUI/models/unet/`
- Verify file permissions

**"Shape mismatch"**
- Ensure latent size matches model expectations
- Use multiples of 64 for dimensions

**Slow performance**
- Check quantization (int4 recommended)
- Verify ANE usage in Activity Monitor
- Close background apps

---

## Node Compatibility Matrix

| Node | Flux | LTX | Wan | Hunyuan | Lumina | Status |
|------|------|-----|-----|---------|--------|--------|
| CoreMLFluxLoader | ✅ | ❌ | ❌ | ❌ | ❌ | Stable |
| CoreMLFluxWithCLIP | ✅ | ❌ | ❌ | ❌ | ❌ | Stable |
| CoreMLLTXVideoLoader | ❌ | ✅ | ❌ | ❌ | ❌ | Beta |
| CoreMLLTXVideoWithCLIP | ❌ | ✅ | ❌ | ❌ | ❌ | Beta |
| CoreMLWanVideoLoader | ❌ | ❌ | ✅ | ❌ | ❌ | Beta |
| CoreMLWanVideoWithCLIP | ❌ | ❌ | ✅ | ❌ | ❌ | Beta |
| CoreMLHunyuanVideoLoader | ❌ | ❌ | ❌ | ✅ | ❌ | Beta |
| CoreMLHunyuanVideoWithCLIP | ❌ | ❌ | ❌ | ✅ | ❌ | Beta |
| CoreMLLuminaLoader | ❌ | ❌ | ❌ | ❌ | ✅ | Stable |
| CoreMLLuminaWithCLIP | ❌ | ❌ | ❌ | ❌ | ✅ | Stable |
| CoreMLModelAnalyzer | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| CoreMLBatchSampler | ✅ | ✅ | ✅ | ✅ | ✅ | Experimental |

---

## Version History

### v0.3.7 (Current)
- Added CoreMLLTXVideoWithCLIP integrated loader (T5 + VAE)
- Added CoreMLWanVideoWithCLIP integrated loader with T2V/I2V support
- Added CoreMLHunyuanVideoWithCLIP integrated loader (dual text encoders)
- All video integrated loaders auto-download components from HuggingFace

### v0.3.6
- Implemented LTX-Video ComfyUI wrapper with packing/unpacking
- Implemented Wan Video ComfyUI wrapper for T2V and I2V models
- Implemented HunyuanVideo ComfyUI wrapper with guidance embedding
- Added CoreMLLuminaWithCLIP integrated loader
- All video loaders now functional (Beta status)

### v0.3.5
- Fixed CoreMLFluxLoader node registration bug (RETURN_TYPES/FUNCTION)
- Added Flux ControlNet support (flux-controlnet model type)
- Added float32 quantization option
- Added skip_validation parameter for faster iteration
- New presets: Flux ControlNet (Canny/Depth), Hunyuan Video, Wan 2.1 Video, Lumina Image 2.0
- Standardized workflow naming (category prefixes)
- Added descriptive Note nodes to all example workflows

### v0.3.4
- Relaxed torch version constraint for broader compatibility
- Fixed torch/torchvision version conflicts
- Simplified ComfyUI requirements.txt

### v0.3.1
- Added HunyuanVideo loader node
- Added Lumina Image 2.0 loader node
- Memory monitoring in converter nodes
- Progress tracking disabled for ComfyUI context

### v0.1.0
- Initial release
- Flux image generation support
- Integrated CLIP/T5 loading
- Model analyzer utility
- Batch sampling (experimental)
- LTX/Wan node structures (implementation pending)

---

## Support & Resources

- **Documentation**: [GitHub README](https://github.com/hybridindie/alloy)
- **Issues**: [GitHub Issues](https://github.com/hybridindie/alloy/issues)
- **Workflows**: `example_workflows/` directory
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
