# Alloy ComfyUI Nodes - Complete Reference

## Node Categories

### Core Loaders
- [CoreMLFluxLoader](#coremlfluxloader) - Flux image generation
- [CoreMLLTXVideoLoader](#coremlltxvideoloader) - LTX video generation  
- [CoreMLWanVideoLoader](#coremlwanvideoloader) - Wan video generation

### Integrated Loaders
- [CoreMLFluxWithCLIP](#coremlfluxwithclip) - All-in-one Flux loader

### Conversion
- [CoreMLConverter](#coremlconverter) - Advanced conversion with options
- [CoreMLQuickConverter](#coremlquickconverter) - One-click conversion presets

### Utilities
- [CoreMLModelAnalyzer](#coremlmodelanalyzer) - Inspect model details
- [CoreMLBatchSampler](#cormlbatchsampler) - Parallel batch generation

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
- `model_type`: flux, ltx, wan, hunyuan, lumina, sd
- `quantization`: int4 (recommended), int8, float16
- `output_name` (string): Optional custom folder name
- `force_reconvert` (bool): If True, overwrites existing conversion
- `lora_stack` (LORA_CONFIG): Optional LoRA stack

**Outputs**:
- `model_path` (STRING): Path to the converted `.mlpackage`

**Usage**:
```
CoreMLConverter → CoreMLFluxLoader
```

**Notes**:
- ⚠️ **Blocks UI**: Conversion takes 5-15 minutes. Console shows progress.
- **Smart Cache**: Skips conversion if model already exists (unless forced).

---

### CoreMLQuickConverter

**Category**: Alloy/Conversion  
**Purpose**: One-click conversion for popular models

**Inputs**:
- `preset`:
  - `Flux Schnell (Fast)` → int4
  - `Flux Dev (Quality)` → int4
  - `LTX Video` → int4
  - `Custom` → Use optional inputs below
- `custom_model` (optional): HF ID for custom
- `custom_type` (optional): Model type for custom

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

**Status**: 🚧 Node structure ready, implementation in progress

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

**Status**: 🚧 Node structure ready, implementation in progress

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

| Node | Flux | LTX | Wan | Status |
|------|------|-----|-----|--------|
| CoreMLFluxLoader | ✅ | ❌ | ❌ | Stable |
| CoreMLFluxWithCLIP | ✅ | ❌ | ❌ | Stable |
| CoreMLLTXVideoLoader | ❌ | 🚧 | ❌ | Beta |
| CoreMLWanVideoLoader | ❌ | ❌ | 🚧 | Beta |
| CoreMLModelAnalyzer | ✅ | ✅ | ✅ | Stable |
| CoreMLBatchSampler | ✅ | ✅ | ✅ | Experimental |

---

## Version History

### v0.1.0 (Current)
- Initial release
- Flux image generation support
- Integrated CLIP/T5 loading
- Model analyzer utility
- Batch sampling (experimental)
- LTX/Wan node structures (implementation pending)

---

## Support & Resources

- **Documentation**: [GitHub README](https://github.com/hybridindie/metal-diffusion)
- **Issues**: [GitHub Issues](https://github.com/hybridindie/metal-diffusion/issues)
- **Workflows**: `example_workflows/` directory
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
