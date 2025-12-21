# Alloy ComfyUI Example Workflows

Example workflows demonstrating Alloy's Core ML nodes for Apple Silicon acceleration.

## Quick Start

1. Convert a model: `alloy convert black-forest-labs/FLUX.1-schnell --type flux`
2. Place `.mlpackage` in `ComfyUI/models/unet/`
3. Load any workflow below and generate!

---

## Image Generation Workflows

### Flux Text-to-Image - Basic (`Alloy - Flux Text to Image (Basic).json`)

Basic Flux.1 Schnell workflow with separate loader nodes.

**Nodes Used:** `CoreMLFluxLoader`, `DualCLIPLoader`, `VAELoader`, `KSampler`

**Best For:** Users who want full control over each component.

---

### Flux Text-to-Image - All-in-One (`Alloy - Flux Text to Image (All-in-One).json`)

Simplified Flux workflow using the integrated loader.

**Nodes Used:** `CoreMLFluxWithCLIP`, `KSampler`

**Best For:** Quick setup - one node loads transformer, CLIP, and VAE together.

---

### Flux Image-to-Image (`Alloy - Flux Image to Image.json`)

Transform existing images with Flux.

**Nodes Used:** `CoreMLFluxWithCLIP`, `LoadImage`, `VAEEncode`, `KSampler`

**Denoise Guide:**
- 0.3-0.5: Subtle refinement
- 0.6-0.7: Moderate style transfer
- 0.8-0.9: Major transformation

---

### Flux with Core ML VAE (`Alloy - Flux with CoreML VAE.json`)

Full Apple Silicon acceleration - both transformer AND VAE on ANE.

**Nodes Used:** `CoreMLFluxWithCLIP`, `CoreMLVAELoader`, `CoreMLVAEDecode`

**Benefit:** VAE decode is ~3x faster than PyTorch for high-resolution images.

**Setup:**
```bash
alloy convert black-forest-labs/FLUX.1-schnell --type flux
alloy convert black-forest-labs/FLUX.1-schnell --type vae --vae-components decoder
```

---

### Flux with ControlNet (`Alloy - Flux with ControlNet.json`)

Guided generation using Canny edges or depth maps.

**Nodes Used:** `CoreMLFluxLoader`, `CoreMLControlNetLoader`, `CoreMLApplyControlNet`, `LoadImage`

**Setup:**
```bash
alloy convert black-forest-labs/FLUX.1-schnell --type flux
alloy convert Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro --type flux-controlnet
```

**Strength Guide:**
- 0.3-0.5: Loose guidance, more creative freedom
- 0.6-0.8: Balanced control
- 0.9-1.0: Strict adherence to control image

---

### Lumina Image 2.0 (`Alloy - Lumina Image 2.0.json`)

High-quality image generation with Alpha-VLLM's Lumina.

**Nodes Used:** `CoreMLLuminaWithCLIP`, `KSampler`

**Features:** Uses Gemma 2B text encoder, great for artistic images.

---

## Video Generation Workflows

### LTX Video Generation (`Alloy - LTX Video Generation.json`)

Generate short video clips with Lightricks LTX-Video.

**Nodes Used:** `CoreMLLTXVideoWithCLIP`, `KSampler`, `VHS_VideoCombine`

**Settings:**
- 25 frames ≈ 1 second at 24fps
- Resolution: 512x512 or 768x512
- Steps: 20-30 for quality

---

### Wan 2.1 Video Generation (`Alloy - Wan 2.1 Video Generation.json`)

Text-to-video generation with Wan 2.1.

**Nodes Used:** `CoreMLWanVideoWithCLIP`, `KSampler`, `VHS_VideoCombine`

**Variants:**
- T2V (Text-to-Video): Generate from prompts
- I2V (Image-to-Video): Animate an image

---

### Hunyuan Video Generation (`Alloy - Hunyuan Video Generation.json`)

Cinematic video generation with Tencent's HunyuanVideo.

**Nodes Used:** `CoreMLHunyuanVideoWithCLIP`, `KSampler`, `VHS_VideoCombine`

**Features:** Dual text encoders (LLAVA + CLIP), embedded guidance.

---

## Conversion Workflows

### Quick Model Conversion (`Alloy - Quick Model Conversion.json`)

One-click conversion with presets.

**Nodes Used:** `CoreMLQuickConverter`

**Presets:** Flux Schnell, Flux Dev, LTX Video, Wan Video, Hunyuan Video, Lumina

---

### Convert with LoRA Baking (`Alloy - Convert with LoRA Baking.json`)

Bake LoRA weights into the model during conversion.

**Nodes Used:** `CoreMLLoraConfig`, `CoreMLConverter`

**Benefit:** No runtime LoRA overhead - styles are permanent in the model.

---

### Convert and Generate Pipeline (`Alloy - Convert and Generate Pipeline.json`)

Complete pipeline demonstrating the full workflow.

**Nodes Used:** `CoreMLQuickConverter`, `CoreMLModelAnalyzer`, `CoreMLFluxLoader`, `KSampler`

**Flow:**
1. Convert model (cached after first run)
2. Analyze to verify correctness
3. Generate images

---

## Utility Workflows

### Model Analysis Utility (`Alloy - Model Analysis Utility.json`)

Inspect converted Core ML models.

**Nodes Used:** `CoreMLModelAnalyzer`, `ShowText`

**Shows:**
- Input/output tensor shapes
- Quantization applied
- Model metadata
- Memory requirements

---

## Node Reference

| Category | Node | Purpose |
|----------|------|---------|
| **Loaders** | `CoreMLFluxLoader` | Load Flux transformer |
| | `CoreMLFluxWithCLIP` | All-in-one Flux loader |
| | `CoreMLLTXVideoWithCLIP` | LTX video + T5 + VAE |
| | `CoreMLWanVideoWithCLIP` | Wan video + encoder + VAE |
| | `CoreMLHunyuanVideoWithCLIP` | Hunyuan + dual encoders + VAE |
| | `CoreMLLuminaWithCLIP` | Lumina + Gemma + VAE |
| **ControlNet** | `CoreMLControlNetLoader` | Load ControlNet model |
| | `CoreMLApplyControlNet` | Apply control to generation |
| **VAE** | `CoreMLVAELoader` | Load Core ML VAE |
| | `CoreMLVAEEncode` | Encode image to latents |
| | `CoreMLVAEDecode` | Decode latents to image |
| | `CoreMLVAETile` | Tiled decode for large images |
| **Conversion** | `CoreMLConverter` | Full conversion options |
| | `CoreMLQuickConverter` | Preset-based conversion |
| | `CoreMLLoraConfig` | Configure LoRA for baking |
| **Utilities** | `CoreMLModelAnalyzer` | Inspect model details |
| | `CoreMLBatchSampler` | Batch image generation |

---

## Performance Tips

### Optimal Settings by Model

| Model | Steps | Sampler | CFG | Resolution |
|-------|-------|---------|-----|------------|
| Flux Schnell | 4 | euler | 1.0 | 1024x1024 |
| Flux Dev | 20-50 | euler | 3.5 | 1024x1024 |
| LTX Video | 20-30 | euler | 7.5 | 512x512 |
| Wan Video | 30-50 | euler | 6.0 | 832x480 |
| Hunyuan Video | 30-50 | euler | 6.0 | 720x480 |
| Lumina | 20-30 | euler | 7.0 | 1024x1024 |

### Quantization

- **int4**: Best performance, recommended for most use cases
- **int8**: Balanced quality/speed
- **float16**: Maximum quality, larger files

### Memory Management

- Close other apps during conversion
- Use int4 for large models (14B+ parameters)
- Monitor Activity Monitor → Memory tab

---

## Troubleshooting

**"Model not found"**
- Check `.mlpackage` is in correct directory (`unet/`, `vae/`, `controlnet/`)
- Verify file permissions

**"Shape mismatch"**
- Use multiples of 64 for dimensions
- Match resolution to model training (check notes in workflow)

**Slow performance**
- Verify ANE usage in Activity Monitor
- Use int4 quantization
- Close background applications

**Video nodes not working**
- Install ComfyUI-VideoHelperSuite for `VHS_VideoCombine`
- Check frame count matches model capabilities

---

## Requirements

- macOS 14+ (Sonoma or newer)
- Apple Silicon Mac (M1/M2/M3/M4)
- ComfyUI with Alloy custom nodes installed
- Converted Core ML models (`.mlpackage`)

For video workflows, install:
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
