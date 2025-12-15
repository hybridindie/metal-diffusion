# Model Compatibility Matrix

## ✅ Fully Supported Models

### Flux Models

| Model | Convert | Run | ComfyUI | Single File | LoRA | Status |
|-------|---------|-----|---------|-------------|------|--------|
| FLUX.1-Schnell | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| FLUX.1-Dev | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| FLUX.2 | ✅ | ✅ | ✅ | ❌ | ❌ | Beta |

**Notes**:
- **Schnell**: 4 steps, guidance_scale=0.0
- **Dev**: 20-50 steps, guidance_scale=3.5
- **LoRA**: Supported via "Baking" (CLI or ComfyUI)
- **FLUX.2**: Newer architecture, no pooled projections
- **Single File**: Civitai `.safetensors` support

**Quantization Support**:
- ✅ float16
- ✅ int4 (recommended)

---

### LTX-Video Models

| Model | Convert | Run | ComfyUI | Single File | Status |
|-------|---------|-----|---------|-------------|--------|
| LTX-Video | ✅ | ✅ | 🚧 | ✅ | Beta |

**Notes**:
- Video generation (up to 257 frames)
- 512x512 or 768x768 resolution
- ComfyUI node structure ready, implementation pending

**Quantization Support**:
- ✅ float16
- ✅ int4

---

### Wan Models

| Model | Convert | Run | ComfyUI | Single File | Status |
|-------|---------|-----|---------|-------------|--------|
| Wan 2.1 | ✅ | ✅ | 🚧 | ❌ | Experimental |
| Wan 2.2 | ✅ | ✅ | 🚧 | ❌ | Experimental |

**Notes**:
- Large models (14B parameters)
- RequiresHugging Face format (not single file yet)
- ComfyUI node structure ready, implementation pending

**Quantization Support**:
- ⚠️ float16 (requires 64GB+ RAM)
- ✅ int4 (recommended, requires 32GB+ RAM)

---

### Hunyuan Models

| Model | Convert | Run | ComfyUI | Single File | Status |
|-------|---------|-----|---------|-------------|--------|
| HunyuanVideo | ✅ | ✅ | ❌ | ❌ | Experimental |

**Notes**:
- Video generation
- High quality but slow
- No ComfyUI integration yet

---

### Stable Diffusion Models

| Model | Convert | Run | ComfyUI | Single File | Status |
|-------|---------|-----|---------|-------------|--------|
| SDXL | ✅ | ✅ | ❌ | ⚠️ | Via Apple's tool |
| SD 1.5 | ✅ | ✅ | ❌ | ⚠️ | Via Apple's tool |
| SD 3 | ✅ | ✅ | ❌ | ⚠️ | Via Apple's tool |

**Notes**:
- Uses Apple's `python-coreml-stable-diffusion`
- Limited to what Apple's tool supports
- No custom node integration

---

## 🚧 Planned Support

### ControlNet

| Feature | Status | ETA |
|---------|--------|-----|
| Flux ControlNet | 🚧 Planned | Q1 2025 |
| SD ControlNet | ⚠️ Via Apple | Now |

### LoRA

| Feature | Status | ETA |
|---------|--------|-----|
| Flux LoRA | ✅ "Baking" | Released |
| SD LoRA | ⚠️ Via Apple | Now |

---

## Feature Support Matrix

### Core Features

| Feature | Flux | LTX | Wan | Hunyuan | SD |
|---------|------|-----|-----|---------|-----|
| Text-to-Image | ✅ | ❌ | ❌ | ❌ | ✅ |
| Text-to-Video | ❌ | ✅ | ✅ | ✅ | ❌ |
| Image-to-Image | ⚠️ | ⚠️ | ✅ | ❌ | ✅ |
| Img2Video | ❌ | ✅ | ✅ | ❌ | ❌ |
| Inpainting | ❌ | ❌ | ❌ | ❌ | ✅ |

### Technical Features

| Feature | Flux | LTX | Wan | Hunyuan | SD |
|---------|------|-----|-----|---------|-----|
| int4 Quant | ✅ | ✅ | ✅ | ⚠️ | ❌ |
| int8 Quant | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| float16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| ANE Accel | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| Hybrid Mode | ✅ | ✅ | ✅ | ✅ | ❌ |

**Legend**:
- ✅ Fully supported
- ⚠️ Partial support / limitations
- 🚧 In progress
- ❌ Not supported

---

## Platform Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **macOS** | 14.0+ (Sonoma) |
| **Chip** | Apple Silicon (M1+) |
| **RAM** | 16GB (32GB recommended) |
| **Storage** | 50GB free |
| **Python** | 3.11+ |

### Recommended Setup

| Component | Recommendation |
|-----------|----------------|
| **macOS** | 14.5+ |
| **Chip** | M2 Max or higher |
| **RAM** | 64GB |
| **Storage** | 200GB+ SSD |
| **Python** | 3.11 with uv |

---

## Model Size Requirements

### RAM Requirements by Model

| Model | float16 | int4 | Recommended RAM |
|-------|---------|------|-----------------|
| Flux Schnell | 14GB | 8GB | 32GB |
| Flux Dev | 14GB | 8GB | 32GB |
| LTX-Video | 12GB | 7GB | 32GB |
| Wan 14B | 28GB+ | 18GB | 64GB |
| HunyuanVideo | 32GB+ | 20GB | 64GB |

### Storage Requirements

| Model | Source | Converted (float16) | Converted (int4) |
|-------|--------|---------------------|------------------|
| Flux | 24GB | 12GB | 3.2GB |
| LTX | 9GB | 5GB | 1.3GB |
| Wan 14B | 28GB | 28GB | 7.5GB |

---

## Known Limitations

### General

- No Windows/Linux support (Apple Silicon required)
- No ControlNet/LoRA yet (planned)
- Large models require significant RAM
- First inference always slower (model loading)

### Model-Specific

**Flux**:
- No negative prompts in Schnell variant
- FLUX.2 single-file support pending

**LTX**:
- Max 257 frames
- Limited to 768x768 resolution

**Wan**:
- Very slow on <64GB RAM
- No single-file support yet
- Requires Hugging Face download

---

## Testing Coverage

| Model Type | Unit Tests | Integration Tests | E2E Tests |
|------------|------------|-------------------|-----------|
| Flux | ✅ | ✅ | ✅ |
| LTX | ✅ | ✅ | ⚠️ |
| Wan | ✅ | ✅ | ❌ |
| Hunyuan | ⚠️ | ⚠️ | ❌ |

---

## Compatibility Notes

### macOS Versions

- **Sonoma (14.x)**: Fully supported ✅
- **Ventura (13.x)**: Limited (older CoreML) ⚠️
- **Monterey (12.x)**: Not recommended ❌

### Apple Silicon

- **M4**: Optimal (best ANE) ✅
- **M3**: Excellent ✅
- **M2**: Great ✅
- **M1**: Good (limited RAM) ⚠️

---

## Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| 0.1.0 | Dec 2024 | Initial release with Flux/LTX/Wan support |
| 0.2.0 | Planned | ComfyUI video nodes, ControlNet support |

---

## Updating This Matrix

Found a compatibility issue or tested a new model? Please:
1. Open an issue on [GitHub](https://github.com/hybridindie/alloy)
2. Include: macOS version, model name, error details
3. Tag with `compatibility`
