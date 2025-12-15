# Alloy - ComfyUI Custom Nodes

Custom nodes to use Core ML-accelerated transformers (Flux, LTX, Wan) in ComfyUI on Apple Silicon.

## Installation

### Option 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "Alloy"
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
   ln -s /path/to/metal-diffusion/comfyui_custom_nodes /path/to/ComfyUI/custom_nodes/alloy
   ```

3. **Restart ComfyUI** and the nodes will appear in the "Alloy" category.

## Usage

### Method 1: All-in-ComfyUI (Recommended) 🆕

1. **Add `CoreMLQuickConverter` node**
2. **Select Preset** (e.g., "Flux Schnell")
3. **Connect to `CoreMLFluxLoader`** (or use result string)
4. **Run**: The first run will convert the model (checks cache automatically).

### Method 2: CLI Conversion (Classic)

1. **Convert via CLI**: `alloy convert ...`
2. **Place in models/unet/**
3. **Load via `CoreMLFluxLoader`**

## Nodes

The suite includes **8 Custom Nodes**:

### Loaders
- **CoreMLFluxLoader**: Basic loader
- **CoreMLFluxWithCLIP**: Integrated CLIP/VAE loader 🌟
- **CoreMLLTXVideoLoader**: LTX-Video support
- **CoreMLWanVideoLoader**: Wan 2.x support

### Conversion 🆕
- **CoreMLConverter**: Advanced conversion
- **CoreMLQuickConverter**: One-click presets

### Utilities
- **CoreMLModelAnalyzer**: Inspect model specs
- **CoreMLBatchSampler**: Parallel generation

## Roadmap

- [x] Flux Image Generation
- [x] Integrated CLIP/T5 Loading
- [x] In-ComfyUI Conversion
- [x] Model Analysis Tools
- [ ] Full LTX-Video support (implementation pending)
- [ ] Wan 2.x support (implementation pending)

## Troubleshooting

**"Module not found: alloy"**
- Ensure you ran `pip install -e .` from the project root directory.

**"Model path not found"**
- Check that the `.mlpackage` is in `ComfyUI/models/unet/`
- Verify the path in the node dropdown matches your file

**Shape mismatches**
- Ensure your latent size matches the model's expected input (Flux typically uses 1024x1024 or 512x512)
