# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### "Command not found: alloy"
**Problem**: The CLI tool is not accessible after installation.

**Solution**:
```bash
# Install in development mode
pip install -e .

# Or use uv
uv sync
uv run alloy --help
```

#### "No module named 'alloy'"
**Problem**: Python cannot find the package.

**Solution**:
1. Ensure you're in the correct virtual environment
2. Reinstall the package:
   ```bash
   pip install -e .
   ```
3. Check your Python path:
   ```bash
   python -c "import sys; print(sys.path)"
   ```

---

### Conversion Issues

#### "Out of Memory" during conversion
**Problem**: System runs out of RAM during model conversion.

**Solutions**:
1. **Use int4 quantization** (most effective):
   ```bash
   alloy convert <model> --quantization int4
   ```

2. **Close other applications** to free up RAM

3. **For large models (14B+ parameters)**:
   - Minimum 32GB RAM recommended
   - 64GB RAM for comfortable conversion
   - Convert on a Mac Studio or Mac Pro if available

4. **Reduce batch size** in conversion scripts (advanced users)

#### "Model loading failed" / "Repository not found"
**Problem**: Cannot download from Hugging Face.

**Solutions**:
1. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```

2. **For gated models**: Request access on the model's Hugging Face page first

3. **Check model ID spelling**:
   ```bash
   # Correct
   black-forest-labs/FLUX.1-schnell
   
   # Incorrect
   black-forest-labs/flux-schnell
   ```

4. **Set HF_TOKEN in .env**:
   ```bash
   echo "HF_TOKEN=hf_..." >> .env
   ```

#### "Conversion is very slow"
**Problem**: Conversion takes hours or never completes.

**Expected Times** (on M2 Max, 64GB RAM):
- Flux Schnell: ~5-10 minutes
- LTX-Video: ~8-15 minutes  
- Wan 14B: ~30-60 minutes

**Solutions**:
1. **Check Activity Monitor** for progress:
   - Look for Python process using CPU
   - Neural Engine activity indicates quantization is running

2. **Enable verbose logging**:
   ```bash
   alloy convert <model> --verbose
   ```

3. **First conversion is always slower** due to downloads

#### "CoreMLTools version mismatch"
**Problem**: Error about incompatible CoreMLTools version.

**Solution**:
```bash
pip install --upgrade coremltools>=8.0
```

---

### Runtime Issues

#### "Slow inference" / "Not using Neural Engine"
**Problem**: Generated images take too long.

**Diagnosis**:
1. **Check Activity Monitor**: Search for "Neural Engine"
2. If Neural Engine shows 0% usage, Core ML is falling back to GPU/CPU

**Solutions**:
1. **Verify model path**: Ensure `.mlpackage` is being loaded
   ```bash
   alloy info <path-to-model>
   ```

2. **Use int4 quantization** for better ANE utilization:
   ```bash
   alloy convert <model> --quantization int4
   ```

3. **Check macOS version**: Neural Engine requires macOS 14+ (Sonoma)

4. **Restart your Mac** (sometimes ANE needs a fresh start)

#### "Shape mismatch" errors
**Problem**: Runtime error about tensor shapes.

**Common Causes**:
1. **Wrong resolution**: Use multiples of 64 for Flux
   ```bash
   # Good
   --height 1024 --width 1024
   --height 512 --width 768
   
   # Bad
   --height 1000 --width 1000
   ```

2. **Model mismatch**: Using wrong model type
   ```bash
   # Ensure type matches actual model
   alloy run <path> --type flux
   ```

#### "Black/corrupted images"
**Problem**: Generated images are solid black or garbled.

**Solutions**:
1. **Check VAE**: Ensure VAE is compatible
2. **Try different seeds**:
   ```bash
   alloy run <path> --prompt "..." --seed 42
   ```
3. **Reduce steps** (Flux Schnell works best with 4 steps)
4. **Reconvert model** if issue persists

---

### ComfyUI Integration Issues

#### "Node not found" in ComfyUI
**Problem**: Alloy nodes don't appear.

**Solutions**:
1. **Check symlink**:
   ```bash
   ls -la ~/ComfyUI/custom_nodes/
   ```

2. **Reinstall**:
   ```bash
   cd /path/to/metal-diffusion
   pip install -e .
   ln -sf $(pwd)/comfyui_custom_nodes ~/ComfyUI/custom_nodes/alloy
   ```

3. **Restart ComfyUI** completely

4. **Check terminal output** for errors when ComfyUI starts

#### "Model path not found" in ComfyUI
**Problem**: Cannot load `.mlpackage` in node.

**Solution**:
1. **Move model to correct directory**:
   ```bash
   cp -r converted_models/flux/Flux_Transformer.mlpackage ~/ComfyUI/models/unet/
   ```

2. **Refresh ComfyUI** node inputs (right-click → "Reload")

3. **Check permissions**: Ensure ComfyUI can read the directory

---

### Validation Issues

#### "Validation failed" for converted model
**Problem**: `alloy validate` reports errors.

**Steps**:
1. **Check detailed error**:
   ```bash
   alloy validate <model> --verbose
   ```

2. **Reconvert with float16** instead of int4:
   ```bash
   alloy convert <model> --quantization float16
   ```

3. **Verify source model** is not corrupted

4. **Report issue** on GitHub with validation output

---

## Performance Optimization

### Best Practices

1. **Always use int4 quantization** for production:
   - 4x smaller model size
   - Better Neural Engine utilization
   - Minimal quality loss

2. **Preload models** for batch generation:
   - First inference is slower (model loading)
   - Subsequent inferences are faster

3. **Optimize resolution**:
   - 512x512: Fastest
   - 1024x1024: Balanced
   - 1024x1536: Slower but high quality

4. **Use appropriate steps**:
   - Flux Schnell: 4 steps
   - Flux Dev: 20-50 steps
   - LTX: 20-30 steps

---

## Getting Help

### Before asking for help:

1. **Try validation**:
   ```bash
   alloy validate <your-model>
   ```

2. **Check model info**:
   ```bash
   alloy info <your-model>
   ```

3. **Enable verbose logging**:
   ```bash
   alloy convert <model> --verbose
   ```

### Reporting Issues

Include in your bug report:
- macOS version
- Mac model (M1/M2/M3/M4)
- RAM amount
- Command used
- Full error output with `--verbose`
- Output from `alloy info <model>`

### Resources

- **GitHub Issues**: [hybridindie/metal-diffusion](https://github.com/hybridindie/metal-diffusion/issues)
- **Model Info**: Use `alloy list-models` to see all converted models
- **Discord/Community**: [Coming soon]
