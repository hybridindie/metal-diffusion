import os
from typing import Optional

from safetensors import safe_open

from alloy.logging import get_logger

logger = get_logger(__name__)

# Model signature patterns for auto-detection
# Each model type has:
#   - required_any: At least one of these key patterns must be present
#   - required_all: All of these key patterns must be present (optional)
#   - forbidden: None of these key patterns can be present (optional)
#   - config_classes: Class names to match in HF config files
MODEL_SIGNATURES = {
    "flux": {
        "required_any": ["double_blocks", "single_blocks"],
        "forbidden": ["txt_in"],  # Distinguishes from Hunyuan
        "config_classes": ["FluxTransformer2DModel", "Flux2Transformer2DModel"],
    },
    "ltx": {
        "required_any": ["caption_projection", "scale_shift_table"],
        "config_classes": ["LTXVideoTransformer3DModel"],
    },
    "hunyuan": {
        "required_all": ["transformer_blocks", "single_transformer_blocks"],
        "required_any": ["txt_in", "guidance_in"],
        "forbidden": ["double_blocks", "single_blocks"],
        "config_classes": ["HunyuanVideoTransformer3DModel"],
    },
    "wan": {
        "required_any": ["patch_embedding", "blocks.0.attn"],
        "forbidden": ["single_transformer_blocks", "double_blocks", "caption_projection"],
        "config_classes": ["WanTransformer3DModel"],
    },
    "lumina": {
        "required_any": ["adaln_single", "layers.0.gate"],
        "forbidden": ["caption_projection", "txt_in", "double_blocks"],
        "config_classes": ["Lumina2Transformer2DModel"],
    },
}


def _key_matches(keys: list, pattern: str) -> bool:
    """Check if any key contains the pattern."""
    return any(pattern in k for k in keys)


def _detect_from_safetensor_keys(keys: list) -> tuple[Optional[str], float]:
    """
    Match safetensor keys against MODEL_SIGNATURES.

    Returns:
        (model_type, confidence) where confidence is 0.0-1.0
    """
    matches = []

    for model_type, signature in MODEL_SIGNATURES.items():
        # Check forbidden keys first - if any present, skip this type
        forbidden = signature.get("forbidden", [])
        if any(_key_matches(keys, f) for f in forbidden):
            continue

        # Check required_all - all must be present
        required_all = signature.get("required_all", [])
        if required_all and not all(_key_matches(keys, r) for r in required_all):
            continue

        # Check required_any - at least one must be present
        required_any = signature.get("required_any", [])
        if not required_any:
            continue

        matching_any = sum(1 for r in required_any if _key_matches(keys, r))
        if matching_any == 0:
            continue

        # Calculate confidence based on how many patterns matched
        confidence = matching_any / len(required_any)
        if required_all:
            confidence = (confidence + 1.0) / 2  # Boost for matching required_all

        matches.append((model_type, confidence))

    if not matches:
        return None, 0.0

    # Sort by confidence descending
    matches.sort(key=lambda x: x[1], reverse=True)

    # Warn if ambiguous (multiple matches with similar confidence)
    if len(matches) > 1 and matches[1][1] > 0.5:
        logger.warning(
            "Ambiguous model detection: %s. Using %s (confidence: %.0f%%)",
            [m for m, _ in matches],
            matches[0][0],
            matches[0][1] * 100,
        )

    return matches[0]


def _detect_from_config(config: dict) -> Optional[str]:
    """Detect model type from HuggingFace config dict."""
    arch = config.get("_class_name", "") or str(config.get("architectures", []))

    for model_type, signature in MODEL_SIGNATURES.items():
        for class_name in signature.get("config_classes", []):
            if class_name in arch:
                return model_type

    # Fallback to simple string matching
    arch_lower = arch.lower()
    for model_type in MODEL_SIGNATURES:
        if model_type in arch_lower:
            return model_type

    return None


def detect_model_type(model_path: str) -> Optional[str]:
    """
    Detect model type from local file or HuggingFace repo.

    Supports: flux, ltx, hunyuan, wan, lumina

    Args:
        model_path: Path to .safetensors file or HuggingFace repo ID

    Returns:
        Model type string or None if unknown
    """
    # 1. Local File Detection
    if os.path.isfile(model_path):
        if not model_path.endswith(".safetensors"):
            return None

        try:
            with safe_open(model_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                model_type, confidence = _detect_from_safetensor_keys(keys)
                if model_type and confidence >= 0.5:
                    logger.debug(
                        "Detected %s from safetensor keys (confidence: %.0f%%)",
                        model_type,
                        confidence * 100,
                    )
                    return model_type
        except Exception as e:
            logger.warning("Error reading safetensor file %s: %s", model_path, e)
            return None

    # 2. Hugging Face Repo Detection
    elif "/" in model_path:
        try:
            from huggingface_hub import hf_hub_download
            import json

            # Try to fetch config.json or transformer/config.json
            config_path = None
            for filename in ["transformer/config.json", "config.json", "model_index.json"]:
                try:
                    config_path = hf_hub_download(repo_id=model_path, filename=filename)
                    break
                except Exception:
                    continue

            if config_path and os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)

                model_type = _detect_from_config(config)
                if model_type:
                    logger.debug("Detected %s from HuggingFace config", model_type)
                    return model_type

        except Exception as e:
            logger.debug("HuggingFace detection failed for %s: %s", model_path, e)

    return None

import shutil
import tempfile
import time

def cleanup_old_temp_files(prefix="alloy_quant_", max_age_hours=1):
    """
    Cleans up old temporary directories created by Alloy that might have been left over due to crashes.
    """
    temp_base = tempfile.gettempdir()
    count = 0
    now = time.time()
    
    try:
        if not os.path.exists(temp_base):
            return 0
            
        for filename in os.listdir(temp_base):
            if filename.startswith(prefix):
                full_path = os.path.join(temp_base, filename)
                if os.path.isdir(full_path):
                    try:
                        # Check age
                        stat = os.stat(full_path)
                        age_hours = (now - stat.st_mtime) / 3600
                        if age_hours > max_age_hours:
                            shutil.rmtree(full_path)
                            count += 1
                    except Exception:
                        pass # Ignore permission errors etc
    except Exception:
        pass
        
    return count
