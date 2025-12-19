"""Memory monitoring utilities for Alloy conversions.

Provides system memory monitoring and estimation for conversion memory requirements.
"""

from dataclasses import dataclass
from typing import Optional

# Try to import psutil, gracefully degrade if not available
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


# Memory requirements in GB for different model types (approximate)
MODEL_MEMORY_REQUIREMENTS = {
    "flux": 48,
    "hunyuan": 50,
    "ltx": 20,
    "wan": 24,
    "lumina": 28,
    "sd": 12,
}

# Quantization reduces memory requirements
QUANTIZATION_FACTORS = {
    "int4": 0.5,
    "int8": 0.7,
    "float16": 1.0,
    None: 1.0,
}


@dataclass
class MemoryStatus:
    """Current system memory status."""

    used_gb: float
    available_gb: float
    total_gb: float
    percent_used: float


def get_memory_status() -> Optional[MemoryStatus]:
    """Get current system memory status.

    Returns:
        MemoryStatus with current memory info, or None if psutil unavailable.
    """
    if not PSUTIL_AVAILABLE:
        return None

    mem = psutil.virtual_memory()
    return MemoryStatus(
        used_gb=mem.used / (1024**3),
        available_gb=mem.available / (1024**3),
        total_gb=mem.total / (1024**3),
        percent_used=mem.percent,
    )


def estimate_memory_requirement(model_type: str, quantization: Optional[str]) -> float:
    """Estimate memory requirement for a model conversion.

    Args:
        model_type: Type of model (flux, hunyuan, ltx, wan, lumina, sd)
        quantization: Quantization type (int4, int8, float16, None)

    Returns:
        Estimated memory requirement in GB.
    """
    base_requirement = MODEL_MEMORY_REQUIREMENTS.get(model_type.lower(), 30)
    factor = QUANTIZATION_FACTORS.get(quantization, 1.0)
    return base_requirement * factor


def check_memory_warning(required_gb: float) -> Optional[str]:
    """Check if system has enough memory and return warning if not.

    Args:
        required_gb: Required memory in GB.

    Returns:
        Warning message if memory is low, None otherwise.
    """
    status = get_memory_status()
    if status is None:
        return None

    if status.available_gb < required_gb:
        return (
            f"Low memory warning: {status.available_gb:.1f} GB available, "
            f"~{required_gb:.0f} GB recommended. "
            f"Conversion may fail or be slow."
        )

    return None
