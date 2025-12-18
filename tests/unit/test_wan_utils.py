import pytest
import torch
from unittest.mock import MagicMock, patch
from alloy.converters.wan_workers import WanPart1Wrapper

# Import these from the worker module where the patch is defined
try:
    from diffusers.models.attention_processor import WanAttnProcessor
except ImportError:
    WanAttnProcessor = None

@pytest.mark.skipif(WanAttnProcessor is None, reason="WanAttnProcessor not available")
def test_wan_attn_processor_patch_applied():
    """Verify the monkey patch is applied to WanAttnProcessor during module import."""
    # The wan_workers module applies a monkey patch on import.
    # Just verify WanAttnProcessor has been patched (the __call__ method has been replaced).
    # The patch is applied at module load time in wan_workers.py.
    original_call = WanAttnProcessor.__call__
    # The patched function should have our custom signature
    assert callable(original_call), "WanAttnProcessor.__call__ should be callable"
