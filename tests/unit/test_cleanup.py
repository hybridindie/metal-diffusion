
import os
import time
import tempfile
import pytest
from unittest.mock import patch
from alloy.utils.general import cleanup_old_temp_files

def test_cleanup_respects_age():
    """Test that cleanup utility respects the max age parameter."""
    with tempfile.TemporaryDirectory() as base_temp:
        # We still patch gettempdir to force the utility to look in our component test dir
        with patch("alloy.utils.general.tempfile.gettempdir", return_value=base_temp):
            # Create a "new" directory (should stay)
            new_dir = os.path.join(base_temp, "alloy_quant_new")
            os.mkdir(new_dir)
            
            # Create an "old" directory (should go)
            old_dir = os.path.join(base_temp, "alloy_quant_old")
            os.mkdir(old_dir)
            
            # Set times
            now = time.time()
            # New: 1 minute ago
            os.utime(new_dir, (now - 60, now - 60))
            # Old: 2 hours ago
            os.utime(old_dir, (now - 7200, now - 7200))

            # Run cleanup
            count = cleanup_old_temp_files(max_age_hours=1)
            
            assert count == 1
            assert os.path.exists(new_dir)
            assert not os.path.exists(old_dir)

def test_cleanup_ignores_non_alloy_files():
    """Test that cleanup ignores files/dirs without the prefix."""
    with tempfile.TemporaryDirectory() as base_temp:
        with patch("alloy.utils.general.tempfile.gettempdir", return_value=base_temp):
            # Create other dir
            other = os.path.join(base_temp, "other_tmp")
            os.mkdir(other)
            
            # Create alloy dir (old)
            alloy = os.path.join(base_temp, "alloy_quant_dead")
            os.mkdir(alloy)
            
            # Set times
            now = time.time()
            # Old
            os.utime(other, (now - 7200, now - 7200))
            os.utime(alloy, (now - 7200, now - 7200))
            
            # Run cleanup
            count = cleanup_old_temp_files(max_age_hours=1)
            
            assert count == 1
            assert os.path.exists(other)
            assert not os.path.exists(alloy)
