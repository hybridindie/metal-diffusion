import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import time
import shutil
from alloy.utils.general import detect_model_type, cleanup_old_temp_files

class TestGeneralUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("alloy.utils.general.os.path.isfile")
    @patch("alloy.utils.general.safe_open")
    def test_detect_model_type_local_flux(self, mock_safe_open, mock_isfile):
        mock_isfile.return_value = True
        mock_f = MagicMock()
        mock_f.keys.return_value = ["double_blocks.0.weight", "single_blocks.0.weight"]
        mock_safe_open.return_value.__enter__.return_value = mock_f
        
        self.assertEqual(detect_model_type("model.safetensors"), "flux")

    @patch("alloy.utils.general.os.path.isfile")
    @patch("alloy.utils.general.safe_open")
    def test_detect_model_type_local_ltx(self, mock_safe_open, mock_isfile):
        mock_isfile.return_value = True
        mock_f = MagicMock()
        mock_f.keys.return_value = ["scale_shift_table", "caption_projection"]
        mock_safe_open.return_value.__enter__.return_value = mock_f
        
        self.assertEqual(detect_model_type("model.safetensors"), "ltx")
        
    @patch("alloy.utils.general.os.path.isfile")
    def test_detect_model_type_not_safetensors(self, mock_isfile):
        mock_isfile.return_value = True
        self.assertIsNone(detect_model_type("model.pth"))

    @patch("alloy.utils.general.os.path.isfile")
    @patch("huggingface_hub.hf_hub_download")
    def test_detect_model_type_hf_flux(self, mock_hf_download, mock_isfile):
        mock_isfile.return_value = False # Treat as repo ID
        
        # Mock config file content
        config_path = os.path.join(self.test_dir, "config.json")
        with open(config_path, "w") as f:
            f.write('{"_class_name": "FluxPipeline"}')
            
        mock_hf_download.return_value = config_path
        
        self.assertEqual(detect_model_type("black-forest-labs/FLUX.1-schnell"), "flux")

    def test_cleanup_old_temp_files(self):
        # Create a "old" directory
        old_dir = os.path.join(tempfile.gettempdir(), "alloy_quant_test_old")
        os.makedirs(old_dir, exist_ok=True)
        
        # Set mtime to 2 hours ago
        past_time = time.time() - (2 * 3600 + 100)
        os.utime(old_dir, (past_time, past_time))
        
        # Create a "new" directory
        new_dir = os.path.join(tempfile.gettempdir(), "alloy_quant_test_new")
        os.makedirs(new_dir, exist_ok=True)
        
        # Run cleanup
        count = cleanup_old_temp_files(prefix="alloy_quant_test_", max_age_hours=1)
        
        self.assertEqual(count, 1)
        self.assertFalse(os.path.exists(old_dir))
        self.assertTrue(os.path.exists(new_dir))
        
        # Clean up new dir
        shutil.rmtree(new_dir)
