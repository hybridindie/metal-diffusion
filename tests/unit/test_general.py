import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import time
import shutil
from alloy.utils.general import (
    detect_model_type,
    cleanup_old_temp_files,
    _detect_from_safetensor_keys,
    MODEL_SIGNATURES,
)


class TestModelSignatures(unittest.TestCase):
    """Tests for MODEL_SIGNATURES structure."""

    def test_all_model_types_have_required_any(self):
        """Every model type must have required_any patterns."""
        for model_type, sig in MODEL_SIGNATURES.items():
            self.assertIn("required_any", sig, f"{model_type} missing required_any")
            self.assertTrue(len(sig["required_any"]) > 0, f"{model_type} has empty required_any")

    def test_all_model_types_have_config_classes(self):
        """Every model type should have config_classes for HF detection."""
        for model_type, sig in MODEL_SIGNATURES.items():
            self.assertIn("config_classes", sig, f"{model_type} missing config_classes")


class TestDetectFromSafetensorKeys(unittest.TestCase):
    """Tests for _detect_from_safetensor_keys helper function."""

    def test_detect_flux_with_double_blocks(self):
        """Flux detected via double_blocks key."""
        keys = ["double_blocks.0.weight", "other_key"]
        model_type, confidence = _detect_from_safetensor_keys(keys)
        self.assertEqual(model_type, "flux")
        self.assertGreaterEqual(confidence, 0.5)

    def test_detect_flux_with_single_blocks(self):
        """Flux detected via single_blocks key."""
        keys = ["single_blocks.0.weight", "other_key"]
        model_type, confidence = _detect_from_safetensor_keys(keys)
        self.assertEqual(model_type, "flux")
        self.assertGreaterEqual(confidence, 0.5)

    def test_detect_ltx_with_caption_projection(self):
        """LTX detected via caption_projection key."""
        keys = ["caption_projection.weight", "transformer.blocks.0"]
        model_type, confidence = _detect_from_safetensor_keys(keys)
        self.assertEqual(model_type, "ltx")
        self.assertGreaterEqual(confidence, 0.5)

    def test_detect_ltx_with_scale_shift_table(self):
        """LTX detected via scale_shift_table key."""
        keys = ["scale_shift_table", "other_key"]
        model_type, confidence = _detect_from_safetensor_keys(keys)
        self.assertEqual(model_type, "ltx")
        self.assertGreaterEqual(confidence, 0.5)

    def test_detect_hunyuan_with_txt_in(self):
        """Hunyuan detected via transformer_blocks + single_transformer_blocks + txt_in."""
        keys = [
            "transformer_blocks.0.weight",
            "single_transformer_blocks.0.weight",
            "txt_in.weight",
        ]
        model_type, confidence = _detect_from_safetensor_keys(keys)
        self.assertEqual(model_type, "hunyuan")
        self.assertGreaterEqual(confidence, 0.5)

    def test_detect_wan_with_patch_embedding(self):
        """Wan detected via patch_embedding key."""
        keys = ["patch_embedding.weight", "transformer_blocks.0"]
        model_type, confidence = _detect_from_safetensor_keys(keys)
        self.assertEqual(model_type, "wan")
        self.assertGreaterEqual(confidence, 0.5)

    def test_detect_lumina_with_adaln_single(self):
        """Lumina detected via adaln_single key."""
        keys = ["adaln_single.weight", "layers.0.norm"]
        model_type, confidence = _detect_from_safetensor_keys(keys)
        self.assertEqual(model_type, "lumina")
        self.assertGreaterEqual(confidence, 0.5)

    def test_flux_not_detected_with_txt_in(self):
        """Flux should NOT be detected if txt_in is present (Hunyuan marker)."""
        keys = ["double_blocks.0.weight", "txt_in.weight"]
        model_type, confidence = _detect_from_safetensor_keys(keys)
        # Should not be flux due to forbidden txt_in
        self.assertNotEqual(model_type, "flux")

    def test_unknown_keys_return_none(self):
        """Unknown key patterns should return None."""
        keys = ["unknown_layer.weight", "random_key"]
        model_type, confidence = _detect_from_safetensor_keys(keys)
        self.assertIsNone(model_type)
        self.assertEqual(confidence, 0.0)

    def test_empty_keys_return_none(self):
        """Empty key list should return None."""
        keys = []
        model_type, confidence = _detect_from_safetensor_keys(keys)
        self.assertIsNone(model_type)


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

    @patch("alloy.utils.general.os.path.isfile")
    @patch("alloy.utils.general.safe_open")
    def test_detect_model_type_local_hunyuan(self, mock_safe_open, mock_isfile):
        """Hunyuan detected from local safetensor file."""
        mock_isfile.return_value = True
        mock_f = MagicMock()
        mock_f.keys.return_value = [
            "transformer_blocks.0.weight",
            "single_transformer_blocks.0.weight",
            "txt_in.weight",
        ]
        mock_safe_open.return_value.__enter__.return_value = mock_f

        self.assertEqual(detect_model_type("model.safetensors"), "hunyuan")

    @patch("alloy.utils.general.os.path.isfile")
    @patch("alloy.utils.general.safe_open")
    def test_detect_model_type_local_wan(self, mock_safe_open, mock_isfile):
        """Wan detected from local safetensor file."""
        mock_isfile.return_value = True
        mock_f = MagicMock()
        mock_f.keys.return_value = ["patch_embedding.weight", "blocks.0.attn"]
        mock_safe_open.return_value.__enter__.return_value = mock_f

        self.assertEqual(detect_model_type("model.safetensors"), "wan")

    @patch("alloy.utils.general.os.path.isfile")
    @patch("alloy.utils.general.safe_open")
    def test_detect_model_type_local_lumina(self, mock_safe_open, mock_isfile):
        """Lumina detected from local safetensor file."""
        mock_isfile.return_value = True
        mock_f = MagicMock()
        mock_f.keys.return_value = ["adaln_single.weight", "layers.0.gate"]
        mock_safe_open.return_value.__enter__.return_value = mock_f

        self.assertEqual(detect_model_type("model.safetensors"), "lumina")

    @patch("alloy.utils.general.os.path.isfile")
    @patch("huggingface_hub.hf_hub_download")
    def test_detect_model_type_hf_hunyuan(self, mock_hf_download, mock_isfile):
        """Hunyuan detected from HuggingFace config."""
        mock_isfile.return_value = False

        config_path = os.path.join(self.test_dir, "config.json")
        with open(config_path, "w") as f:
            f.write('{"_class_name": "HunyuanVideoTransformer3DModel"}')

        mock_hf_download.return_value = config_path

        self.assertEqual(detect_model_type("tencent/HunyuanVideo"), "hunyuan")

    @patch("alloy.utils.general.os.path.isfile")
    @patch("huggingface_hub.hf_hub_download")
    def test_detect_model_type_hf_wan(self, mock_hf_download, mock_isfile):
        """Wan detected from HuggingFace config."""
        mock_isfile.return_value = False

        config_path = os.path.join(self.test_dir, "config.json")
        with open(config_path, "w") as f:
            f.write('{"_class_name": "WanTransformer3DModel"}')

        mock_hf_download.return_value = config_path

        self.assertEqual(detect_model_type("Wan-AI/Wan2.1"), "wan")

    @patch("alloy.utils.general.os.path.isfile")
    @patch("huggingface_hub.hf_hub_download")
    def test_detect_model_type_hf_lumina(self, mock_hf_download, mock_isfile):
        """Lumina detected from HuggingFace config."""
        mock_isfile.return_value = False

        config_path = os.path.join(self.test_dir, "config.json")
        with open(config_path, "w") as f:
            f.write('{"_class_name": "Lumina2Transformer2DModel"}')

        mock_hf_download.return_value = config_path

        self.assertEqual(detect_model_type("Alpha-VLLM/Lumina-Image-2.0"), "lumina")

    @patch("alloy.utils.general.os.path.isfile")
    @patch("huggingface_hub.hf_hub_download")
    def test_detect_model_type_hf_ltx(self, mock_hf_download, mock_isfile):
        """LTX detected from HuggingFace config."""
        mock_isfile.return_value = False

        config_path = os.path.join(self.test_dir, "config.json")
        with open(config_path, "w") as f:
            f.write('{"_class_name": "LTXVideoTransformer3DModel"}')

        mock_hf_download.return_value = config_path

        self.assertEqual(detect_model_type("Lightricks/LTX-Video"), "ltx")
