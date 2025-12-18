import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import tempfile
from alloy.converters.wan import WanConverter
from alloy.exceptions import WorkerError, UnsupportedModelError


class TestWanConverter(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.model_id = "Wan-AI/Wan2.1-T2V-1.3B"

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    @patch.object(WanConverter, 'download_source_weights', return_value="/mocked/path")
    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.ct.utils.make_pipeline")
    @patch("alloy.converters.base.console.print")
    def test_convert_success(self, mock_print, mock_make_pipeline, mock_mlmodel, mock_process, mock_download):
        """Test successful 2-phase conversion."""
        # Mock successful processes
        mock_p1 = MagicMock()
        mock_p1.exitcode = 0
        mock_p2 = MagicMock()
        mock_p2.exitcode = 0
        mock_process.side_effect = [mock_p1, mock_p2]

        # Mock pipeline model
        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline

        converter = WanConverter(self.model_id, self.output_dir, "float16")
        converter.convert()

        # Verify 2 processes were spawned
        self.assertEqual(mock_process.call_count, 2)
        mock_pipeline.save.assert_called()

    @patch.object(WanConverter, 'download_source_weights', return_value="/mocked/path")
    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.console.print")
    def test_convert_part1_failure(self, mock_print, mock_process, mock_download):
        """Test error handling when Part 1 fails."""
        mock_p1 = MagicMock()
        mock_p1.exitcode = 1
        mock_process.return_value = mock_p1

        converter = WanConverter(self.model_id, self.output_dir, "float16")

        with self.assertRaises(WorkerError) as ctx:
            converter.convert()
        self.assertEqual(ctx.exception.model_name, "Wan")
        self.assertEqual(ctx.exception.exit_code, 1)

    @patch.object(WanConverter, 'download_source_weights', return_value="/mocked/path")
    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.console.print")
    def test_convert_part2_failure(self, mock_print, mock_mlmodel, mock_process, mock_download):
        """Test error handling when Part 2 fails."""
        mock_p1 = MagicMock()
        mock_p1.exitcode = 0
        mock_p2 = MagicMock()
        mock_p2.exitcode = 1
        mock_process.side_effect = [mock_p1, mock_p2]

        converter = WanConverter(self.model_id, self.output_dir, "float16")

        with self.assertRaises(WorkerError) as ctx:
            converter.convert()
        self.assertEqual(ctx.exception.model_name, "Wan")
        self.assertEqual(ctx.exception.exit_code, 1)

    @patch("alloy.converters.base.os.path.exists")
    @patch("alloy.converters.base.console.print")
    def test_convert_skips_existing(self, mock_print, mock_exists):
        """Test that conversion is skipped if final model already exists."""
        def side_effect(path):
            if "Wan2.1_Transformer_float16.mlpackage" in path and "intermediates" not in path:
                return True
            return False

        mock_exists.side_effect = side_effect

        converter = WanConverter(self.model_id, self.output_dir, "float16")
        converter.convert()

        # Should skip without error
        skip_logged = any("skipping" in str(call).lower() for call in mock_print.call_args_list)
        self.assertTrue(skip_logged)

    @patch.object(WanConverter, 'download_source_weights', return_value="/mocked/path")
    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.ct.utils.make_pipeline")
    @patch("alloy.converters.base.console.print")
    def test_resume_from_part1(self, mock_print, mock_make_pipeline, mock_mlmodel, mock_process, mock_download):
        """Test resuming from existing Part 1 intermediate."""
        # Create dummy Part 1 intermediate
        intermediates_dir = os.path.join(self.output_dir, "intermediates")
        os.makedirs(intermediates_dir)
        part1_path = os.path.join(intermediates_dir, "WanPart1.mlpackage")
        os.makedirs(part1_path)

        # Mock valid MLModel load
        mock_mlmodel.return_value = MagicMock()

        # Mock Part 2 process only (Part 1 should be skipped)
        mock_p2 = MagicMock()
        mock_p2.exitcode = 0
        mock_process.return_value = mock_p2

        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline

        converter = WanConverter(self.model_id, self.output_dir, "float16")
        converter.convert()

        # Only 1 process should be spawned (Part 2)
        self.assertEqual(mock_process.call_count, 1)

    @patch.object(WanConverter, 'download_source_weights', return_value="/mocked/path")
    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.ct.utils.make_pipeline")
    @patch("alloy.converters.base.console.print")
    def test_resume_both_parts(self, mock_print, mock_make_pipeline, mock_mlmodel, mock_process, mock_download):
        """Test resuming when both parts exist."""
        # Create dummy intermediates
        intermediates_dir = os.path.join(self.output_dir, "intermediates")
        os.makedirs(intermediates_dir)
        part1_path = os.path.join(intermediates_dir, "WanPart1.mlpackage")
        part2_path = os.path.join(intermediates_dir, "WanPart2.mlpackage")
        os.makedirs(part1_path)
        os.makedirs(part2_path)

        # Mock valid MLModel load
        mock_mlmodel.return_value = MagicMock()

        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline

        converter = WanConverter(self.model_id, self.output_dir, "float16")
        converter.convert()

        # No processes should be spawned (both parts exist)
        self.assertFalse(mock_process.called)
        mock_pipeline.save.assert_called()

    @patch("alloy.converters.wan.os.path.isfile")
    def test_convert_fails_single_file(self, mock_isfile):
        """Test that single file input raises UnsupportedModelError."""
        mock_isfile.return_value = True

        converter = WanConverter("local.safetensors", self.output_dir, "float16")

        with self.assertRaises(UnsupportedModelError) as ctx:
            converter.convert()
        self.assertEqual(ctx.exception.model_name, "Wan")
        self.assertEqual(ctx.exception.model_type, "single_file")


if __name__ == "__main__":
    unittest.main()
