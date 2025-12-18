import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import tempfile
from alloy.converters.ltx import LTXConverter
from alloy.exceptions import WorkerError


class TestLTXConverter(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.model_id = "Lightricks/LTX-Video"

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    @patch.object(LTXConverter, 'download_source_weights', return_value="/mocked/path")
    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.ct.utils.make_pipeline")
    @patch("alloy.converters.base.console.print")
    def test_convert_success(self, mock_print, mock_make_pipeline, mock_mlmodel, mock_process, mock_download):
        """Test successful 2-phase conversion."""
        mock_p1 = MagicMock()
        mock_p1.exitcode = 0
        mock_p2 = MagicMock()
        mock_p2.exitcode = 0
        mock_process.side_effect = [mock_p1, mock_p2]

        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline

        converter = LTXConverter(self.model_id, self.output_dir, "float16")
        converter.convert()

        self.assertEqual(mock_process.call_count, 2)
        mock_pipeline.save.assert_called()

    @patch.object(LTXConverter, 'download_source_weights', return_value="/mocked/path")
    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.console.print")
    def test_convert_part1_failure(self, mock_print, mock_process, mock_download):
        """Test error handling when Part 1 fails."""
        mock_p1 = MagicMock()
        mock_p1.exitcode = 1
        mock_process.return_value = mock_p1

        converter = LTXConverter(self.model_id, self.output_dir, "float16")

        with self.assertRaises(WorkerError) as ctx:
            converter.convert()
        self.assertEqual(ctx.exception.model_name, "LTX")
        self.assertEqual(ctx.exception.exit_code, 1)

    @patch.object(LTXConverter, 'download_source_weights', return_value="/mocked/path")
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

        converter = LTXConverter(self.model_id, self.output_dir, "float16")

        with self.assertRaises(WorkerError) as ctx:
            converter.convert()
        self.assertEqual(ctx.exception.model_name, "LTX")
        self.assertEqual(ctx.exception.exit_code, 1)

    @patch("alloy.converters.base.os.path.exists")
    @patch("alloy.converters.base.console.print")
    def test_convert_skips_existing(self, mock_print, mock_exists):
        """Test that conversion is skipped if final model already exists."""
        def side_effect(path):
            if "LTXVideo_Transformer_float16.mlpackage" in path and "intermediates" not in path:
                return True
            return False

        mock_exists.side_effect = side_effect

        converter = LTXConverter(self.model_id, self.output_dir, "float16")
        converter.convert()

        skip_logged = any("skipping" in str(call).lower() for call in mock_print.call_args_list)
        self.assertTrue(skip_logged)

    @patch.object(LTXConverter, 'download_source_weights', return_value="/mocked/path")
    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.ct.utils.make_pipeline")
    @patch("alloy.converters.base.console.print")
    def test_resume_both_parts(self, mock_print, mock_make_pipeline, mock_mlmodel, mock_process, mock_download):
        """Test resuming when both parts exist."""
        intermediates_dir = os.path.join(self.output_dir, "intermediates")
        os.makedirs(intermediates_dir)
        part1_path = os.path.join(intermediates_dir, "LTXPart1.mlpackage")
        part2_path = os.path.join(intermediates_dir, "LTXPart2.mlpackage")
        os.makedirs(part1_path)
        os.makedirs(part2_path)

        mock_mlmodel.return_value = MagicMock()

        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline

        converter = LTXConverter(self.model_id, self.output_dir, "float16")
        converter.convert()

        self.assertFalse(mock_process.called)
        mock_pipeline.save.assert_called()

    def test_init_default_model_id(self):
        """Test that default model ID is set for short names."""
        converter = LTXConverter("ltx", self.output_dir, "float16")
        self.assertEqual(converter.model_id, "Lightricks/LTX-Video")


if __name__ == "__main__":
    unittest.main()
