import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import tempfile
from alloy.converters.flux import FluxConverter

class TestFluxConverter(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.model_id = "black-forest-labs/FLUX.1-schnell"

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    @patch("alloy.converters.flux.multiprocessing.Process")
    @patch("alloy.converters.flux.ct.models.MLModel")
    @patch("alloy.converters.flux.ct.utils.make_pipeline")
    @patch("alloy.converters.flux.Console.print")
    def test_convert_success(self, mock_print, mock_make_pipeline, mock_mlmodel, mock_process):
        # Mock successful processes
        mock_p1 = MagicMock()
        mock_p1.exitcode = 0
        mock_p2 = MagicMock()
        mock_p2.exitcode = 0
        mock_process.side_effect = [mock_p1, mock_p2]
        
        # Mock pipeline model
        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline
        
        converter = FluxConverter(self.model_id, self.output_dir, "float16")
        converter.convert()
        
        self.assertTrue(mock_process.called)
        self.assertEqual(mock_process.call_count, 2)
        mock_pipeline.save.assert_called_with(os.path.join(self.output_dir, "Flux_Transformer_float16.mlpackage"))

    @patch("alloy.converters.flux.multiprocessing.Process")
    @patch("alloy.converters.flux.Console.print")
    def test_convert_part1_failure(self, mock_print, mock_process):
        # Mock failed part 1
        mock_p1 = MagicMock()
        mock_p1.exitcode = 1
        mock_process.return_value = mock_p1
        
        converter = FluxConverter(self.model_id, self.output_dir, "float16")
        
        with self.assertRaisesRegex(RuntimeError, "Part 1 Worker Failed"):
            converter.convert()

    @patch("alloy.converters.flux.os.path.exists")
    @patch("alloy.converters.flux.Console.print")
    def test_convert_skips_existing(self, mock_print, mock_exists):
        mock_exists.return_value = True # Pretend output exists
        
        converter = FluxConverter(self.model_id, self.output_dir, "float16")
        converter.convert()
        
        # Verify it printed "Model exists" and returned
        # How to check return? Just check if it didn't do work
        # We can check if Console.print was called with specific text
        args, _ = mock_print.call_args
        self.assertIn("Model exists", str(args))

    @patch("alloy.converters.flux.multiprocessing.Process")
    @patch("alloy.converters.flux.ct.models.MLModel")
    @patch("alloy.converters.flux.ct.utils.make_pipeline")
    @patch("alloy.converters.flux.Console.print")
    def test_resume_intermediates(self, mock_print, mock_make_pipeline, mock_mlmodel, mock_process):
        # Create dummy intermediates
        intermediaries_dir = os.path.join(self.output_dir, "intermediates")
        os.makedirs(intermediaries_dir)
        part1 = os.path.join(intermediaries_dir, "FluxPart1.mlpackage")
        part2 = os.path.join(intermediaries_dir, "FluxPart2.mlpackage")
        os.makedirs(part1)
        os.makedirs(part2)
        
        # Setup mocks to succeed validation
        mock_mlmodel.return_value = MagicMock() # Valid model
        
        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline
        
        converter = FluxConverter(self.model_id, self.output_dir, "float16")
        converter.convert()
        
        # Process should NOT be called if resumption works
        self.assertFalse(mock_process.called)
        mock_pipeline.save.assert_called()
