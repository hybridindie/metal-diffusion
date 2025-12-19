import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import tempfile
from alloy.converters.flux import FluxConverter
from alloy.exceptions import WorkerError

class TestFluxConverter(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp()
        self.model_id = "black-forest-labs/FLUX.1-schnell"

    def tearDown(self):
        shutil.rmtree(self.output_dir)

    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.ct.utils.make_pipeline")
    @patch("alloy.converters.base.logger")
    def test_convert_success(self, mock_logger, mock_make_pipeline, mock_mlmodel, mock_process):
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

    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.logger")
    def test_convert_part1_failure(self, mock_logger, mock_process):
        # Mock failed part 1
        mock_p1 = MagicMock()
        mock_p1.exitcode = 1
        mock_process.return_value = mock_p1

        converter = FluxConverter(self.model_id, self.output_dir, "float16")

        with self.assertRaises(WorkerError) as ctx:
            converter.convert()
        self.assertEqual(ctx.exception.model_name, "Flux")
        self.assertEqual(ctx.exception.exit_code, 1)

    @patch("alloy.converters.base.os.path.exists")
    @patch("alloy.converters.base.logger")
    def test_convert_skips_existing(self, mock_logger, mock_exists):
        mock_exists.return_value = True  # Pretend output exists

        converter = FluxConverter(self.model_id, self.output_dir, "float16")
        converter.convert()

        # Verify it logged "Model exists" warning
        calls = [str(c) for c in mock_logger.warning.call_args_list]
        self.assertTrue(any("Model exists" in c for c in calls))

    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.ct.utils.make_pipeline")
    @patch("alloy.converters.base.logger")
    def test_resume_intermediates(self, mock_logger, mock_make_pipeline, mock_mlmodel, mock_process):
        # Create dummy intermediates
        intermediaries_dir = os.path.join(self.output_dir, "intermediates")
        os.makedirs(intermediaries_dir)
        part1 = os.path.join(intermediaries_dir, "FluxPart1.mlpackage")
        part2 = os.path.join(intermediaries_dir, "FluxPart2.mlpackage")
        os.makedirs(part1)
        os.makedirs(part2)

        # Setup mocks to succeed validation
        mock_mlmodel.return_value = MagicMock()  # Valid model

        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline

        converter = FluxConverter(self.model_id, self.output_dir, "float16")
        converter.convert()

        # Process should NOT be called if resumption works
        self.assertFalse(mock_process.called)
        mock_pipeline.save.assert_called()

    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.ct.utils.make_pipeline")
    @patch("alloy.converters.base.shutil.rmtree")
    @patch("alloy.converters.base.logger")
    def test_corrupt_intermediate_reconverts(
        self, mock_logger, mock_rmtree, mock_make_pipeline, mock_mlmodel, mock_process
    ):
        """Test that corrupt intermediate triggers re-conversion."""
        # Create dummy intermediate that will fail validation
        intermediates_dir = os.path.join(self.output_dir, "intermediates")
        os.makedirs(intermediates_dir)
        part1 = os.path.join(intermediates_dir, "FluxPart1.mlpackage")
        os.makedirs(part1)

        # Mock MLModel to fail validation for part1 (corrupt), then succeed for assembly
        # Sequence: Part 1 validation (fails) -> Assembly part1 -> Assembly part2
        # Part 2 doesn't exist initially, so no validation call for it
        mock_mlmodel.side_effect = [
            Exception("Invalid MLModel"),  # Part 1 validation fails
            MagicMock(),  # Assembly loads part1
            MagicMock(),  # Assembly loads part2
        ]

        # Mock successful process after reconversion
        mock_p = MagicMock()
        mock_p.exitcode = 0
        mock_process.return_value = mock_p

        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline

        converter = FluxConverter(self.model_id, self.output_dir, "float16")
        converter.convert()

        # rmtree should be called to clean up corrupt intermediate
        self.assertTrue(mock_rmtree.called)
        # Process should be called for reconversion
        self.assertTrue(mock_process.called)

    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.logger")
    def test_worker_oom_exitcode(self, mock_logger, mock_process):
        """Test that OOM exit code (-9) raises WorkerError with memory suggestions."""
        mock_p = MagicMock()
        mock_p.exitcode = -9  # SIGKILL (typically OOM)
        mock_process.return_value = mock_p

        converter = FluxConverter(self.model_id, self.output_dir, "float16")

        with self.assertRaises(WorkerError) as ctx:
            converter.convert()

        self.assertEqual(ctx.exception.exit_code, -9)
        # Suggestions should contain memory-related advice
        self.assertTrue(
            any("memory" in s.lower() for s in ctx.exception.suggestions),
            f"Expected memory suggestion in {ctx.exception.suggestions}"
        )

    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.logger")
    def test_worker_segfault_exitcode(self, mock_logger, mock_process):
        """Test that segfault exit code (-11) raises WorkerError."""
        mock_p = MagicMock()
        mock_p.exitcode = -11  # SIGSEGV
        mock_process.return_value = mock_p

        converter = FluxConverter(self.model_id, self.output_dir, "float16")

        with self.assertRaises(WorkerError) as ctx:
            converter.convert()

        self.assertEqual(ctx.exception.exit_code, -11)
        self.assertIsNotNone(ctx.exception.suggestions)

    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.logger")
    def test_part2_failure_after_part1_success(self, mock_logger, mock_mlmodel, mock_process):
        """Test that Part 2 failure after Part 1 success raises correct error."""
        # Create part 1 intermediate that passes validation
        intermediates_dir = os.path.join(self.output_dir, "intermediates")
        os.makedirs(intermediates_dir)
        part1 = os.path.join(intermediates_dir, "FluxPart1.mlpackage")
        os.makedirs(part1)

        # Mock MLModel to succeed for part1 validation
        mock_mlmodel.return_value = MagicMock()

        # Mock part2 process to fail
        mock_p2 = MagicMock()
        mock_p2.exitcode = 1
        mock_process.return_value = mock_p2

        converter = FluxConverter(self.model_id, self.output_dir, "float16")

        with self.assertRaises(WorkerError) as ctx:
            converter.convert()

        # Error should indicate Part 2 failure
        self.assertEqual(ctx.exception.phase, "Part 2")
        self.assertEqual(ctx.exception.model_name, "Flux")

    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.ct.utils.make_pipeline")
    @patch("alloy.converters.base.logger")
    def test_int4_quantization_passed_to_worker(
        self, mock_logger, mock_make_pipeline, mock_mlmodel, mock_process
    ):
        """Test that int4 quantization is passed to worker correctly."""
        mock_p = MagicMock()
        mock_p.exitcode = 0
        mock_process.return_value = mock_p

        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline

        converter = FluxConverter(self.model_id, self.output_dir, "int4")
        converter.convert()

        # Check that Process was called with int4 quantization
        for call in mock_process.call_args_list:
            args = call[1]["args"]  # (model_id, output_path, quantization)
            self.assertEqual(args[2], "int4")

    @patch("alloy.converters.base.multiprocessing.Process")
    @patch("alloy.converters.base.ct.models.MLModel")
    @patch("alloy.converters.base.ct.utils.make_pipeline")
    @patch("alloy.converters.base.logger")
    def test_int8_quantization_passed_to_worker(
        self, mock_logger, mock_make_pipeline, mock_mlmodel, mock_process
    ):
        """Test that int8 quantization is passed to worker correctly."""
        mock_p = MagicMock()
        mock_p.exitcode = 0
        mock_process.return_value = mock_p

        mock_pipeline = MagicMock()
        mock_make_pipeline.return_value = mock_pipeline

        converter = FluxConverter(self.model_id, self.output_dir, "int8")
        converter.convert()

        # Check that Process was called with int8 quantization
        for call in mock_process.call_args_list:
            args = call[1]["args"]
            self.assertEqual(args[2], "int8")
