"""Tests for the logging configuration module."""

import logging
import os
import tempfile
import unittest
from multiprocessing import Queue

from alloy.logging import (
    Verbosity,
    setup_logging,
    get_log_queue,
    setup_worker_logging,
    get_logger,
    shutdown_logging,
    parse_log_level,
    AlloyJSONFormatter,
    MarkupStripFormatter,
)


class TestVerbosity(unittest.TestCase):
    """Tests for Verbosity enum."""

    def test_verbosity_values(self):
        """Test that verbosity levels have correct values."""
        self.assertEqual(Verbosity.QUIET, 0)
        self.assertEqual(Verbosity.NORMAL, 1)
        self.assertEqual(Verbosity.VERBOSE, 2)
        self.assertEqual(Verbosity.DEBUG, 3)

    def test_verbosity_ordering(self):
        """Test that verbosity levels can be compared."""
        self.assertLess(Verbosity.QUIET, Verbosity.NORMAL)
        self.assertLess(Verbosity.NORMAL, Verbosity.VERBOSE)
        self.assertLess(Verbosity.VERBOSE, Verbosity.DEBUG)


class TestParseLogLevel(unittest.TestCase):
    """Tests for parse_log_level function."""

    def test_parse_debug(self):
        """Test parsing 'debug' level."""
        self.assertEqual(parse_log_level("debug"), Verbosity.DEBUG)

    def test_parse_verbose(self):
        """Test parsing 'verbose' level."""
        self.assertEqual(parse_log_level("verbose"), Verbosity.VERBOSE)

    def test_parse_info(self):
        """Test parsing 'info' level."""
        self.assertEqual(parse_log_level("info"), Verbosity.NORMAL)

    def test_parse_warning(self):
        """Test parsing 'warning' level."""
        self.assertEqual(parse_log_level("warning"), Verbosity.NORMAL)

    def test_parse_error(self):
        """Test parsing 'error' level."""
        self.assertEqual(parse_log_level("error"), Verbosity.QUIET)

    def test_parse_quiet(self):
        """Test parsing 'quiet' level."""
        self.assertEqual(parse_log_level("quiet"), Verbosity.QUIET)

    def test_parse_case_insensitive(self):
        """Test that parsing is case insensitive."""
        self.assertEqual(parse_log_level("DEBUG"), Verbosity.DEBUG)
        self.assertEqual(parse_log_level("Verbose"), Verbosity.VERBOSE)

    def test_parse_unknown_returns_normal(self):
        """Test that unknown values return NORMAL."""
        self.assertEqual(parse_log_level("unknown"), Verbosity.NORMAL)
        self.assertEqual(parse_log_level(""), Verbosity.NORMAL)


class TestGetLogger(unittest.TestCase):
    """Tests for get_logger function."""

    def test_get_logger_with_module_name(self):
        """Test getting logger with module name."""
        logger = get_logger("test_module")
        self.assertEqual(logger.name, "alloy.test_module")

    def test_get_logger_already_prefixed(self):
        """Test getting logger with already-prefixed name."""
        logger = get_logger("alloy.test_module")
        self.assertEqual(logger.name, "alloy.test_module")

    def test_get_logger_alloy_root(self):
        """Test getting the alloy root logger."""
        logger = get_logger("alloy")
        self.assertEqual(logger.name, "alloy")

    def test_get_logger_nested_alloy(self):
        """Test getting logger with nested alloy path."""
        logger = get_logger("src.alloy.converters.flux")
        self.assertEqual(logger.name, "alloy.converters.flux")


class TestAlloyJSONFormatter(unittest.TestCase):
    """Tests for AlloyJSONFormatter."""

    def setUp(self):
        self.formatter = AlloyJSONFormatter()

    def test_format_basic_record(self):
        """Test formatting a basic log record."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        result = self.formatter.format(record)

        import json
        parsed = json.loads(result)

        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["logger"], "test")
        self.assertEqual(parsed["message"], "Test message")
        self.assertIn("timestamp", parsed)

    def test_format_with_model_name(self):
        """Test formatting record with model_name extra."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.model_name = "Flux"
        result = self.formatter.format(record)

        import json
        parsed = json.loads(result)

        self.assertEqual(parsed["model_name"], "Flux")

    def test_format_with_phase(self):
        """Test formatting record with phase extra."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.phase = "Part 1"
        result = self.formatter.format(record)

        import json
        parsed = json.loads(result)

        self.assertEqual(parsed["phase"], "Part 1")


class TestMarkupStripFormatter(unittest.TestCase):
    """Tests for MarkupStripFormatter."""

    def setUp(self):
        self.formatter = MarkupStripFormatter("%(message)s")

    def test_strips_rich_markup(self):
        """Test that Rich markup tags are stripped."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="[cyan]Worker:[/cyan] Starting Part 1",
            args=(),
            exc_info=None
        )
        result = self.formatter.format(record)

        self.assertEqual(result, "Worker: Starting Part 1")

    def test_strips_bold_markup(self):
        """Test that bold markup is stripped."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="[bold]Important[/bold] message",
            args=(),
            exc_info=None
        )
        result = self.formatter.format(record)

        self.assertEqual(result, "Important message")

    def test_strips_dim_markup(self):
        """Test that dim markup is stripped."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="[dim]Debug info[/dim]",
            args=(),
            exc_info=None
        )
        result = self.formatter.format(record)

        self.assertEqual(result, "Debug info")


class TestSetupLogging(unittest.TestCase):
    """Tests for setup_logging function."""

    def setUp(self):
        # Reset logging state before each test
        shutdown_logging()

    def tearDown(self):
        # Cleanup after each test
        shutdown_logging()

    def test_setup_returns_alloy_logger(self):
        """Test that setup_logging returns the alloy logger."""
        logger = setup_logging(Verbosity.NORMAL)
        self.assertEqual(logger.name, "alloy")

    def test_quiet_verbosity_sets_error_level(self):
        """Test that QUIET verbosity sets ERROR level."""
        logger = setup_logging(Verbosity.QUIET)
        self.assertEqual(logger.level, logging.ERROR)

    def test_normal_verbosity_sets_info_level(self):
        """Test that NORMAL verbosity sets INFO level."""
        logger = setup_logging(Verbosity.NORMAL)
        self.assertEqual(logger.level, logging.INFO)

    def test_verbose_verbosity_sets_debug_level(self):
        """Test that VERBOSE verbosity sets DEBUG level."""
        logger = setup_logging(Verbosity.VERBOSE)
        self.assertEqual(logger.level, logging.DEBUG)

    def test_get_log_queue_returns_queue(self):
        """Test that get_log_queue returns a queue after setup."""
        setup_logging(Verbosity.NORMAL)
        queue = get_log_queue()
        self.assertIsNotNone(queue)
        # Check it has queue-like methods
        self.assertTrue(hasattr(queue, 'put'))
        self.assertTrue(hasattr(queue, 'get'))

    def test_file_logging_creates_file(self):
        """Test that file logging creates a log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            setup_logging(Verbosity.NORMAL, log_file=log_file)

            # Log something
            logger = logging.getLogger("alloy")
            logger.info("Test message")

            # File should exist (may not have content yet due to buffering)
            self.assertTrue(os.path.exists(log_file))


class TestSetupWorkerLogging(unittest.TestCase):
    """Tests for setup_worker_logging function."""

    def test_returns_alloy_logger(self):
        """Test that setup_worker_logging returns the alloy logger."""
        logger = setup_worker_logging()
        self.assertEqual(logger.name, "alloy")

    def test_with_queue(self):
        """Test setup_worker_logging with a queue."""
        queue = Queue()
        setup_worker_logging(queue)

        # Should have a handler
        root = logging.getLogger()
        self.assertGreater(len(root.handlers), 0)

    def test_without_queue_uses_stderr(self):
        """Test setup_worker_logging without queue uses stderr."""
        logger = setup_worker_logging(None)
        self.assertEqual(logger.name, "alloy")

        # Should have a handler
        root = logging.getLogger()
        self.assertGreater(len(root.handlers), 0)


class TestShutdownLogging(unittest.TestCase):
    """Tests for shutdown_logging function."""

    def test_clears_queue_after_setup(self):
        """Test that shutdown_logging clears the queue."""
        setup_logging(Verbosity.NORMAL)

        # Verify queue exists
        self.assertIsNotNone(get_log_queue())

        # Shutdown
        shutdown_logging()

        # Queue should be None
        self.assertIsNone(get_log_queue())

    def test_can_call_multiple_times(self):
        """Test that shutdown_logging can be called multiple times safely."""
        # Should not raise
        shutdown_logging()
        shutdown_logging()
        shutdown_logging()


if __name__ == "__main__":
    unittest.main()
