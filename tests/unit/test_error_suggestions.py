"""Tests for error suggestion helpers."""

import unittest

from alloy.utils.errors import (
    get_worker_suggestions,
    get_download_suggestions,
    get_config_suggestions,
    get_dependency_suggestions,
)
from alloy.exceptions import WorkerError, HuggingFaceError


class TestGetWorkerSuggestions(unittest.TestCase):
    """Tests for get_worker_suggestions function."""

    def test_sigkill_suggests_oom_solutions(self):
        """Test that SIGKILL (-9) suggests OOM-related solutions."""
        suggestions = get_worker_suggestions(-9, "Part 1")

        self.assertIn("Close other applications to free memory", suggestions)
        self.assertTrue(any("quantization" in s.lower() for s in suggestions))
        self.assertTrue(any("ram" in s.lower() for s in suggestions))

    def test_sigterm_notes_external_termination(self):
        """Test that SIGTERM (-15) notes external termination."""
        suggestions = get_worker_suggestions(-15, "Part 2")

        self.assertTrue(any("terminated externally" in s.lower() for s in suggestions))

    def test_exit_code_1_suggests_checking_output(self):
        """Test that exit code 1 suggests checking error output."""
        suggestions = get_worker_suggestions(1, "Part 1")

        self.assertTrue(any("check" in s.lower() for s in suggestions))

    def test_other_negative_exit_codes_mention_signal(self):
        """Test that other negative exit codes mention the signal number."""
        suggestions = get_worker_suggestions(-6, "Part 1")  # SIGABRT

        self.assertTrue(any("signal 6" in s for s in suggestions))

    def test_exit_code_0_returns_empty(self):
        """Test that exit code 0 (success) returns no suggestions."""
        suggestions = get_worker_suggestions(0, "Part 1")

        self.assertEqual(suggestions, [])

    def test_none_exit_code_returns_empty(self):
        """Test that None exit code returns no suggestions."""
        suggestions = get_worker_suggestions(None, "Part 1")

        self.assertEqual(suggestions, [])


class TestGetDownloadSuggestions(unittest.TestCase):
    """Tests for get_download_suggestions function."""

    def test_includes_repo_id_when_provided(self):
        """Test that repo_id is included in suggestions."""
        suggestions = get_download_suggestions("org/model-name")

        self.assertTrue(any("org/model-name" in s for s in suggestions))

    def test_suggests_checking_connection(self):
        """Test that it suggests checking internet connection."""
        suggestions = get_download_suggestions("org/model")

        self.assertTrue(any("internet" in s.lower() or "connection" in s.lower() for s in suggestions))

    def test_suggests_hf_token_for_private(self):
        """Test that it suggests HF_TOKEN for private models."""
        suggestions = get_download_suggestions("org/model")

        self.assertTrue(any("hf_token" in s.lower() or "private" in s.lower() for s in suggestions))

    def test_suggests_status_page(self):
        """Test that it suggests checking HF status page."""
        suggestions = get_download_suggestions("org/model")

        self.assertTrue(any("status.huggingface.co" in s for s in suggestions))

    def test_no_repo_id_still_returns_suggestions(self):
        """Test that it returns suggestions even without repo_id."""
        suggestions = get_download_suggestions(None)

        self.assertGreater(len(suggestions), 0)


class TestGetConfigSuggestions(unittest.TestCase):
    """Tests for get_config_suggestions function."""

    def test_includes_missing_fields(self):
        """Test that missing fields are included in suggestions."""
        suggestions = get_config_suggestions(missing_fields=["model", "type"])

        self.assertTrue(any("model" in s and "type" in s for s in suggestions))

    def test_suggests_format_check(self):
        """Test that it suggests checking file format."""
        suggestions = get_config_suggestions()

        self.assertTrue(any("json" in s.lower() or "yaml" in s.lower() for s in suggestions))

    def test_mentions_required_fields(self):
        """Test that it mentions required fields."""
        suggestions = get_config_suggestions()

        self.assertTrue(any("model" in s.lower() and "type" in s.lower() for s in suggestions))


class TestGetDependencySuggestions(unittest.TestCase):
    """Tests for get_dependency_suggestions function."""

    def test_includes_install_command_with_package(self):
        """Test that install command includes package name."""
        suggestions = get_dependency_suggestions(package_name="diffusers")

        self.assertTrue(any("pip install diffusers" in s for s in suggestions))

    def test_suggests_environment_check(self):
        """Test that it suggests checking Python environment."""
        suggestions = get_dependency_suggestions()

        self.assertTrue(any("environment" in s.lower() for s in suggestions))


class TestExceptionSuggestions(unittest.TestCase):
    """Tests for exception classes with suggestions."""

    def test_worker_error_stores_suggestions(self):
        """Test that WorkerError stores suggestions."""
        suggestions = ["Try this", "Or that"]
        error = WorkerError(
            "Test error",
            model_name="Test",
            phase="Part 1",
            exit_code=1,
            suggestions=suggestions,
        )

        self.assertEqual(error.suggestions, suggestions)

    def test_worker_error_defaults_to_empty_suggestions(self):
        """Test that WorkerError defaults to empty suggestions list."""
        error = WorkerError("Test error")

        self.assertEqual(error.suggestions, [])

    def test_huggingface_error_stores_suggestions(self):
        """Test that HuggingFaceError stores suggestions."""
        suggestions = ["Check connection", "Verify repo"]
        error = HuggingFaceError(
            "Download failed",
            repo_id="org/model",
            suggestions=suggestions,
        )

        self.assertEqual(error.suggestions, suggestions)

    def test_huggingface_error_defaults_to_empty_suggestions(self):
        """Test that HuggingFaceError defaults to empty suggestions list."""
        error = HuggingFaceError("Download failed")

        self.assertEqual(error.suggestions, [])


if __name__ == "__main__":
    unittest.main()
