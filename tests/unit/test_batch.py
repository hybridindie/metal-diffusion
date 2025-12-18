"""Tests for batch conversion utilities."""
import unittest
import tempfile
import os
from pathlib import Path

from alloy.utils.batch import parse_batch_file, validate_batch_config
from alloy.exceptions import ConfigError, UnsupportedModelError


class TestParseBatchFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_parse_json_file(self):
        """Test parsing a valid JSON batch file."""
        batch_file = os.path.join(self.temp_dir, "batch.json")
        with open(batch_file, "w") as f:
            f.write('[{"model": "test/model", "type": "flux"}]')

        result = parse_batch_file(batch_file)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model"], "test/model")

    def test_parse_yaml_file(self):
        """Test parsing a valid YAML batch file."""
        batch_file = os.path.join(self.temp_dir, "batch.yaml")
        with open(batch_file, "w") as f:
            f.write("- model: test/model\n  type: flux\n")

        result = parse_batch_file(batch_file)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model"], "test/model")

    def test_file_not_found_raises_config_error(self):
        """Test that missing file raises ConfigError."""
        with self.assertRaises(ConfigError) as ctx:
            parse_batch_file("/nonexistent/batch.json")
        self.assertIn("not found", str(ctx.exception))

    def test_invalid_json_raises_config_error(self):
        """Test that invalid JSON raises ConfigError."""
        batch_file = os.path.join(self.temp_dir, "batch.json")
        with open(batch_file, "w") as f:
            f.write("{invalid json}")

        with self.assertRaises(ConfigError) as ctx:
            parse_batch_file(batch_file)
        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_invalid_yaml_raises_config_error(self):
        """Test that invalid YAML raises ConfigError."""
        batch_file = os.path.join(self.temp_dir, "batch.yaml")
        with open(batch_file, "w") as f:
            f.write("invalid: yaml: content: [")

        with self.assertRaises(ConfigError) as ctx:
            parse_batch_file(batch_file)
        self.assertIn("Invalid YAML", str(ctx.exception))

    def test_non_list_raises_config_error(self):
        """Test that non-list content raises ConfigError."""
        batch_file = os.path.join(self.temp_dir, "batch.json")
        with open(batch_file, "w") as f:
            f.write('{"model": "test"}')

        with self.assertRaises(ConfigError) as ctx:
            parse_batch_file(batch_file)
        self.assertIn("must contain a list", str(ctx.exception))


class TestValidateBatchConfig(unittest.TestCase):
    def test_valid_config(self):
        """Test validation of a valid config."""
        config = {"model": "test/model", "type": "flux"}
        result = validate_batch_config(config)
        self.assertEqual(result["model"], "test/model")
        self.assertEqual(result["type"], "flux")

    def test_missing_model_raises_config_error(self):
        """Test that missing 'model' field raises ConfigError."""
        config = {"type": "flux"}

        with self.assertRaises(ConfigError) as ctx:
            validate_batch_config(config)
        self.assertIn("model", ctx.exception.missing_fields)

    def test_missing_type_raises_config_error(self):
        """Test that missing 'type' field raises ConfigError."""
        config = {"model": "test/model"}

        with self.assertRaises(ConfigError) as ctx:
            validate_batch_config(config)
        self.assertIn("type", ctx.exception.missing_fields)

    def test_missing_both_fields_raises_config_error(self):
        """Test that missing both fields raises ConfigError with both in missing_fields."""
        config = {}

        with self.assertRaises(ConfigError) as ctx:
            validate_batch_config(config)
        self.assertIn("model", ctx.exception.missing_fields)
        self.assertIn("type", ctx.exception.missing_fields)

    def test_defaults_applied(self):
        """Test that defaults are applied for optional fields."""
        config = {"model": "org/model-name", "type": "flux"}
        result = validate_batch_config(config)

        self.assertEqual(result["quantization"], "float16")
        self.assertIn("model-name", result["output_dir"])


if __name__ == "__main__":
    unittest.main()
