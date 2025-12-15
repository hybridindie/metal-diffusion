import subprocess
import pytest
import shutil

@pytest.mark.smoke
def test_cli_help():
    """Verify that the CLI help command works without crashing."""
    result = subprocess.run(["uv", "run", "alloy", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Diffusion to Core ML Converter" in result.stdout

@pytest.mark.smoke
def test_cli_version():
    """Verify that the package is installed and importable."""
    # Run a simple python script to verify import
    script = "import alloy; print('Successfully imported alloy')"
    result = subprocess.run(["uv", "run", "python", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Successfully imported alloy" in result.stdout
