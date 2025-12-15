import pytest
import argparse
from alloy.cli import main
from unittest.mock import patch, MagicMock

def test_download_args():
    with patch("argparse.ArgumentParser.parse_args",
               return_value=argparse.Namespace(
                   command="download",
                   repo_id="stabilityai/sd-turbo",
                   output_dir="models"
               )):
        with patch("alloy.cli.HFManager") as MockHF:
            main()
            MockHF.return_value.download_model.assert_called_with(
                "stabilityai/sd-turbo", 
                local_dir="models/sd-turbo"
            )

def test_convert_sd_args():
    with patch("argparse.ArgumentParser.parse_args",
               return_value=argparse.Namespace(
                   command="convert",
                   model_id="models/sd-turbo",
                   type="sd",
                   output_dir="converted_models",
                   quantization="float16"
               )):
        with patch("alloy.cli.SDConverter") as MockConverter:
            main()
            MockConverter.assert_called_with("models/sd-turbo", "converted_models", "float16")
            MockConverter.return_value.convert.assert_called_once()

def test_convert_wan_args():
    with patch("argparse.ArgumentParser.parse_args",
               return_value=argparse.Namespace(
                   command="convert",
                   model_id="models/wan-1.3b",
                   type="wan",
                   output_dir="converted_models",
                   quantization="int4"
               )):
        with patch("alloy.cli.WanConverter") as MockConverter:
            main()
            MockConverter.assert_called_with("models/wan-1.3b", "converted_models", "int4")
            MockConverter.return_value.convert.assert_called_once()
