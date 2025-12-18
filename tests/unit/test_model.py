import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from alloy.utils.model import validate_model, show_model_info, list_models, detect_safetensors_precision

class TestModelUtils(unittest.TestCase):
    
    @patch("alloy.utils.model.os.path.exists")
    @patch("alloy.utils.model.ct.models.MLModel")
    @patch("alloy.utils.model.console.print")
    def test_validate_model_success(self, mock_print, mock_mlmodel, mock_exists):
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.get_spec.return_value.WhichOneof.return_value = "neuralNetwork"
        # Setup description inputs/outputs
        desc = MagicMock()
        desc.input = [1, 2] # Dummy list with len()
        desc.output = [1]
        mock_model.get_spec.return_value.description = desc
        mock_mlmodel.return_value = mock_model
        
        result = validate_model("test_model.mlpackage")
        self.assertTrue(result)
        mock_mlmodel.assert_called_with("test_model.mlpackage")

    @patch("alloy.utils.model.os.path.exists")
    def test_validate_model_not_found(self, mock_exists):
        mock_exists.return_value = False
        result = validate_model("missing.mlpackage")
        self.assertFalse(result)

    @patch("alloy.utils.model.ct.models.MLModel")
    @patch("alloy.utils.model.console.print")
    def test_show_model_info(self, mock_print, mock_mlmodel):
        mock_model = MagicMock()
        mock_spec = MagicMock()
        mock_spec.WhichOneof.return_value = "neuralNetwork"
        
        # Inputs
        inp1 = MagicMock()
        inp1.name = "input1"
        inp1.type.WhichOneof.return_value = "multiArrayType"
        inputs = [inp1]
        
        # Outputs
        out1 = MagicMock()
        out1.name = "output1"
        out1.type.WhichOneof.return_value = "imageType"
        outputs = [out1]
        
        mock_spec.description.input = inputs
        mock_spec.description.output = outputs
        
        mock_model.get_spec.return_value = mock_spec
        mock_mlmodel.return_value = mock_model
        
        # Mock file size calculation? 
        # The function uses Path(model_path).rglob('*')
        # We can patch Path but it is tricky. 
        # Let's just create a real dummy temp dir or patch Path?
        # Patching Path is cleaner if possible, or just let it fail/warn?
        # Actually validation wraps in try/except.
        
        with patch("alloy.utils.model.Path") as mock_path:
             mock_path.return_value.rglob.return_value = []
             mock_path.return_value.name = "test.mlpackage"
             show_model_info("test.mlpackage")
             
        # Verify it printed a Table
        self.assertTrue(mock_print.called)

    @patch("alloy.utils.model.Path")
    @patch("alloy.utils.model.console.print")
    def test_list_models(self, mock_print, mock_path):
        mock_path.return_value.exists.return_value = True
        
        # Setup found models
        m1 = MagicMock()
        m1.stat.return_value.st_size = 1000
        m1.stat.return_value.st_mtime = 1234567890
        m1.relative_to.return_value = "model1.mlpackage"
        m1.rglob.return_value = [m1] # To calculate size
        
        mock_path.return_value.rglob.return_value = [m1]
        
        list_models("dummy_dir")
        self.assertTrue(mock_print.called)

    @patch("safetensors.safe_open")
    def test_detect_safetensors_precision(self, mock_safe_open):
        mock_f = MagicMock()
        mock_f.keys.return_value = ["model.weight"]
        
        mock_tensor = MagicMock()
        mock_tensor.dtype = "torch.float8_e4m3fn"
        mock_f.get_tensor.return_value = mock_tensor
        
        mock_safe_open.return_value.__enter__.return_value = mock_f
        
        precision = detect_safetensors_precision("model.safetensors")
        self.assertEqual(precision, "int8")
