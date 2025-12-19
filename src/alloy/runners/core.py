"""Base class for CoreML hybrid runners."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import coremltools as ct
import numpy as np
import os
import torch

from alloy.logging import get_logger

logger = get_logger(__name__)


class BaseCoreMLRunner(ABC):
    """
    Abstract base class for CoreML hybrid runners.

    Provides:
    - Device detection (MPS/CPU)
    - Pipeline loading with single-file support
    - CoreML model loading
    - Tensor conversion utilities

    Subclasses must implement:
    - model_name: Human-readable model name
    - transformer_filename: CoreML package filename
    - pipeline_class: diffusers Pipeline class
    - default_model_id: Default HuggingFace model ID
    - _load_pipeline(): Custom pipeline loading
    - _load_coreml_models(): Load CoreML model(s)
    - generate(): Main generation method
    """

    def __init__(
        self,
        model_dir: str,
        model_id: Optional[str] = None,
        compute_unit: str = "ALL",
    ):
        """
        Initialize the runner.

        Args:
            model_dir: Directory containing CoreML model packages
            model_id: HuggingFace model ID or local path (defaults to default_model_id)
            compute_unit: CoreML compute unit ("ALL", "CPU_AND_GPU", "CPU_ONLY")
        """
        self.model_dir = model_dir
        self.model_id = model_id or self.default_model_id
        self.compute_unit = compute_unit
        self.device = self._detect_device()

        self._load_pipeline()
        self._load_coreml_models()

    # --- Abstract Properties (must override) ---

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name (e.g., 'Flux', 'LTX-Video')."""
        pass

    @property
    @abstractmethod
    def transformer_filename(self) -> str:
        """CoreML transformer package filename."""
        pass

    @property
    @abstractmethod
    def pipeline_class(self):
        """Return the diffusers Pipeline class to use."""
        pass

    @property
    @abstractmethod
    def default_model_id(self) -> str:
        """Default HuggingFace model ID."""
        pass

    # --- Optional Properties (can override) ---

    @property
    def supports_single_file(self) -> bool:
        """Whether the pipeline supports from_single_file loading."""
        return True

    @property
    def default_dtype(self) -> torch.dtype:
        """Default torch dtype for pipeline."""
        return torch.float16

    @property
    def output_key(self) -> str:
        """Key for noise prediction in CoreML output dict."""
        return "sample"

    # --- Abstract Methods (must override) ---

    @abstractmethod
    def _load_pipeline(self) -> None:
        """
        Load the diffusers pipeline with PyTorch components.

        Subclasses should implement model-specific loading logic and
        set self.pipe to the loaded pipeline.
        """
        pass

    @abstractmethod
    def _load_coreml_models(self) -> None:
        """
        Load CoreML model(s).

        Subclasses should implement loading of transformer and any
        additional CoreML models (VAE, text encoder, etc.).
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, output_path: str, **kwargs) -> None:
        """
        Main generation method.

        Args:
            prompt: Text prompt for generation
            output_path: Path to save output
            **kwargs: Model-specific generation parameters
        """
        pass

    # --- Shared Implementation ---

    @staticmethod
    def _detect_device() -> str:
        """Detect available compute device."""
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def _load_pipeline_with_fallback(
        self,
        exclude_transformer: bool = True,
        **kwargs,
    ) -> Any:
        """
        Load pipeline with single-file fallback support.

        Args:
            exclude_transformer: Whether to exclude transformer (using CoreML)
            **kwargs: Additional kwargs for pipeline loading

        Returns:
            Loaded pipeline
        """
        load_kwargs = {
            "torch_dtype": self.default_dtype,
            **kwargs,
        }

        if exclude_transformer:
            load_kwargs["transformer"] = None

        if self.supports_single_file and os.path.isfile(self.model_id):
            logger.info("Detected single file checkpoint: %s", self.model_id)
            return self.pipeline_class.from_single_file(
                self.model_id, **load_kwargs
            ).to(self.device)
        else:
            return self.pipeline_class.from_pretrained(
                self.model_id, **load_kwargs
            ).to(self.device)

    def _load_coreml_transformer(self, filename: Optional[str] = None) -> ct.models.MLModel:
        """
        Load CoreML transformer model.

        Args:
            filename: Override transformer filename (defaults to transformer_filename)

        Returns:
            Loaded CoreML model
        """
        filename = filename or self.transformer_filename
        model_path = os.path.join(self.model_dir, filename)
        logger.info("Loading Core ML Transformer from %s...", filename)
        return ct.models.MLModel(
            model_path,
            compute_units=ct.ComputeUnit[self.compute_unit],
        )

    def _load_coreml_model(self, filename: str, description: str = "model") -> ct.models.MLModel:
        """
        Load a CoreML model by filename.

        Args:
            filename: Model package filename
            description: Human-readable description for logging

        Returns:
            Loaded CoreML model
        """
        model_path = os.path.join(self.model_dir, filename)
        logger.info("Loading Core ML %s from %s...", description, filename)
        return ct.models.MLModel(
            model_path,
            compute_units=ct.ComputeUnit[self.compute_unit],
        )

    # --- Utility Methods ---

    def to_numpy(
        self,
        tensor: torch.Tensor,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """
        Convert torch tensor to numpy array for CoreML.

        Args:
            tensor: Input tensor
            dtype: Target numpy dtype

        Returns:
            Numpy array
        """
        return tensor.cpu().numpy().astype(dtype)

    def from_numpy(
        self,
        array: np.ndarray,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Convert numpy array from CoreML to torch tensor.

        Args:
            array: Input numpy array
            dtype: Optional target torch dtype

        Returns:
            Torch tensor on self.device
        """
        tensor = torch.from_numpy(array).to(self.device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor

    def predict_coreml(
        self,
        model: ct.models.MLModel,
        inputs: Dict[str, np.ndarray],
        output_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Run CoreML prediction and convert output to tensor.

        Args:
            model: CoreML model to run
            inputs: Dictionary of numpy array inputs
            output_key: Key to extract from output dict (defaults to self.output_key)

        Returns:
            Output tensor on self.device
        """
        output = model.predict(inputs)
        key = output_key or self.output_key
        return self.from_numpy(output[key])
