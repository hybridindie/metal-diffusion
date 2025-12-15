import torch
import torch.nn as nn
import coremltools as ct
from typing import Dict, Any, Tuple, Optional
import logging
from .converter import BaseConverter
import numpy as np

logger = logging.getLogger(__name__)

class LuminaConverter(BaseConverter):
    """
    Converter for Lumina-Image 2.0 models (Next-Gen DiT).
    Uses Gemma-2B as text encoder and Lumina2Transformer2DModel.
    """

    def __init__(self, model_id: str, output_dir: str, quantization: str = "float16", 
                 img_height: int = 1024, img_width: int = 1024):
        super().__init__(model_id, output_dir, quantization)
        self.img_height = img_height
        self.img_width = img_width
        self.pipe = None

    def _get_pipeline(self):
        if self.pipe is None:
            logger.info(f"Loading Lumina pipeline: {self.model_id}")
            from diffusers import Lumina2Pipeline
            self.pipe = Lumina2Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.quantization != "float32" else torch.float32
            )
        return self.pipe

    def convert(self):
        pipeline = self._get_pipeline()
        
        # 1. Convert Text Encoder (Gemma 2B)
        self.convert_text_encoder(pipeline.text_encoder, pipeline.tokenizer)

        # 2. Convert Transformer (DiT)
        self.convert_transformer(pipeline.transformer)
        
        # 3. VAE (Standard SDXL/SD3 VAE usually, or similar)
        # Check if it's the standard AutoencoderKL
        if hasattr(pipeline, "vae"):
            self.convert_vae(pipeline.vae)

    def convert_text_encoder(self, text_encoder, tokenizer):
        name = "Gemma2_TextEncoder"
        logger.info(f"Converting {name}...")
        
        # Gemma inputs
        # We usually use a fixed max length for diffusion text encoders
        max_length = 256 
        vocab_size = text_encoder.config.vocab_size
        hidden_size = text_encoder.config.hidden_size # 2304

        class GemmaWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                # Returns last_hidden_state
                # Gemma output: BaseModelOutputWithPast(last_hidden_state=..., ...)
                outputs = self.model(input_ids=input_ids, output_hidden_states=False)
                return outputs[0]

        wrapped_model = GemmaWrapper(text_encoder).eval()
        
        # Trace
        input_ids = torch.randint(0, vocab_size, (1, max_length), dtype=torch.int32)
        example_input = (input_ids,)

        traced_model = torch.jit.trace(wrapped_model, example_input)

        # Core ML Conversion
        inputs = [
            ct.TensorType(name="input_ids", shape=(1, max_length), dtype=np.int32)
        ]
        outputs = [
            ct.TensorType(name="last_hidden_state")
        ]

        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            outputs=outputs,
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE, # NPU preferred
            skip_model_load=True
        )

        self._save_model(mlmodel, name)
        logger.info(f"Saved {name}")

    def convert_transformer(self, transformer):
        name = "Lumina2_Transformer"
        logger.info(f"Converting {name}...")

        class TransformerWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, hidden_states, encoder_hidden_states, temb):
                # Lumina forward: (hidden_states, encoder_hidden_states, timestep, encoder_mask, ...)
                # Checking signatures...
                # Note: `temb` usually implies timestep embeddings OR raw timestep.
                # Diffusers transformers normally take `timestep` (scalar or tensor).
                
                # IMPORTANT: Lumina input shape checks
                # hidden_states: [B, C, H_latent, W_latent] -> [1, 16, 128, 128] for 1024px
                return self.model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=temb,
                    return_dict=False
                )[0]

        wrapped_model = TransformerWrapper(transformer).eval()

        # Input Shapes
        # Latent size = img_size / 8 (usually) * patch_size?
        # VAE downsample factor is 8.
        h_latent = self.img_height // 8
        w_latent = self.img_width // 8
        in_channels = transformer.config.in_channels # 16
        enc_dim = transformer.config.hidden_size # 2304

        hidden_states = torch.randn(1, in_channels, h_latent, w_latent)
        encoder_hidden_states = torch.randn(1, 256, enc_dim) # [B, SeqLen, Dim]
        temb = torch.tensor([1.0])

        example_input = (hidden_states, encoder_hidden_states, temb)

        traced_model = torch.jit.trace(wrapped_model, example_input)

        inputs = [
            ct.TensorType(name="hidden_states", shape=hidden_states.shape),
            ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
            ct.TensorType(name="timestep", shape=(1,))
        ]

        mlmodel = ct.convert(
            traced_model,
            inputs=inputs,
            minimum_deployment_target=ct.target.macOS15,
            skip_model_load=True
        )

        # Quantize
        if self.quantization == "int8":
            from coremltools.optimize.coreml import linear_quantize_weights, OpLinearQuantizerConfig
            config = OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
            mlmodel = linear_quantize_weights(mlmodel, config=config)
        elif self.quantization == "int4":
             from coremltools.optimize.coreml import linear_quantize_weights, OpLinearQuantizerConfig
             config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", weight_threshold=512)
             mlmodel = linear_quantize_weights(mlmodel, config=config)

        self._save_model(mlmodel, name)
        logger.info(f"Saved {name}")
    
    def convert_vae(self, vae):
        # reuse standard VAE conversion or generic one
        # Assuming standard AutoencoderKL-like
        pass
