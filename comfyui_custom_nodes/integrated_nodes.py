import torch
from diffusers import FluxPipeline, DiffusionPipeline
import folder_paths
import comfy.sd
import comfy.model_patcher

class CoreMLFluxWithCLIP:
    """
    All-in-one Flux loader with integrated text encoders (CLIP-L + T5).
    Eliminates need for separate DualCLIPLoader node.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "transformer_path": (folder_paths.get_filename_list("unet"),),
            "clip_model": (["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"], 
                          {"default": "black-forest-labs/FLUX.1-schnell"}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_model"
    CATEGORY = "Alloy"

    def load_model(self, transformer_path, clip_model):
        """Load Core ML transformer + PyTorch text encoders + VAE"""
        import comfy.model_management
        import comfy.sd
        import comfy.utils
        from alloy.runners.flux import FluxCoreMLRunner
        from .video_wrappers import CoreMLLTXVideoWrapper, CoreMLWanVideoWrapper
        
        # Get full path to transformer
        transformer_full_path = folder_paths.get_full_path("unet", transformer_path)
        print(f"[CoreMLFluxWithCLIP] Loading transformer: {transformer_full_path}")
        
        # Load Core ML transformer
        from ..nodes import CoreMLFluxWrapper
        model_wrapper = CoreMLFluxWrapper(transformer_full_path)
        model = comfy.model_patcher.ModelPatcher(model_wrapper, load_device="cpu", offload_device="cpu")
        
        # Load CLIP + T5 from Hugging Face
        print(f"[CoreMLFluxWithCLIP] Loading CLIP/T5 from: {clip_model}")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        try:
            pipe = DiffusionPipeline.from_pretrained(
                clip_model,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                transformer=None  # Don't load transformer
            )
        except Exception as e:
            print(f"[CoreMLFluxWithCLIP] Error loading from HF: {e}")
            raise
        
        # Extract text encoders
        text_encoder = pipe.text_encoder.to(device)
        text_encoder_2 = pipe.text_encoder_2.to(device)
        tokenizer = pipe.tokenizer
        tokenizer_2 = pipe.tokenizer_2
        
        # Create ComfyUI CLIP object
        clip = FluxCLIPWrapper(text_encoder, text_encoder_2, tokenizer, tokenizer_2, device)
        
        # Extract VAE
        vae = pipe.vae.to(device)
        
        # Wrap VAE for ComfyUI
        from comfy.sd import VAE
        vae_wrapper = VAE(sd=vae)
        
        print("[CoreMLFluxWithCLIP] ✓ All components loaded")
        return (model, clip, vae_wrapper)


class FluxCLIPWrapper:
    """Wrapper to make Flux CLIP/T5 compatible with ComfyUI's CLIP interface"""
    def __init__(self, text_encoder, text_encoder_2, tokenizer, tokenizer_2, device):
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2  # T5
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.device = device
        
    def tokenize(self, text):
        """Tokenize text for both encoders"""
        tokens_1 = self.tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        tokens_2 = self.tokenizer_2(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        return {"tokens_1": tokens_1, "tokens_2": tokens_2}
    
    def encode_from_tokens(self, tokens, return_pooled=False):
        """Encode tokens to embeddings"""
        tokens_1 = tokens["tokens_1"]["input_ids"].to(self.device)
        tokens_2 = tokens["tokens_2"]["input_ids"].to(self.device)
        
        # CLIP-L encoding
        with torch.no_grad():
            output_1 = self.text_encoder(tokens_1, output_hidden_states=True)
            pooled_output = output_1.pooler_output
            hidden_states_1 = output_1.hidden_states[-2]  # Penultimate layer
        
        # T5 encoding
        with torch.no_grad():
            output_2 = self.text_encoder_2(tokens_2, output_hidden_states=True)
            hidden_states_2 = output_2.hidden_states[-1]  # Last layer
        
        # Concatenate along sequence dimension
        # ComfyUI expects (batch, seq, dim)
        cond = torch.cat([hidden_states_1, hidden_states_2], dim=1)
        
        if return_pooled:
            return cond, pooled_output
        return cond
    
    def encode(self, text):
        """Full encode from text"""
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens, return_pooled=True)


class CoreMLLuminaWithCLIP:
    """
    All-in-one Lumina Image 2.0 loader with integrated Gemma text encoder.
    Eliminates need for separate text encoder and VAE loader nodes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "transformer_path": (folder_paths.get_filename_list("unet"),),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_model"
    CATEGORY = "Alloy"

    def load_model(self, transformer_path):
        """Load Core ML transformer + PyTorch Gemma text encoder + VAE"""
        import comfy.model_management
        import comfy.sd
        import comfy.utils
        from diffusers import Lumina2Pipeline
        from .video_wrappers import CoreMLLuminaWrapper

        # Get full path to transformer
        transformer_full_path = folder_paths.get_full_path("unet", transformer_path)
        print(f"[CoreMLLuminaWithCLIP] Loading transformer: {transformer_full_path}")

        # Load Core ML transformer
        model_wrapper = CoreMLLuminaWrapper(transformer_full_path)
        model = comfy.model_patcher.ModelPatcher(model_wrapper, load_device="cpu", offload_device="cpu")

        # Load Gemma + VAE from Hugging Face
        model_id = "Alpha-VLLM/Lumina-Image-2.0"
        print(f"[CoreMLLuminaWithCLIP] Loading Gemma/VAE from: {model_id}")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        try:
            pipe = Lumina2Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                transformer=None  # Don't load transformer
            )
        except Exception as e:
            print(f"[CoreMLLuminaWithCLIP] Error loading from HF: {e}")
            raise

        # Extract text encoder (Gemma 2B)
        text_encoder = pipe.text_encoder.to(device)
        tokenizer = pipe.tokenizer

        # Create ComfyUI CLIP object
        clip = LuminaCLIPWrapper(text_encoder, tokenizer, device)

        # Extract VAE
        vae = pipe.vae.to(device)

        # Wrap VAE for ComfyUI
        from comfy.sd import VAE
        vae_wrapper = VAE(sd=vae)

        print("[CoreMLLuminaWithCLIP] All components loaded")
        return (model, clip, vae_wrapper)


class LuminaCLIPWrapper:
    """Wrapper to make Lumina's Gemma 2B text encoder compatible with ComfyUI's CLIP interface"""
    def __init__(self, text_encoder, tokenizer, device):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device

    def tokenize(self, text):
        """Tokenize text for Gemma encoder"""
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=256,  # Lumina uses 256 max length
            truncation=True,
            return_tensors="pt"
        )
        return {"tokens": tokens}

    def encode_from_tokens(self, tokens, return_pooled=False):
        """Encode tokens to embeddings"""
        input_ids = tokens["tokens"]["input_ids"].to(self.device)
        attention_mask = tokens["tokens"].get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Gemma encoding
        with torch.no_grad():
            if attention_mask is not None:
                output = self.text_encoder(input_ids, attention_mask=attention_mask)
            else:
                output = self.text_encoder(input_ids)

            # Get last hidden state
            if hasattr(output, "last_hidden_state"):
                hidden_states = output.last_hidden_state
            else:
                hidden_states = output[0]

        # ComfyUI expects (batch, seq, dim)
        if return_pooled:
            # Use mean pooling for pooled output
            pooled = hidden_states.mean(dim=1)
            return hidden_states, pooled
        return hidden_states

    def encode(self, text):
        """Full encode from text"""
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens, return_pooled=True)


NODE_CLASS_MAPPINGS = {
    "CoreMLFluxWithCLIP": CoreMLFluxWithCLIP,
    "CoreMLLuminaWithCLIP": CoreMLLuminaWithCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLFluxWithCLIP": "Core ML Flux (All-in-One)",
    "CoreMLLuminaWithCLIP": "Core ML Lumina (All-in-One)",
}
