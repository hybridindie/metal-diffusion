import torch
from diffusers import DiffusionPipeline
import folder_paths
import comfy.sd
import comfy.model_patcher

# Try to import pipelines (may not be available in older diffusers)
try:
    from diffusers import LTXPipeline
    LTX_AVAILABLE = True
except ImportError:
    LTX_AVAILABLE = False

try:
    from diffusers import WanPipeline
    WAN_AVAILABLE = True
except ImportError:
    WAN_AVAILABLE = False

try:
    from diffusers import HunyuanVideoPipeline
    HUNYUAN_AVAILABLE = True
except ImportError:
    HUNYUAN_AVAILABLE = False

class CoreMLFluxWithCLIP:
    """
    All-in-one Flux loader with integrated text encoders (CLIP-L + T5).
    Eliminates need for separate DualCLIPLoader node.
    """
    @classmethod
    def INPUT_TYPES(cls):
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
    def INPUT_TYPES(cls):
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

        print("[CoreMLLuminaWithCLIP] ✓ All components loaded")
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


# =============================================================================
# LTX-Video Integrated Loader
# =============================================================================

class CoreMLLTXVideoWithCLIP:
    """
    All-in-one LTX-Video loader with integrated T5 text encoder and VAE.
    Eliminates need for separate text encoder and VAE loader nodes.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "transformer_path": (folder_paths.get_filename_list("unet"),),
            "num_frames": ("INT", {"default": 25, "min": 1, "max": 257}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_model"
    CATEGORY = "Alloy/Video"

    def load_model(self, transformer_path, num_frames):
        """Load Core ML transformer + PyTorch T5 text encoder + VAE"""
        if not LTX_AVAILABLE:
            raise ImportError("LTXPipeline not available. Please upgrade diffusers: pip install -U diffusers")

        from .video_wrappers import CoreMLLTXVideoWrapper

        # Get full path to transformer
        transformer_full_path = folder_paths.get_full_path("unet", transformer_path)
        print(f"[CoreMLLTXVideoWithCLIP] Loading transformer: {transformer_full_path}")

        # Load Core ML transformer
        model_wrapper = CoreMLLTXVideoWrapper(transformer_full_path, num_frames)
        model = comfy.model_patcher.ModelPatcher(model_wrapper, load_device="cpu", offload_device="cpu")

        # Load T5 + VAE from Hugging Face
        model_id = "Lightricks/LTX-Video"
        print(f"[CoreMLLTXVideoWithCLIP] Loading T5/VAE from: {model_id}")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        try:
            pipe = LTXPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                transformer=None  # Don't load transformer
            )
        except Exception as e:
            print(f"[CoreMLLTXVideoWithCLIP] Error loading from HF: {e}")
            raise

        # Extract text encoder (T5)
        text_encoder = pipe.text_encoder.to(device)
        tokenizer = pipe.tokenizer

        # Create ComfyUI CLIP object
        clip = LTXCLIPWrapper(text_encoder, tokenizer, device)

        # Extract VAE
        vae = pipe.vae.to(device)

        # Wrap VAE for ComfyUI
        from comfy.sd import VAE
        vae_wrapper = VAE(sd=vae)

        print("[CoreMLLTXVideoWithCLIP] ✓ All components loaded")
        return (model, clip, vae_wrapper)


class LTXCLIPWrapper:
    """Wrapper to make LTX's T5 text encoder compatible with ComfyUI's CLIP interface"""
    def __init__(self, text_encoder, tokenizer, device):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device

    def tokenize(self, text):
        """Tokenize text for T5 encoder"""
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=512,  # LTX uses 512 max length for T5
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

        # T5 encoding
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
            if attention_mask is not None:
                pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
            else:
                pooled = hidden_states.mean(dim=1)
            return hidden_states, pooled
        return hidden_states

    def encode(self, text):
        """Full encode from text"""
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens, return_pooled=True)


# =============================================================================
# Wan Video Integrated Loader
# =============================================================================

class CoreMLWanVideoWithCLIP:
    """
    All-in-one Wan Video loader with integrated text encoder and VAE.
    Supports Wan 2.1 models (T2V and I2V).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "transformer_path": (folder_paths.get_filename_list("unet"),),
            "num_frames": ("INT", {"default": 16, "min": 1, "max": 128}),
            "model_variant": (["Wan-AI/Wan2.1-T2V-14B-Diffusers", "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"],
                             {"default": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_model"
    CATEGORY = "Alloy/Video"

    def load_model(self, transformer_path, num_frames, model_variant):
        """Load Core ML transformer + PyTorch text encoder + VAE"""
        if not WAN_AVAILABLE:
            raise ImportError("WanPipeline not available. Please upgrade diffusers: pip install -U diffusers")

        from .video_wrappers import CoreMLWanVideoWrapper

        # Get full path to transformer
        transformer_full_path = folder_paths.get_full_path("unet", transformer_path)
        print(f"[CoreMLWanVideoWithCLIP] Loading transformer: {transformer_full_path}")

        # Load Core ML transformer
        model_wrapper = CoreMLWanVideoWrapper(transformer_full_path, num_frames)
        model = comfy.model_patcher.ModelPatcher(model_wrapper, load_device="cpu", offload_device="cpu")

        # Load text encoder + VAE from Hugging Face
        print(f"[CoreMLWanVideoWithCLIP] Loading text encoder/VAE from: {model_variant}")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        try:
            pipe = WanPipeline.from_pretrained(
                model_variant,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                transformer=None  # Don't load transformer
            )
        except Exception as e:
            print(f"[CoreMLWanVideoWithCLIP] Error loading from HF: {e}")
            raise

        # Extract text encoder
        text_encoder = pipe.text_encoder.to(device)
        tokenizer = pipe.tokenizer

        # Create ComfyUI CLIP object
        clip = WanCLIPWrapper(text_encoder, tokenizer, device)

        # Extract VAE
        vae = pipe.vae.to(device)

        # Wrap VAE for ComfyUI
        from comfy.sd import VAE
        vae_wrapper = VAE(sd=vae)

        print("[CoreMLWanVideoWithCLIP] ✓ All components loaded")
        return (model, clip, vae_wrapper)


class WanCLIPWrapper:
    """Wrapper to make Wan's text encoder compatible with ComfyUI's CLIP interface"""
    def __init__(self, text_encoder, tokenizer, device):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device

    def tokenize(self, text):
        """Tokenize text for Wan encoder"""
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=512,
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

        # Text encoding
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
            if attention_mask is not None:
                pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
            else:
                pooled = hidden_states.mean(dim=1)
            return hidden_states, pooled
        return hidden_states

    def encode(self, text):
        """Full encode from text"""
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens, return_pooled=True)


# =============================================================================
# Hunyuan Video Integrated Loader
# =============================================================================

class CoreMLHunyuanVideoWithCLIP:
    """
    All-in-one HunyuanVideo loader with integrated text encoders and VAE.
    Uses LLAVA and CLIP text encoders for prompt conditioning.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "transformer_path": (folder_paths.get_filename_list("unet"),),
            "num_frames": ("INT", {"default": 16, "min": 1, "max": 128}),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_model"
    CATEGORY = "Alloy/Video"

    def load_model(self, transformer_path, num_frames):
        """Load Core ML transformer + PyTorch text encoders + VAE"""
        if not HUNYUAN_AVAILABLE:
            raise ImportError("HunyuanVideoPipeline not available. Please upgrade diffusers: pip install -U diffusers")

        from .video_wrappers import CoreMLHunyuanVideoWrapper

        # Get full path to transformer
        transformer_full_path = folder_paths.get_full_path("unet", transformer_path)
        print(f"[CoreMLHunyuanVideoWithCLIP] Loading transformer: {transformer_full_path}")

        # Load Core ML transformer
        model_wrapper = CoreMLHunyuanVideoWrapper(transformer_full_path, num_frames)
        model = comfy.model_patcher.ModelPatcher(model_wrapper, load_device="cpu", offload_device="cpu")

        # Load text encoders + VAE from Hugging Face
        model_id = "hunyuanvideo-community/HunyuanVideo"
        print(f"[CoreMLHunyuanVideoWithCLIP] Loading text encoders/VAE from: {model_id}")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        try:
            pipe = HunyuanVideoPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                transformer=None  # Don't load transformer
            )
        except Exception as e:
            print(f"[CoreMLHunyuanVideoWithCLIP] Error loading from HF: {e}")
            raise

        # Extract text encoders
        text_encoder = pipe.text_encoder.to(device)
        tokenizer = pipe.tokenizer

        # Check for secondary text encoder (CLIP)
        text_encoder_2 = getattr(pipe, "text_encoder_2", None)
        tokenizer_2 = getattr(pipe, "tokenizer_2", None)
        if text_encoder_2 is not None:
            text_encoder_2 = text_encoder_2.to(device)

        # Create ComfyUI CLIP object
        clip = HunyuanCLIPWrapper(text_encoder, tokenizer, text_encoder_2, tokenizer_2, device)

        # Extract VAE
        vae = pipe.vae.to(device)

        # Wrap VAE for ComfyUI
        from comfy.sd import VAE
        vae_wrapper = VAE(sd=vae)

        print("[CoreMLHunyuanVideoWithCLIP] ✓ All components loaded")
        return (model, clip, vae_wrapper)


class HunyuanCLIPWrapper:
    """Wrapper to make Hunyuan's text encoders compatible with ComfyUI's CLIP interface"""
    def __init__(self, text_encoder, tokenizer, text_encoder_2, tokenizer_2, device):
        self.text_encoder = text_encoder  # Primary (LLAVA/mT5)
        self.tokenizer = tokenizer
        self.text_encoder_2 = text_encoder_2  # Secondary (CLIP, optional)
        self.tokenizer_2 = tokenizer_2
        self.device = device

    def tokenize(self, text):
        """Tokenize text for Hunyuan encoders"""
        tokens_1 = self.tokenizer(
            text,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )

        result = {"tokens_1": tokens_1}

        # Tokenize for secondary encoder if available
        if self.tokenizer_2 is not None:
            tokens_2 = self.tokenizer_2(
                text,
                padding="max_length",
                max_length=77,  # CLIP uses 77
                truncation=True,
                return_tensors="pt"
            )
            result["tokens_2"] = tokens_2

        return result

    def encode_from_tokens(self, tokens, return_pooled=False):
        """Encode tokens to embeddings"""
        input_ids_1 = tokens["tokens_1"]["input_ids"].to(self.device)
        attention_mask_1 = tokens["tokens_1"].get("attention_mask", None)
        if attention_mask_1 is not None:
            attention_mask_1 = attention_mask_1.to(self.device)

        # Primary encoder
        with torch.no_grad():
            if attention_mask_1 is not None:
                output_1 = self.text_encoder(input_ids_1, attention_mask=attention_mask_1)
            else:
                output_1 = self.text_encoder(input_ids_1)

            if hasattr(output_1, "last_hidden_state"):
                hidden_states = output_1.last_hidden_state
            else:
                hidden_states = output_1[0]

        # Secondary encoder (CLIP) if available
        pooled_output = None
        if self.text_encoder_2 is not None and "tokens_2" in tokens:
            input_ids_2 = tokens["tokens_2"]["input_ids"].to(self.device)
            attention_mask_2 = tokens["tokens_2"].get("attention_mask", None)
            if attention_mask_2 is not None:
                attention_mask_2 = attention_mask_2.to(self.device)
            with torch.no_grad():
                if attention_mask_2 is not None:
                    output_2 = self.text_encoder_2(input_ids_2, attention_mask=attention_mask_2, output_hidden_states=True)
                else:
                    output_2 = self.text_encoder_2(input_ids_2, output_hidden_states=True)
                if hasattr(output_2, "pooler_output"):
                    pooled_output = output_2.pooler_output
                else:
                    # Fallback to mean pooling
                    pooled_output = output_2.last_hidden_state.mean(dim=1)

        # ComfyUI expects (batch, seq, dim)
        if return_pooled:
            if pooled_output is None:
                # Use mean pooling from primary encoder
                if attention_mask_1 is not None:
                    pooled_output = (hidden_states * attention_mask_1.unsqueeze(-1)).sum(dim=1) / attention_mask_1.sum(dim=1, keepdim=True).clamp(min=1e-8)
                else:
                    pooled_output = hidden_states.mean(dim=1)
            return hidden_states, pooled_output
        return hidden_states

    def encode(self, text):
        """Full encode from text"""
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens, return_pooled=True)


NODE_CLASS_MAPPINGS = {
    "CoreMLFluxWithCLIP": CoreMLFluxWithCLIP,
    "CoreMLLuminaWithCLIP": CoreMLLuminaWithCLIP,
    "CoreMLLTXVideoWithCLIP": CoreMLLTXVideoWithCLIP,
    "CoreMLWanVideoWithCLIP": CoreMLWanVideoWithCLIP,
    "CoreMLHunyuanVideoWithCLIP": CoreMLHunyuanVideoWithCLIP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoreMLFluxWithCLIP": "Core ML Flux (All-in-One)",
    "CoreMLLuminaWithCLIP": "Core ML Lumina (All-in-One)",
    "CoreMLLTXVideoWithCLIP": "Core ML LTX Video (All-in-One)",
    "CoreMLWanVideoWithCLIP": "Core ML Wan Video (All-in-One)",
    "CoreMLHunyuanVideoWithCLIP": "Core ML Hunyuan Video (All-in-One)",
}
