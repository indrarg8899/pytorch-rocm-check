"""Model compatibility database for ROCm."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelCompat:
    name: str
    family: str
    compatible: bool
    min_rocm: str
    max_batch_size: int = 0
    notes: str = ""
    workarounds: List[str] = None

    def __post_init__(self):
        if self.workarounds is None:
            self.workarounds = []


MODEL_COMPAT_DB: Dict[str, ModelCompat] = {
    # Vision models
    "resnet18": ModelCompat("ResNet-18", "ResNet", True, "5.5"),
    "resnet34": ModelCompat("ResNet-34", "ResNet", True, "5.5"),
    "resnet50": ModelCompat("ResNet-50", "ResNet", True, "5.5"),
    "resnet101": ModelCompat("ResNet-101", "ResNet", True, "5.5"),
    "resnet152": ModelCompat("ResNet-152", "ResNet", True, "5.5"),
    "vgg16": ModelCompat("VGG-16", "VGG", True, "5.5"),
    "vgg19": ModelCompat("VGG-19", "VGG", True, "5.5"),
    "densenet121": ModelCompat("DenseNet-121", "DenseNet", True, "5.5"),
    "mobilenet_v2": ModelCompat("MobileNet-V2", "MobileNet", True, "5.5"),
    "mobilenet_v3_large": ModelCompat("MobileNet-V3-Large", "MobileNet", True, "5.6"),
    "efficientnet_b0": ModelCompat("EfficientNet-B0", "EfficientNet", True, "5.6"),
    "efficientnet_b4": ModelCompat("EfficientNet-B4", "EfficientNet", True, "5.6"),
    "vit_b_16": ModelCompat("ViT-B/16", "VisionTransformer", True, "5.7"),
    "vit_l_16": ModelCompat("ViT-L/16", "VisionTransformer", True, "5.7"),
    "swin_t": ModelCompat("Swin-T", "SwinTransformer", True, "5.7"),
    "convnext_tiny": ModelCompat("ConvNeXt-Tiny", "ConvNeXt", True, "5.7"),
    "inception_v3": ModelCompat("Inception-V3", "Inception", True, "5.5"),

    # NLP models
    "bert-base-uncased": ModelCompat("BERT-Base", "BERT", True, "5.6"),
    "bert-large-uncased": ModelCompat("BERT-Large", "BERT", True, "5.6"),
    "gpt2": ModelCompat("GPT-2", "GPT", True, "5.7"),
    "gpt2-medium": ModelCompat("GPT-2-Medium", "GPT", True, "5.7"),
    "gpt2-large": ModelCompat("GPT-2-Large", "GPT", True, "5.7"),
    "llama-7b": ModelCompat("LLaMA-7B", "LLaMA", True, "5.7", notes="Needs flash-attn workaround"),
    "llama-13b": ModelCompat("LLaMA-13B", "LLaMA", True, "5.7", notes="Needs flash-attn workaround"),
    "llama-70b": ModelCompat("LLaMA-70B", "LLaMA", True, "6.0", notes="Multi-GPU only"),
    "mistral-7b": ModelCompat("Mistral-7B", "Mistral", True, "5.7"),
    "t5-small": ModelCompat("T5-Small", "T5", True, "5.6"),
    "t5-base": ModelCompat("T5-Base", "T5", True, "5.6"),
    "roberta-base": ModelCompat("RoBERTa-Base", "RoBERTa", True, "5.6"),
    "distilbert-base-uncased": ModelCompat("DistilBERT", "DistilBERT", True, "5.6"),

    # Diffusion models
    "stable-diffusion-v1-5": ModelCompat("SD-v1.5", "StableDiffusion", True, "5.7", notes="Half precision recommended"),
    "stable-diffusion-xl": ModelCompat("SDXL", "StableDiffusion", True, "6.0"),

    # Problematic models
    "wav2vec2-base": ModelCompat("Wav2Vec2", "Wav2Vec2", False, "5.7", notes="Custom CUDA kernels unsupported"),
    "deformable-detr": ModelCompat("Deformable-DETR", "DETR", False, "5.7", notes="Deformable attention CUDA ops"),
}


def get_model_compat(name: str) -> ModelCompat:
    name_lower = name.lower().replace(" ", "_").replace("-", "_")
    if name_lower in MODEL_COMPAT_DB:
        return MODEL_COMPAT_DB[name_lower]
    for key, val in MODEL_COMPAT_DB.items():
        if name_lower in key or key in name_lower:
            return val
    return ModelCompat(name, "Unknown", False, "N/A", notes="Not in database")


def get_compatible_models() -> List[ModelCompat]:
    return [m for m in MODEL_COMPAT_DB.values() if m.compatible]


def get_incompatible_models() -> List[ModelCompat]:
    return [m for m in MODEL_COMPAT_DB.values() if not m.compatible]
