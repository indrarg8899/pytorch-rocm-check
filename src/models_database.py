"""Extended models database with detailed compatibility info."""
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class PerfImpact(Enum):
    NONE = "no_impact"
    MINOR = "minor_slowdown"
    MODERATE = "moderate_slowdown"
    MAJOR = "major_slowdown"
    BROKEN = "broken"


@dataclass
class ModelEntry:
    name: str
    family: str
    size_params: int
    compatible: bool
    min_rocm: str
    perf_impact: PerfImpact = PerfImpact.NONE
    notes: str = ""
    required_workarounds: List[str] = None
    tested_rocm_versions: List[str] = None

    def __post_init__(self):
        if self.required_workarounds is None:
            self.required_workarounds = []
        if self.tested_rocm_versions is None:
            self.tested_rocm_versions = []


KNOWN_MODELS: Dict[str, ModelEntry] = {
    "resnet50": ModelEntry("ResNet-50", "vision", 25_000_000, True, "5.5",
                           tested_rocm_versions=["5.5", "5.6", "5.7", "6.0"]),
    "bert-base-uncased": ModelEntry("BERT-Base", "nlp", 110_000_000, True, "5.6",
                                    tested_rocm_versions=["5.6", "5.7"]),
    "gpt2": ModelEntry("GPT-2", "nlp", 124_000_000, True, "5.7",
                       tested_rocm_versions=["5.7", "6.0"]),
    "llama-7b": ModelEntry("LLaMA-7B", "llm", 7_000_000_000, True, "5.7",
                           perf_impact=PerfImpact.MINOR,
                           required_workarounds=["flash_attn"],
                           tested_rocm_versions=["5.7", "6.0"]),
    "stable-diffusion-v1-5": ModelEntry("Stable Diffusion v1.5", "diffusion", 860_000_000, True, "5.7",
                                        perf_impact=PerfImpact.MINOR,
                                        tested_rocm_versions=["5.7", "6.0"]),
    "deformable-detr": ModelEntry("Deformable DETR", "detection", 40_000_000, False, "N/A",
                                  notes="Custom CUDA kernels not ported"),
}


def query_models(
    family: Optional[str] = None,
    compatible_only: bool = False,
    max_rocm: str = "6.0",
) -> List[ModelEntry]:
    results = list(KNOWN_MODELS.values())
    if family:
        results = [m for m in results if m.family == family]
    if compatible_only:
        results = [m for m in results if m.compatible]
    return results
