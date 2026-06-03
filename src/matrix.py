"""
Version Compatibility Matrix for ROCm + PyTorch.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


class SupportLevel(Enum):
    FULL = "full"
    PARTIAL = "partial"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


@dataclass
class CompatEntry:
    pytorch: str
    rocm: str
    gpu: str
    support: SupportLevel
    notes: str = ""
    performance_pct: float = 100.0


class CompatibilityMatrix:
    PYTORCH_VERSIONS = ["2.0", "2.1", "2.2", "2.3", "2.4"]
    ROCM_VERSIONS = ["5.6", "5.7", "6.0", "6.1", "6.2"]
    GPU_MODELS = ["MI50", "MI100", "MI210", "MI250X", "MI300X"]

    VERSION_MATRIX: Dict[Tuple[str, str], SupportLevel] = {
        ("2.0", "5.6"): SupportLevel.FULL,
        ("2.0", "5.7"): SupportLevel.FULL,
        ("2.0", "6.0"): SupportLevel.PARTIAL,
        ("2.1", "5.6"): SupportLevel.FULL,
        ("2.1", "5.7"): SupportLevel.FULL,
        ("2.1", "6.0"): SupportLevel.FULL,
        ("2.1", "6.1"): SupportLevel.PARTIAL,
        ("2.2", "5.7"): SupportLevel.FULL,
        ("2.2", "6.0"): SupportLevel.FULL,
        ("2.2", "6.1"): SupportLevel.FULL,
        ("2.3", "6.0"): SupportLevel.FULL,
        ("2.3", "6.1"): SupportLevel.FULL,
        ("2.3", "6.2"): SupportLevel.EXPERIMENTAL,
        ("2.4", "6.0"): SupportLevel.FULL,
        ("2.4", "6.1"): SupportLevel.FULL,
        ("2.4", "6.2"): SupportLevel.FULL,
    }

    GPU_ROCM_SUPPORT: Dict[str, List[str]] = {
        "MI50": ["5.6", "5.7"],
        "MI100": ["5.6", "5.7", "6.0"],
        "MI210": ["5.7", "6.0", "6.1"],
        "MI250X": ["5.7", "6.0", "6.1"],
        "MI300X": ["6.0", "6.1", "6.2"],
    }

    GPU_PERFORMANCE: Dict[str, Dict[str, float]] = {
        "MI50": {"fp64": 110, "fp32": 35, "fp16": 70},
        "MI100": {"fp64": 150, "fp32": 45, "fp16": 115},
        "MI210": {"fp64": 140, "fp32": 42, "fp16": 105},
        "MI250X": {"fp64": 280, "fp32": 85, "fp16": 220},
        "MI300X": {"fp64": 300, "fp32": 130, "fp16": 350, "bf16": 350, "fp8": 700},
    }

    def __init__(self):
        self.entries: List[CompatEntry] = self._build_entries()

    def _build_entries(self) -> List[CompatEntry]:
        entries = []
        for (pt, rocm), level in self.VERSION_MATRIX.items():
            for gpu in self.GPU_MODELS:
                if rocm in self.GPU_ROCM_SUPPORT.get(gpu, []):
                    entries.append(CompatEntry(
                        pytorch=pt, rocm=rocm, gpu=gpu, support=level,
                        performance_pct=self._estimate_performance(gpu, level),
                    ))
        return entries

    def _estimate_performance(self, gpu, level):
        base = self.GPU_PERFORMANCE.get(gpu, {}).get("fp32", 50)
        mult = {SupportLevel.FULL: 1.0, SupportLevel.PARTIAL: 0.85,
                SupportLevel.EXPERIMENTAL: 0.7, SupportLevel.UNSUPPORTED: 0.0}
        return base * mult.get(level, 0)

    def query(self, pytorch=None, rocm=None, gpu=None) -> List[CompatEntry]:
        results = self.entries
        if pytorch:
            results = [e for e in results if e.pytorch == pytorch]
        if rocm:
            results = [e for e in results if e.rocm == rocm]
        if gpu:
            results = [e for e in results if e.gpu == gpu]
        return results

    def get_best_combo(self, gpu: str) -> Optional[CompatEntry]:
        candidates = [e for e in self.entries
                      if e.gpu == gpu and e.support in (SupportLevel.FULL, SupportLevel.PARTIAL)]
        if not candidates:
            return None
        return max(candidates, key=lambda e: (2 if e.support == SupportLevel.FULL else 1, float(e.rocm)))

    def print_matrix(self):
        print(f"\n{'PyTorch':<10} {'ROCm':<8}", end="")
        for gpu in self.GPU_MODELS:
            print(f" {gpu:<10}", end="")
        print()
        print("-" * 90)
        for pt in self.PYTORCH_VERSIONS:
            for rocm in self.ROCM_VERSIONS:
                key = (pt, rocm)
                if key not in self.VERSION_MATRIX:
                    continue
                print(f"{pt:<10} {rocm:<8}", end="")
                for gpu in self.GPU_MODELS:
                    if rocm in self.GPU_ROCM_SUPPORT.get(gpu, []):
                        sym = {SupportLevel.FULL: "Y", SupportLevel.PARTIAL: "~",
                               SupportLevel.EXPERIMENTAL: "E", SupportLevel.UNSUPPORTED: "N"}
                        print(f" {sym[self.VERSION_MATRIX[key]]:<10}", end="")
                    else:
                        print(f" {'-':<10}", end="")
                print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ROCm Compatibility Matrix")
    parser.add_argument("--pytorch", type=str)
    parser.add_argument("--rocm", type=str)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--best", type=str)
    args = parser.parse_args()

    matrix = CompatibilityMatrix()
    if args.best:
        entry = matrix.get_best_combo(args.best)
        if entry:
            print(f"Best for {args.best}: PyTorch {entry.pytorch} + ROCm {entry.rocm}")
        else:
            print(f"No compatible combo for {args.best}")
    elif args.pytorch or args.rocm or args.gpu:
        for r in matrix.query(pytorch=args.pytorch, rocm=args.rocm, gpu=args.gpu):
            print(f"PT {r.pytorch} + ROCm {r.rocm} + {r.gpu}: {r.support.value}")
    else:
        matrix.print_matrix()


if __name__ == "__main__":
    main()
