"""
PyTorch ROCm Compatibility Checker Engine.

Provides automated compatibility checking between PyTorch versions
and AMD ROCm/HIP stack configurations.
"""

import subprocess
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum


class CompatLevel(Enum):
    FULL = "full"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


@dataclass
class CompatResult:
    """Result of a compatibility check."""
    pytorch_version: str
    rocm_version: str
    is_compatible: bool
    level: CompatLevel
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    known_issues: List[str] = field(default_factory=list)


@dataclass
class GPUInfo:
    """Detected GPU information."""
    name: str
    vram_gb: float
    compute_capability: str
    supported_rocm: List[str]


class ROCmChecker:
    """Core ROCm ↔ PyTorch compatibility checking engine."""

    # Compatibility matrix: (pytorch_major.minor, rocm_major.minor) -> CompatLevel
    COMPAT_MATRIX: Dict[Tuple[str, str], CompatLevel] = {
        ("2.0", "5.6"): CompatLevel.FULL,
        ("2.0", "5.7"): CompatLevel.FULL,
        ("2.0", "6.0"): CompatLevel.PARTIAL,
        ("2.1", "5.6"): CompatLevel.FULL,
        ("2.1", "5.7"): CompatLevel.FULL,
        ("2.1", "6.0"): CompatLevel.FULL,
        ("2.1", "6.1"): CompatLevel.PARTIAL,
        ("2.2", "5.7"): CompatLevel.FULL,
        ("2.2", "6.0"): CompatLevel.FULL,
        ("2.2", "6.1"): CompatLevel.FULL,
        ("2.3", "6.0"): CompatLevel.FULL,
        ("2.3", "6.1"): CompatLevel.FULL,
        ("2.4", "6.0"): CompatLevel.FULL,
        ("2.4", "6.1"): CompatLevel.FULL,
    }

    SUPPORTED_GPUS = {
        "MI50": GPUInfo("Radeon VII / MI50", 16.0, "gfx906", ["5.6", "5.7"]),
        "MI100": GPUInfo("Instinct MI100", 32.0, "gfx908", ["5.6", "5.7", "6.0"]),
        "MI210": GPUInfo("Instinct MI210", 64.0, "gfx90a", ["5.7", "6.0", "6.1"]),
        "MI250X": GPUInfo("Instinct MI250X", 128.0, "gfx90a", ["5.7", "6.0", "6.1"]),
        "MI300X": GPUInfo("Instinct MI300X", 192.0, "gfx942", ["6.0", "6.1", "6.2"]),
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._detected_pytorch: Optional[str] = None
        self._detected_rocm: Optional[str] = None
        self._detected_gpu: Optional[str] = None

    def check(
        self,
        pytorch_version: Optional[str] = None,
        rocm_version: Optional[str] = None,
    ) -> CompatResult:
        """Run full compatibility check."""
        if pytorch_version is None:
            pytorch_version = self._detect_pytorch_version()
        if rocm_version is None:
            rocm_version = self._detect_rocm_version()

        if pytorch_version is None or rocm_version is None:
            return CompatResult(
                pytorch_version=pytorch_version or "unknown",
                rocm_version=rocm_version or "unknown",
                is_compatible=False,
                level=CompatLevel.UNSUPPORTED,
                recommendations=["Could not detect versions. Please specify manually."],
            )

        pt_short = self._short_version(pytorch_version)
        rocm_short = self._short_version(rocm_version)

        key = (pt_short, rocm_short)
        level = self.COMPAT_MATRIX.get(key, CompatLevel.UNSUPPORTED)

        recommendations = []
        warnings = []
        known_issues = []

        if level == CompatLevel.UNSUPPORTED:
            compatible_versions = [
                (k, v) for k, v in self.COMPAT_MATRIX.items()
                if k[0] == pt_short or k[1] == rocm_short
            ]
            if compatible_versions:
                best = max(compatible_versions, key=lambda x: 1 if x[1] == CompatLevel.FULL else 0)
                recommendations.append(
                    f"Try PyTorch {best[0][0]} with ROCm {best[0][1]} for full support."
                )
            else:
                recommendations.append("No compatible combination found in matrix. Check AMD docs.")

        elif level == CompatLevel.PARTIAL:
            warnings.append("Partial support only. Some features may not work correctly.")
            recommendations.append("Review known issues for specific limitations.")

        if self._detected_gpu:
            gpu_match = self._match_gpu(self._detected_gpu)
            if gpu_match and rocm_short not in self.SUPPORTED_GPUS[gpu_match].supported_rocm:
                warnings.append(f"{gpu_match} not officially supported on ROCm {rocm_short}.")

        return CompatResult(
            pytorch_version=pytorch_version,
            rocm_version=rocm_version,
            is_compatible=level in (CompatLevel.FULL, CompatLevel.PARTIAL),
            level=level,
            recommendations=recommendations,
            warnings=warnings,
            known_issues=known_issues,
        )

    def _detect_pytorch_version(self) -> Optional[str]:
        """Detect installed PyTorch version."""
        try:
            import torch
            return torch.__version__
        except ImportError:
            pass
        # Fallback: pip
        try:
            result = subprocess.run(
                ["pip", "show", "torch"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return None

    def _detect_rocm_version(self) -> Optional[str]:
        """Detect installed ROCm version."""
        # Try rocminfo
        try:
            result = subprocess.run(
                ["rocminfo"], capture_output=True, text=True, timeout=10
            )
            match = re.search(r"HSA Runtime.*?(\d+\.\d+(?:\.\d+)?)", result.stdout)
            if match:
                return match.group(1)
        except Exception:
            pass
        # Try /opt/rocm version
        try:
            result = subprocess.run(
                ["ls", "-la", "/opt/rocm"],
                capture_output=True, text=True, timeout=5
            )
            match = re.search(r"rocm-(\d+\.\d+(?:\.\d+)?)", result.stdout)
            if match:
                return match.group(1)
        except Exception:
            pass
        # Try hipconfig
        try:
            result = subprocess.run(
                ["hipconfig", "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _short_version(self, version: str) -> str:
        """Extract major.minor from version string."""
        parts = version.split(".")
        return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else version

    def _match_gpu(self, gpu_name: str) -> Optional[str]:
        """Match detected GPU name to known GPU type."""
        name_upper = gpu_name.upper()
        for key in ["MI300X", "MI250X", "MI210", "MI100", "MI50"]:
            if key in name_upper:
                return key
        return None


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch ROCm Compatibility Checker")
    parser.add_argument("--pytorch", type=str, help="PyTorch version (e.g., 2.1.0)")
    parser.add_argument("--rocm", type=str, help="ROCm version (e.g., 5.7)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--matrix", action="store_true", help="Show full compatibility matrix")
    args = parser.parse_args()

    checker = ROCmChecker(verbose=args.verbose)
    result = checker.check(pytorch_version=args.pytorch, rocm_version=args.rocm)

    print(f"\n{'='*60}")
    print(f"  PyTorch {result.pytorch_version} + ROCm {result.rocm_version}")
    print(f"{'='*60}")
    print(f"  Status: {'✅ COMPATIBLE' if result.is_compatible else '❌ NOT COMPATIBLE'}")
    print(f"  Level:  {result.level.value.upper()}")

    if result.warnings:
        print(f"\n  ⚠️  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")
    if result.recommendations:
        print(f"\n  💡 Recommendations:")
        for r in result.recommendations:
            print(f"    - {r}")
    if result.known_issues:
        print(f"\n  🐛 Known Issues:")
        for i in result.known_issues:
            print(f"    - {i}")

    print()


if __name__ == "__main__":
    main()
