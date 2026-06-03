"""ROCm Environment Detection."""
import os
import re
import subprocess
import platform
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class GPUInfo:
    name: str = "Unknown"
    vram_mb: int = 0
    compute_unit_count: int = 0
    driver_version: str = ""
    pci_id: str = ""


@dataclass
class ROCmEnvironment:
    rocm_version: str = "Not Installed"
    hip_version: str = ""
    gpu_devices: List[GPUInfo] = field(default_factory=list)
    os_info: str = ""
    python_version: str = ""
    torch_version: str = ""
    torch_cuda_available: bool = False
    torch_rocm_available: bool = False
    hsa_agents: List[str] = field(default_factory=list)
    is_wsl: bool = False

    @property
    def is_rocm_available(self) -> bool:
        return self.rocm_version != "Not Installed"

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  ROCm Environment Summary",
            "=" * 50,
            f"  ROCm Version:     {self.rocm_version}",
            f"  HIP Version:      {self.hip_version or 'N/A'}",
            f"  OS:               {self.os_info}",
            f"  Python:           {self.python_version}",
            f"  PyTorch:          {self.torch_version}",
            f"  ROCm Available:   {self.is_rocm_available}",
            f"  WSL:              {self.is_wsl}",
            f"  GPU Devices:      {len(self.gpu_devices)}",
        ]
        for i, gpu in enumerate(self.gpu_devices):
            lines.append(f"    [{i}] {gpu.name} ({gpu.vram_mb}MB)")
        lines.append("=" * 50)
        return "\n".join(lines)


def detect_rocm_version() -> str:
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    version_file = os.path.join(rocm_path, ".info", "version")
    if os.path.exists(version_file):
        with open(version_file) as f:
            return f.read().strip()
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            match = re.search(r"ROCM Version:\s*(\S+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        result = subprocess.run(
            ["hipconfig", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return f"Detected via HIP ({result.stdout.strip()})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "Not Installed"


def detect_gpu_devices() -> List[GPUInfo]:
    devices = []
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            for key, val in data.items():
                if key.startswith("card"):
                    devices.append(GPUInfo(
                        name=val.get("Card series", "Unknown AMD GPU"),
                        vram_mb=int(val.get("VRAM Total Memory (B)", 0)) // (1024 * 1024),
                    ))
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    if not devices:
        try:
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                if "AMD" in line and ("VGA" in line or "Display" in line or "3D" in line):
                    devices.append(GPUInfo(name=line.split(":", 2)[-1].strip()))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return devices


def detect_wsl() -> bool:
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


def detect_environment() -> ROCmEnvironment:
    import sys
    env = ROCmEnvironment()
    env.rocm_version = detect_rocm_version()
    env.gpu_devices = detect_gpu_devices()
    env.os_info = f"{platform.system()} {platform.release()}"
    env.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    env.is_wsl = detect_wsl()

    try:
        import torch
        env.torch_version = torch.__version__
        env.torch_cuda_available = torch.cuda.is_available()
        env.torch_rocm_available = hasattr(torch, "hip") and torch.cuda.is_available()
    except ImportError:
        env.torch_version = "Not Installed"

    return env


if __name__ == "__main__":
    env = detect_environment()
    print(env.summary())
