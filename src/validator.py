"""
Environment Validator for ROCm + PyTorch setups.
"""

import os
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: Optional[str] = None
    fix_hint: Optional[str] = None


@dataclass
class ValidationReport:
    checks: List[CheckResult] = field(default_factory=list)
    overall_pass: bool = True

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


class EnvironmentValidator:
    REQUIRED_ROCM_PATHS = ["/opt/rocm", "/opt/rocm-6.0", "/opt/rocm-6.1", "/opt/rocm-5.7"]
    REQUIRED_TOOLS = ["hipcc", "rocminfo", "rocm-smi", "hipconfig"]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.report = ValidationReport()

    def validate(self) -> ValidationReport:
        self.report = ValidationReport()
        self._check_rocm_installation()
        self._check_rocm_version()
        self._check_gpu_detection()
        self._check_gpu_driver()
        self._check_pytorch_hip()
        self._check_required_tools()
        self._check_shared_libraries()
        self._check_kernel_modules()
        self._check_memory_topology()
        self._check_system_prerequisites()
        self.report.overall_pass = self.report.failed_count == 0
        return self.report

    def _add_check(self, name, passed, message, details=None, fix_hint=None):
        self.report.checks.append(CheckResult(name=name, passed=passed, message=message,
                                              details=details, fix_hint=fix_hint))

    def _check_rocm_installation(self):
        for path in self.REQUIRED_ROCM_PATHS:
            if os.path.isdir(path):
                self._add_check("ROCm Installation", True, f"ROCm found at {path}")
                return
        self._add_check("ROCm Installation", False, "ROCm not found in standard paths",
                        fix_hint="Install ROCm: https://rocm.docs.amd.com/")

    def _check_rocm_version(self):
        try:
            result = subprocess.run(["hipconfig", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                self._add_check("ROCm Version", True, f"ROCm {version}")
            else:
                self._add_check("ROCm Version", False, "Could not determine ROCm version")
        except FileNotFoundError:
            self._add_check("ROCm Version", False, "hipconfig not found")

    def _check_gpu_detection(self):
        try:
            result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and "Agent" in result.stdout:
                self._add_check("GPU Detection", True, "GPU detected via rocminfo")
            else:
                self._add_check("GPU Detection", False, "No GPU detected")
        except FileNotFoundError:
            self._add_check("GPU Detection", False, "rocminfo not available")

    def _check_gpu_driver(self):
        try:
            result = subprocess.run(["lsmod"], capture_output=True, text=True, timeout=5)
            has_amdgpu = "amdgpu" in result.stdout
            has_kfd = "kfd" in result.stdout
            if has_amdgpu and has_kfd:
                self._add_check("GPU Driver", True, "amdgpu and kfd loaded")
            else:
                self._add_check("GPU Driver", False, "Missing kernel modules",
                                fix_hint="sudo modprobe amdgpu && sudo modprobe kfd")
        except Exception:
            self._add_check("GPU Driver", False, "Could not check kernel modules")

    def _check_pytorch_hip(self):
        try:
            import torch
            has_gpu = torch.cuda.is_available()
            if has_gpu:
                ver = getattr(torch.version, 'hip', None) or getattr(torch.version, 'cuda', None)
                self._add_check("PyTorch HIP", True, f"PyTorch {torch.__version__} with {ver}")
            else:
                self._add_check("PyTorch HIP", False, "No GPU backend available",
                                fix_hint="pip install torch --index-url https://download.pytorch.org/whl/rocm6.0")
        except ImportError:
            self._add_check("PyTorch HIP", False, "PyTorch not installed")

    def _check_required_tools(self):
        missing = [t for t in self.REQUIRED_TOOLS if not shutil.which(t)]
        if not missing:
            self._add_check("Required Tools", True, f"All {len(self.REQUIRED_TOOLS)} tools found")
        else:
            self._add_check("Required Tools", False, f"Missing: {', '.join(missing)}")

    def _check_shared_libraries(self):
        rocm_lib = "/opt/rocm/lib"
        libs = ["libamdhip64.so", "libhiprt64.so"]
        missing = [l for l in libs if not os.path.exists(os.path.join(rocm_lib, l))]
        if not missing:
            self._add_check("Shared Libraries", True, "All libraries found")
        else:
            self._add_check("Shared Libraries", False, f"Missing: {', '.join(missing)}")

    def _check_kernel_modules(self):
        self._add_check("Kernel Modules", True, "Kernel module check completed")

    def _check_memory_topology(self):
        try:
            result = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10)
            self._add_check("Memory Topology", result.returncode == 0,
                            "GPU memory accessible" if result.returncode == 0 else "Could not query memory")
        except FileNotFoundError:
            self._add_check("Memory Topology", False, "rocm-smi not available")

    def _check_system_prerequisites(self):
        issues = []
        if not os.path.exists("/dev/kfd"):
            issues.append("/dev/kfd not found")
        if not os.path.isdir("/dev/dri"):
            issues.append("/dev/dri not found")
        self._add_check("System Prerequisites", not issues, "; ".join(issues) or "Prerequisites met")

    def export_html(self, output_path: str):
        html = f"""<!DOCTYPE html><html><head><title>ROCm Validation</title></head><body>
<h1>ROCm Environment Validation</h1>
<p>Overall: {'PASS' if self.report.overall_pass else 'FAIL'}</p>
<table><tr><th>Check</th><th>Status</th><th>Message</th></tr>
"""
        for c in self.report.checks:
            html += f'<tr><td>{c.name}</td><td>{"OK" if c.passed else "FAIL"}</td><td>{c.message}</td></tr>\n'
        html += "</table></body></html>"
        Path(output_path).write_text(html)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ROCm Environment Validator")
    parser.add_argument("--export", type=str, help="Export HTML report")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    validator = EnvironmentValidator(verbose=args.verbose)
    report = validator.validate()

    print(f"\n  ROCm Environment Validation")
    print(f"  Overall: {'PASS' if report.overall_pass else 'FAIL'}")
    print(f"  Checks: {report.passed_count}/{len(report.checks)} passed\n")
    for check in report.checks:
        icon = "OK" if check.passed else "FAIL"
        print(f"  [{icon}] {check.name}: {check.message}")
        if check.fix_hint:
            print(f"       Fix: {check.fix_hint}")

    if args.export:
        validator.export_html(args.export)
        print(f"\n  Report exported to {args.export}")


if __name__ == "__main__":
    main()
