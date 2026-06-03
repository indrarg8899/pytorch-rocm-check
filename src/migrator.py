"""
CUDA to ROCm Migration Scanner.

Scans Python/C++ source files for CUDA-specific patterns and
generates ROCm/HIP migration suggestions with code patches.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class MigrationIssue:
    """A single migration issue found in source code."""
    file: str
    line: int
    column: int
    category: str
    severity: str
    description: str
    original_code: str
    suggested_fix: Optional[str] = None
    rocm_equivalent: Optional[str] = None


@dataclass
class MigrationReport:
    """Full migration scan report."""
    project_path: str
    files_scanned: int = 0
    issues: List[MigrationIssue] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


CUDA_TO_HIP_MAPPINGS = {
    "cudaMalloc": "hipMalloc",
    "cudaFree": "hipFree",
    "cudaMemcpy": "hipMemcpy",
    "cudaMemcpyAsync": "hipMemcpyAsync",
    "cudaMemcpyToSymbol": "hipMemcpyToSymbol",
    "cudaMemcpyFromSymbol": "hipMemcpyFromSymbol",
    "cudaMemset": "hipMemset",
    "cudaMemsetAsync": "hipMemsetAsync",
    "cudaGetDevice": "hipGetDevice",
    "cudaSetDevice": "hipSetDevice",
    "cudaGetDeviceProperties": "hipGetDeviceProperties",
    "cudaDeviceGetAttribute": "hipDeviceGetAttribute",
    "cudaGetDeviceCount": "hipGetDeviceCount",
    "cudaDeviceSynchronize": "hipDeviceSynchronize",
    "cudaGetLastError": "hipGetLastError",
    "cudaPeekAtLastError": "hipPeekAtLastError",
    "cudaGetErrorString": "hipGetErrorString",
    "cudaGetErrorName": "hipGetErrorName",
    "cudaStreamCreate": "hipStreamCreate",
    "cudaStreamDestroy": "hipStreamDestroy",
    "cudaStreamSynchronize": "hipStreamSynchronize",
    "cudaStreamCreateWithFlags": "hipStreamCreateWithFlags",
    "cudaEventCreate": "hipEventCreate",
    "cudaEventRecord": "hipEventRecord",
    "cudaEventSynchronize": "hipEventSynchronize",
    "cudaEventElapsedTime": "hipEventElapsedTime",
    "cudaMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "cudaMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "cudaLaunchKernel": "hipModuleLaunchKernel",
    "cudaConfigureCall": "hipModuleConfigureCall",
    "cudaSuccess": "hipSuccess",
    "cudaErrorNotReady": "hipErrorNotReady",
    "cudaErrorMemoryAllocation": "hipErrorOutOfMemory",
    "cuda_fp16.h": "hip_fp16.h",
    "cuda_bf16.h": "hip/bf16.h",
    "cublasCreate": "rocblas_create_handle",
    "cublasDestroy": "rocblas_destroy_handle",
    "cublasSgemm": "rocblas_sgemm",
    "cublasDgemm": "rocblas_dgemm",
    "cudnnCreate": "miopenCreate",
    "cudnnDestroy": "miopenDestroy",
}

PYTHON_CUDA_PATTERNS = [
    (r'\.cuda\(\)', "GPU tensor placement", "Use `.to(device)` or `.hip()` for ROCm"),
    (r'torch\.cuda\.', "CUDA runtime call", "Replace with `torch.hip.` equivalent"),
    (r'torch\.cuda\.is_available\(\)', "CUDA availability check",
     "Use `torch.cuda.is_available()` (works on ROCm) or `torch.version.hip`"),
    (r'torch\.backends\.cuda', "CUDA backend config", "Check ROCm backend compatibility"),
    (r'torch\.cuda\.amp', "CUDA AMP", "Use `torch.amp` (cross-platform)"),
    (r'nccl', "NCCL backend", "RCCL is used on ROCm — usually transparent"),
]


class MigrationScanner:
    """Scans project for CUDA-specific code and suggests ROCm equivalents."""

    SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist"}
    CPP_EXTENSIONS = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".c"}
    PYTHON_EXTENSIONS = {".py"}

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.issues: List[MigrationIssue] = []

    def scan(self) -> MigrationReport:
        report = MigrationReport(project_path=str(self.project_path))
        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
            for filename in files:
                filepath = Path(root) / filename
                ext = filepath.suffix.lower()
                if ext in self.CPP_EXTENSIONS:
                    report.files_scanned += 1
                    self._scan_cpp_file(filepath, report)
                elif ext in self.PYTHON_EXTENSIONS:
                    report.files_scanned += 1
                    self._scan_python_file(filepath, report)
        report.issues = self.issues
        report.summary = {
            "total_issues": len(self.issues),
            "errors": report.error_count,
            "warnings": report.warning_count,
            "files_scanned": report.files_scanned,
        }
        return report

    def _scan_cpp_file(self, filepath: Path, report: MigrationReport):
        try:
            content = filepath.read_text(errors="replace")
        except Exception:
            return
        for line_num, line in enumerate(content.splitlines(), 1):
            if "#include" in line and "cuda" in line.lower():
                suggested = line.replace("cuda", "hip").replace("cublas", "rocblas")
                self.issues.append(MigrationIssue(
                    file=str(filepath), line=line_num, column=0,
                    category="include", severity="warning",
                    description="CUDA include detected",
                    original_code=line.strip(),
                    suggested_fix=suggested.strip(),
                ))
            for cuda_api, hip_api in CUDA_TO_HIP_MAPPINGS.items():
                if cuda_api in line:
                    self.issues.append(MigrationIssue(
                        file=str(filepath), line=line_num,
                        column=line.find(cuda_api),
                        category="api_call", severity="error",
                        description=f"CUDA API `{cuda_api}` -> `{hip_api}`",
                        original_code=line.strip(),
                        suggested_fix=line.replace(cuda_api, hip_api).strip(),
                        rocm_equivalent=hip_api,
                    ))

    def _scan_python_file(self, filepath: Path, report: MigrationReport):
        try:
            content = filepath.read_text(errors="replace")
        except Exception:
            return
        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern, description, suggestion in PYTHON_CUDA_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    severity = "info" if "check" in description.lower() else "warning"
                    self.issues.append(MigrationIssue(
                        file=str(filepath), line=line_num,
                        column=match.start(),
                        category="python_cuda", severity=severity,
                        description=description,
                        original_code=line.strip(),
                        suggested_fix=suggestion,
                    ))


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="CUDA->ROCm Migration Scanner")
    parser.add_argument("action", choices=["scan", "plan", "apply"])
    parser.add_argument("path", type=str, help="Project directory to scan")
    parser.add_argument("--output", type=str, help="Output file for plan")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--format", choices=["json", "text"], default="text")
    args = parser.parse_args()

    scanner = MigrationScanner(args.path)
    report = scanner.scan()

    if args.format == "json":
        output = {
            "project_path": report.project_path,
            "summary": report.summary,
            "issues": [
                {"file": i.file, "line": i.line, "column": i.column,
                 "category": i.category, "severity": i.severity,
                 "description": i.description, "original_code": i.original_code,
                 "suggested_fix": i.suggested_fix, "rocm_equivalent": i.rocm_equivalent}
                for i in report.issues
            ],
        }
        if args.output:
            Path(args.output).write_text(json.dumps(output, indent=2))
        else:
            print(json.dumps(output, indent=2))
    else:
        print(f"\n  CUDA->ROCm Migration Report")
        print(f"  Project: {report.project_path}")
        print(f"  Files scanned: {report.files_scanned}")
        print(f"  Issues found: {report.summary['total_issues']}")
        for issue in report.issues:
            icon = "ERR" if issue.severity == "error" else "WRN" if issue.severity == "warning" else "INF"
            print(f"  [{icon}] {issue.file}:{issue.line} - {issue.description}")
            if issue.suggested_fix:
                print(f"       Fix: {issue.suggested_fix}")


if __name__ == "__main__":
    main()
