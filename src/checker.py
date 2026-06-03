"""Main ROCm compatibility checker."""
import ast
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from src.ops import get_op_compat, is_cuda_specific, CompatStatus, get_stats
from src.models import get_model_compat, MODEL_COMPAT_DB
from src.environment import detect_environment, ROCmEnvironment


@dataclass
class Issue:
    file: str
    line: int
    severity: str  # error, warning, info
    category: str
    message: str
    suggestion: str = ""


@dataclass
class CheckResult:
    target: str
    issues: List[Issue] = field(default_factory=list)
    compatible_ops: int = 0
    incompatible_ops: int = 0
    partial_ops: int = 0
    score: float = 0.0
    environment: Optional[ROCmEnvironment] = None
    model_info: Optional[Any] = None

    def add_issue(self, issue: Issue):
        self.issues.append(issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def calculate_score(self) -> float:
        total = self.compatible_ops + self.partial_ops + self.incompatible_ops
        if total == 0:
            return 100.0
        score = (self.compatible_ops * 1.0 + self.partial_ops * 0.5) / total * 100
        penalty = self.error_count * 5 + self.warning_count * 2
        self.score = max(0, min(100, score - penalty))
        return self.score

    def summary(self) -> str:
        lines = [
            f"Compatibility Report: {self.target}",
            "=" * 50,
            f"  Score:              {self.calculate_score():.1f}/100",
            f"  Compatible ops:     {self.compatible_ops}",
            f"  Partial ops:        {self.partial_ops}",
            f"  Incompatible ops:   {self.incompatible_ops}",
            f"  Errors:             {self.error_count}",
            f"  Warnings:           {self.warning_count}",
        ]
        if self.model_info:
            lines.append(f"  Model:              {self.model_info.name}")
            lines.append(f"  Compatible:         {'✅' if self.model_info.compatible else '❌'}")
        if self.issues:
            lines.append("\nIssues:")
            for issue in self.issues:
                icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "?")
                lines.append(f"  {icon} [{issue.category}] {issue.message}")
                if issue.suggestion:
                    lines.append(f"    → {issue.suggestion}")
        lines.append("=" * 50)
        return "\n".join(lines)


class ROCmChecker:
    def __init__(self, rocm_version: str = "5.7"):
        self.rocm_version = rocm_version
        self.environment = detect_environment()

    def check_model(self, model) -> CheckResult:
        result = CheckResult(target=str(type(model).__name__), environment=self.environment)
        model_name = type(model).__name__.lower()
        result.model_info = get_model_compat(model_name)

        if hasattr(model, "parameters"):
            for name, param in model.named_parameters():
                self._check_tensor_ops(name, result)
        if hasattr(model, "modules"):
            for name, module in model.named_modules():
                module_type = type(module).__name__
                compat = get_op_compat(f"torch.nn.{module_type}")
                if compat:
                    if compat.status == CompatStatus.FULL:
                        result.compatible_ops += 1
                    elif compat.status == CompatStatus.PARTIAL:
                        result.partial_ops += 1
                        result.issues.append(Issue(
                            file="model", line=0, severity="warning",
                            category="operator",
                            message=f"Module {module_type} has partial ROCm support",
                            suggestion=compat.notes or "Check ROCm docs for details",
                        ))
                    else:
                        result.incompatible_ops += 1
                        result.issues.append(Issue(
                            file="model", line=0, severity="error",
                            category="operator",
                            message=f"Module {module_type} not supported on ROCm",
                            suggestion=f"Use alternative: {compat.alternative}" if compat.alternative else "",
                        ))

        if not result.model_info or not result.model_info.compatible:
            result.issues.append(Issue(
                file="model", line=0, severity="warning",
                category="compatibility",
                message=f"Model '{model_name}' not in known compatible list",
                suggestion="Test thoroughly before production use",
            ))

        return result

    def check_file(self, filepath: str) -> CheckResult:
        result = CheckResult(target=filepath, environment=self.environment)
        with open(filepath) as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            if is_cuda_specific(line):
                if "nccl" in line.lower():
                    result.issues.append(Issue(
                        file=filepath, line=i, severity="error",
                        category="distributed",
                        message="NCCL backend detected (CUDA only)",
                        suggestion="Use RCCL backend: torch.distributed.init_process_group('gloo')",
                    ))
                elif "cudnn" in line.lower():
                    result.issues.append(Issue(
                        file=filepath, line=i, severity="warning",
                        category="backend",
                        message="cuDNN-specific code detected",
                        suggestion="MIOpen handles this automatically in most cases",
                    ))
                elif "cuda" in line.lower():
                    result.issues.append(Issue(
                        file=filepath, line=i, severity="info",
                        category="cuda",
                        message="CUDA-specific code detected",
                        suggestion="Review for ROCm compatibility",
                    ))

        try:
            tree = ast.parse("".join(lines))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    self._check_ast_call(node, filepath, result)
        except SyntaxError:
            result.issues.append(Issue(
                file=filepath, line=0, severity="warning",
                category="parse",
                message="Could not parse file as Python",
            ))

        return result

    def _check_ast_call(self, node, filepath, result):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                obj = node.func.value.id
                attr = node.func.attr
                if obj == "torch" and attr in ("cuda", "cuda_amp"):
                    result.issues.append(Issue(
                        file=filepath, line=getattr(node, "lineno", 0),
                        severity="warning", category="cuda",
                        message=f"torch.{obj} usage detected",
                        suggestion="Replace with ROCm equivalent",
                    ))

    def _check_tensor_ops(self, name, result):
        if any(kw in name.lower() for kw in ("cudnn", "cuda", "nccl")):
            result.issues.append(Issue(
                file="model", line=0, severity="warning",
                category="naming",
                message=f"Parameter name '{name}' suggests CUDA-specific code",
            ))

    def check_code_string(self, code: str) -> CheckResult:
        result = CheckResult(target="<code_string>", environment=self.environment)
        for i, line in enumerate(code.splitlines(), 1):
            if is_cuda_specific(line):
                result.issues.append(Issue(
                    file="<code>", line=i, severity="warning",
                    category="cuda", message=f"CUDA-specific: {line.strip()}",
                    suggestion="Review for ROCm compatibility",
                ))
        return result


def main():
    import click

    @click.command()
    @click.option("--model", help="Model name to check")
    @click.option("--file", help="Python file to scan")
    @click.option("--rocm-version", default="5.7", help="Target ROCm version")
    @click.option("--output", "-o", help="Output report file")
    def cli(model, file, rocm_version, output):
        checker = ROCmChecker(rocm_version=rocm_version)

        if model:
            try:
                import torch
                net = torch.hub.load("pytorch/vision", model, pretrained=True)
                result = checker.check_model(net)
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)
        elif file:
            result = checker.check_file(file)
        else:
            print("Provide --model or --file")
            sys.exit(1)

        print(result.summary())

        if output:
            from src.report import ReportGenerator
            gen = ReportGenerator()
            gen.add_result(result)
            if output.endswith(".json"):
                gen.generate_json(output)
            else:
                gen.generate_html(output)
            print(f"Report saved to {output}")

    cli()


if __name__ == "__main__":
    main()
