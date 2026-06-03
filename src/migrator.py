"""CUDA to ROCm auto-migration tool."""
import re
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class MigrationRule:
    pattern: str
    replacement: str
    category: str
    description: str
    severity: str = "info"  # info, warning, auto_fix


# Migration rules: regex pattern -> replacement
MIGRATION_RULES: List[MigrationRule] = [
    # Device references
    MigrationRule(
        r"torch\.device\(['\"]cuda['\"]\)",
        "torch.device('cuda')  # ROCm uses 'cuda' as device name",
        "device", "ROCm uses 'cuda' device identifier",
    ),
    MigrationRule(
        r"torch\.cuda\.is_available\(\)",
        "torch.cuda.is_available()  # Works for ROCm too",
        "device", "is_available() works for both CUDA and ROCm",
    ),
    # cuDNN -> MIOpen
    MigrationRule(
        r"torch\.backends\.cudnn\.enabled",
        "torch.backends.cudnn.enabled  # Maps to MIOpen on ROCm",
        "backend", "cuDNN maps to MIOpen automatically",
    ),
    MigrationRule(
        r"torch\.backends\.cudnn\.benchmark",
        "torch.backends.cudnn.benchmark  # MIOpen benchmark on ROCm",
        "backend", "cuDNN benchmark maps to MIOpen",
    ),
    # NCCL -> RCCL
    MigrationRule(
        r"nccl",
        "nccl  # WARNING: Use RCCL on ROCm",
        "distributed", "NCCL → RCCL migration needed",
    ),
    # CUDA extensions
    MigrationRule(
        r"from\s+torch\.utils\.cpp_extension\s+import\s+CUDAExtension",
        "from torch.utils.hip_extension import HIPExtension",
        "extension", "CUDAExtension → HIPExtension",
        "warning",
    ),
    MigrationRule(
        r"CUDAExtension\(",
        "HIPExtension(",
        "extension", "CUDAExtension → HIPExtension",
        "warning",
    ),
    # CUDA kernels
    MigrationRule(
        r"\.cuda\(\)",
        ".cuda()  # Works on ROCm",
        "device", ".cuda() works on ROCm",
    ),
    # Memory management
    MigrationRule(
        r"torch\.cuda\.empty_cache\(\)",
        "torch.cuda.empty_cache()  # Works on ROCm",
        "memory", "Memory management compatible",
    ),
    MigrationRule(
        r"torch\.cuda\.max_memory_allocated\(\)",
        "torch.cuda.max_memory_allocated()  # Works on ROCm",
        "memory", "Memory tracking compatible",
    ),
    # Mixed precision
    MigrationRule(
        r"torch\.cuda\.amp\.autocast",
        "torch.cuda.amp.autocast  # ROCm 5.7+ recommended",
        "amp", "AMP supported on ROCm 5.7+",
    ),
    # Compilation
    MigrationRule(
        r"torch\.compile\b",
        "torch.compile  # ROCm 5.7+ has partial torch.compile support",
        "compile", "torch.compile support improving in ROCm 5.7+",
    ),
]


class CUDAToROCmMigrator:
    def __init__(self):
        self.rules = MIGRATION_RULES
        self.changes: List[Dict] = []
        self.stats = {"total_changes": 0, "by_category": {}}

    def migrate_content(self, content: str) -> str:
        self.changes = []
        result = content

        for rule in self.rules:
            matches = list(re.finditer(rule.pattern, result))
            if matches:
                for match in reversed(matches):
                    start, end = match.span()
                    original = match.group(0)
                    replaced = re.sub(rule.pattern, rule.replacement, original)
                    if original != replaced:
                        result = result[:start] + replaced + result[end:]
                        self.changes.append({
                            "category": rule.category,
                            "description": rule.description,
                            "original": original.strip(),
                            "replaced": replaced.strip(),
                            "severity": rule.severity,
                        })
                        self.stats["total_changes"] += 1
                        self.stats["by_category"][rule.category] = (
                            self.stats["by_category"].get(rule.category, 0) + 1
                        )

        return result

    def migrate_file(self, input_path: str, output_path: str) -> None:
        with open(input_path) as f:
            content = f.read()

        migrated = self.migrate_content(content)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(migrated)

    def get_report(self) -> str:
        lines = [
            "CUDA → ROCm Migration Report",
            "=" * 50,
            f"Total changes: {self.stats['total_changes']}",
        ]
        for cat, count in self.stats["by_category"].items():
            lines.append(f"  {cat}: {count} changes")
        lines.append("")
        for change in self.changes:
            icon = {"auto_fix": "✅", "warning": "⚠️", "info": "ℹ️"}.get(change["severity"], "?")
            lines.append(f"{icon} [{change['category']}] {change['description']}")
            lines.append(f"  - {change['original']}")
            lines.append(f"  + {change['replaced']}")
        lines.append("=" * 50)
        return "\n".join(lines)


def main():
    import click

    @click.command()
    @click.option("--input", "-i", "input_file", required=True, help="Input CUDA file")
    @click.option("--output", "-o", "output_file", required=True, help="Output ROCm file")
    @click.option("--dry-run", is_flag=True, help="Show changes without writing")
    def cli(input_file, output_file, dry_run):
        migrator = CUDAToROCmMigrator()
        if dry_run:
            with open(input_file) as f:
                content = f.read()
            migrator.migrate_content(content)
        else:
            migrator.migrate_file(input_file, output_file)
        print(migrator.get_report())

    cli()


if __name__ == "__main__":
    main()
