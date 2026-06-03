"""Shared utilities."""
import os
import sys
from typing import Optional


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def load_config(config_path: Optional[str] = None) -> dict:
    import yaml
    if config_path is None:
        config_path = os.path.join(get_project_root(), "configs", "default.yml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def version_tuple(version_str: str) -> tuple:
    parts = version_str.strip().split(".")
    return tuple(int(p) for p in parts if p.isdigit())


def check_rocm_minimum(current: str, required: str) -> bool:
    return version_tuple(current) >= version_tuple(required)
