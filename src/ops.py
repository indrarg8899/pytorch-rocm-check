"""Operator compatibility matrix for ROCm."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class CompatStatus(Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass
class OpCompat:
    name: str
    status: CompatStatus
    min_rocm: str = "5.5"
    notes: str = ""
    alternative: str = ""


# ROCm operator compatibility database
OPERATOR_MATRIX: Dict[str, OpCompat] = {
    # Core tensor ops
    "torch.add": OpCompat("torch.add", CompatStatus.FULL, "5.5"),
    "torch.mul": OpCompat("torch.mul", CompatStatus.FULL, "5.5"),
    "torch.mm": OpCompat("torch.mm", CompatStatus.FULL, "5.5"),
    "torch.matmul": OpCompat("torch.matmul", CompatStatus.FULL, "5.5"),
    "torch.bmm": OpCompat("torch.bmm", CompatStatus.FULL, "5.5"),
    "torch.conv2d": OpCompat("torch.conv2d", CompatStatus.FULL, "5.5"),
    "torch.conv1d": OpCompat("torch.conv1d", CompatStatus.FULL, "5.5"),
    "torch.conv_transpose2d": OpCompat("torch.conv_transpose2d", CompatStatus.FULL, "5.5"),
    "torch.max_pool2d": OpCompat("torch.max_pool2d", CompatStatus.FULL, "5.5"),
    "torch.avg_pool2d": OpCompat("torch.avg_pool2d", CompatStatus.FULL, "5.5"),
    "torch.adaptive_avg_pool2d": OpCompat("torch.adaptive_avg_pool2d", CompatStatus.FULL, "5.5"),
    "torch.batch_norm": OpCompat("torch.batch_norm", CompatStatus.FULL, "5.5"),
    "torch.layer_norm": OpCompat("torch.layer_norm", CompatStatus.FULL, "5.5"),
    "torch.group_norm": OpCompat("torch.group_norm", CompatStatus.FULL, "5.5"),
    "torch.relu": OpCompat("torch.relu", CompatStatus.FULL, "5.5"),
    "torch.gelu": OpCompat("torch.gelu", CompatStatus.FULL, "5.5"),
    "torch.silu": OpCompat("torch.silu", CompatStatus.FULL, "5.5"),
    "torch.softmax": OpCompat("torch.softmax", CompatStatus.FULL, "5.5"),
    "torch.log_softmax": OpCompat("torch.log_softmax", CompatStatus.FULL, "5.5"),
    "torch.dropout": OpCompat("torch.dropout", CompatStatus.FULL, "5.5"),
    "torch.embedding": OpCompat("torch.embedding", CompatStatus.FULL, "5.5"),
    "torch.linear": OpCompat("torch.linear", CompatStatus.FULL, "5.5"),
    "torch.bmm": OpCompat("torch.bmm", CompatStatus.FULL, "5.5"),
    "torch.einsum": OpCompat("torch.einsum", CompatStatus.FULL, "5.6"),
    # Partial support
    "torch.fft": OpCompat("torch.fft", CompatStatus.PARTIAL, "5.6", "Some transforms may fail"),
    "torch.scatter_reduce": OpCompat("torch.scatter_reduce", CompatStatus.PARTIAL, "5.7"),
    "torch.index_put": OpCompat("torch.index_put", CompatStatus.PARTIAL, "5.7", "Accumulate mode issues"),
    "torch.unique": OpCompat("torch.unique", CompatStatus.PARTIAL, "5.6"),
    "torch.histc": OpCompat("torch.histc", CompatStatus.PARTIAL, "5.7"),
    "torch.lu": OpCompat("torch.lu", CompatStatus.PARTIAL, "5.7", "Numerical precision issues"),
    # Unsupported / problematic
    "torch.cudnn_convolution": OpCompat("torch.cudnn_convolution", CompatStatus.NONE, notes="cuDNN-specific", alternative="Use torch.conv2d with miopen backend"),
    "torch.cudnn_batch_norm": OpCompat("torch.cudnn_batch_norm", CompatStatus.NONE, notes="cuDNN-specific", alternative="torch.batch_norm"),
    "torch._C._nccl_all_reduce": OpCompat("torch._C._nccl_all_reduce", CompatStatus.NONE, notes="NCCL-specific", alternative="Use RCCL via torch.distributed"),
    "torch.cuda.amp.autocast": OpCompat("torch.cuda.amp.autocast", CompatStatus.PARTIAL, "5.7", "Mixed precision needs ROCm 5.7+", "torch.amp.autocast('cuda')"),
    # Flash Attention
    "torch.nn.functional.scaled_dot_product_attention": OpCompat("torch.nn.functional.scaled_dot_product_attention", CompatStatus.PARTIAL, "5.7", "Flash attention backend needs ROCm 6.0+"),
}

# Categorized ops
CUDA_SPECIFIC_PATTERNS = [
    "cudnn", "cublas", "cusparse", "cufft", "curand",
    "nccl", "cuda_kernel", "CUDAExtension",
]


def get_op_compat(op_name: str) -> Optional[OpCompat]:
    if op_name in OPERATOR_MATRIX:
        return OPERATOR_MATRIX[op_name]
    for key, val in OPERATOR_MATRIX.items():
        if op_name.endswith(key.split(".")[-1]):
            return val
    return None


def is_cuda_specific(code_line: str) -> bool:
    lower = code_line.lower()
    return any(p in lower for p in CUDA_SPECIFIC_PATTERNS)


def get_incompatible_ops() -> List[OpCompat]:
    return [op for op in OPERATOR_MATRIX.values() if op.status == CompatStatus.NONE]


def get_partial_ops() -> List[OpCompat]:
    return [op for op in OPERATOR_MATRIX.values() if op.status == CompatStatus.PARTIAL]


def get_stats() -> Dict[str, int]:
    counts = {"full": 0, "partial": 0, "none": 0, "unknown": 0}
    for op in OPERATOR_MATRIX.values():
        counts[op.status.value] += 1
    return counts
