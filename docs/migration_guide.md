# CUDA to ROCm Migration Guide

## Overview

This guide covers migrating PyTorch models from NVIDIA CUDA to AMD ROCm.

## Key Differences

| CUDA Concept | ROCm Equivalent |
|-------------|----------------|
| cuDNN | MIOpen |
| NCCL | RCCL |
| CUDA Extension | HIP Extension |
| NVIDIA GPU | AMD GPU (MI250, MI300, etc.) |

## Step-by-Step Migration

### 1. Install ROCm + PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### 2. Check Compatibility

```python
from src.checker import ROCmChecker
checker = ROCmChecker(rocm_version="5.7")
result = checker.check_file("my_model.py")
print(result.summary())
```

### 3. Auto-Migrate

```python
from src.migrator import CUDAToROCmMigrator
migrator = CUDAToROCmMigrator()
migrator.migrate_file("my_model.py", "my_model_rocm.py")
```

### 4. Manual Changes

#### Custom CUDA Kernels
Replace custom CUDA kernels with HIP equivalents:
```bash
# Use hipify-perl or hipify-python
python -m pyrocm.tools.hipify --inplace my_extension/
```

#### NCCL to RCCL
```python
# Before
dist.init_process_group(backend='nccl')

# After - RCCL uses the same interface
dist.init_process_group(backend='nccl')  # RCCL intercepts automatically
```

#### cuDNN to MIOpen
```python
# MIOpen is used automatically - no code changes needed
torch.backends.cudnn.enabled = True  # Maps to MIOpen
```

### 5. Test on AMD Hardware

```bash
# Verify ROCm installation
rocm-smi

# Run tests
pytest tests/ -v
python scripts/check_model.py --model my_model
```

## Common Issues

- **Flash Attention**: Needs `flash-attn` compiled for ROCm
- **Custom ops**: May need CUDA→HIP translation
- **Docker**: Use `rocm/pytorch:latest` image
- **Multi-GPU**: RCCL handles communication; verify PCIe topology

## Performance Tips

- Use `torch.amp` for mixed precision (ROCm 5.7+)
- Enable MIOpen benchmark: `torch.backends.cudnn.benchmark = True`
- MI250X+ supports fp64 natively

## Resources

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Guide](https://pytorch.org/docs/stable/notes/hip.html)
- [AMD Developer Tools](https://developer.amd.com/)
