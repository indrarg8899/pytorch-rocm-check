# Migration Guide

## Overview

This guide covers the process of migrating CUDA-based PyTorch projects to AMD ROCm.

## Prerequisites

- AMD Instinct GPU (MI100, MI210, MI250X, or MI300X)
- ROCm 5.6+ installed
- PyTorch built with HIP support

## Step 1: Scan Your Project

```bash
rocm-migrate scan /path/to/project
```

This identifies CUDA-specific patterns and suggests ROCm equivalents.

## Step 2: Review the Migration Plan

```bash
rocm-migrate plan /path/to/project --output plan.json
```

## Step 3: Apply Changes

```bash
# Dry run first
rocm-migrate apply /path/to/project --dry-run

# Apply for real
rocm-migrate apply /path/to/project --confirm
```

## Common Migration Patterns

### Tensor Device Placement

```python
# Before (CUDA)
tensor = tensor.cuda()
model = model.cuda()

# After (ROCm - works as-is or use .hip())
tensor = tensor.to("cuda")  # Works on ROCm
model = model.to("cuda")    # Works on ROCm
```

### CUDA API Calls

```cpp
// Before
cudaMalloc(&ptr, size);

// After
hipMalloc(&ptr, size);
```

### Kernel Launches

```cpp
// Before
myKernel<<<grid, block>>>(args);

// After (hipLaunchKernelGGL)
hipLaunchKernelGGL(myKernel, grid, block, 0, 0, args);
```

## Performance Optimization

- Use `rocprof` instead of `nsys` for profiling
- Enable `HSA_OVERRIDE_GFX_VERSION` for compatibility testing
- Use ROCm's `rccl` for multi-GPU communication

## Resources

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch ROCm Guide](https://pytorch.org/docs/stable/notes/rocm.html)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
