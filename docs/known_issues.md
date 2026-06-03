# Known Issues

## ROCm 5.7

### Flash Attention
- `torch.nn.functional.scaled_dot_product_attention` with flash backend requires ROCm 6.0+
- Workaround: Use `math` backend or `sdpa_math` kernel

### Mixed Precision
- `torch.cuda.amp.autocast` partially supported
- Workaround: Use `torch.amp.autocast('cuda')` (PyTorch 2.0+)

### Distributed Training
- NCCL backend automatically intercepted by RCCL
- Some NCCL-only features may not be available
- Workaround: Use Gloo backend for CPU communication

## ROCm 5.6

### FFT Issues
- Some 3D FFT transforms may produce incorrect results
- Workaround: Use 2D transforms or CPU fallback

### Index Operations
- `index_put` with `accumulate=True` has precision issues
- Workaround: Use manual accumulation pattern

## ROCm 5.5

### Limited Model Support
- Many transformer models require ROCm 5.6+
- Workaround: Upgrade to ROCm 5.7+

### Memory Management
- Larger models may see higher memory usage vs CUDA
- Workaround: Increase batch size carefully; monitor with `rocm-smi`

## General Issues

### Docker/GPU Access
- Ensure `/dev/kfd` and `/dev/dri` are accessible in containers
- Use `--device=/dev/kfd --device=/dev/dri --group-add video` docker flags

### Build Errors
- Custom CUDA extensions need `hipify-perl` translation
- Use `TORCH_USE_HIP_DSA=1` for debugging

### Multi-GPU
- Verify RCCL topology with `rocm-smi --showtopo`
- PCIe bandwidth affects inter-GPU communication

## Reporting Issues

If you encounter a bug:
1. Run `python -m src.environment` and include output
2. Include ROCm version, GPU model, PyTorch version
3. Provide minimal reproduction code
4. Open issue at: https://github.com/indrarg8899/pytorch-rocm-check/issues
