# Compatibility Matrix

## Operator Support

### Full Support ✅
- Element-wise ops (add, mul, div, etc.)
- Matrix operations (mm, bmm, matmul)
- Convolution ops (conv1d, conv2d, conv_transpose)
- Pooling ops (max_pool, avg_pool, adaptive_avg_pool)
- Normalization (batch_norm, layer_norm, group_norm)
- Activation (relu, gelu, silu, softmax, log_softmax)
- Dropout, embedding, linear
- einsum (ROCm 5.6+)

### Partial Support ⚠️
- FFT transforms (some transforms may fail on older ROCm)
- scatter_reduce (ROCm 5.7+)
- index_put accumulate mode
- unique (ROCm 5.6+)
- LU decomposition (numerical precision)
- torch.cuda.amp.autocast (ROCm 5.7+)
- Scaled dot product attention (flash attn needs ROCm 6.0+)

### Not Supported ❌
- cuDNN-specific ops (cudnn_convolution, cudnn_batch_norm)
- NCCL direct calls (use RCCL through distributed API)
- Custom CUDA kernels without HIP port

## Model Compatibility

| Model | Compatible | Min ROCm | Notes |
|-------|-----------|----------|-------|
| ResNet 18/34/50/101/152 | ✅ | 5.5 | Full support |
| VGG 16/19 | ✅ | 5.5 | Full support |
| DenseNet 121 | ✅ | 5.5 | Full support |
| MobileNet V2/V3 | ✅ | 5.5 | Full support |
| EfficientNet B0-B4 | ✅ | 5.6 | Full support |
| ViT B/L | ✅ | 5.7 | Full support |
| Swin Transformer | ✅ | 5.7 | Full support |
| BERT Base/Large | ✅ | 5.6 | Full support |
| GPT-2 | ✅ | 5.7 | Full support |
| LLaMA 7B/13B | ✅ | 5.7 | flash-attn workaround |
| LLaMA 70B | ✅ | 6.0 | Multi-GPU required |
| Mistral 7B | ✅ | 5.7 | Full support |
| T5 Small/Base | ✅ | 5.6 | Full support |
| Stable Diffusion v1.5 | ✅ | 5.7 | Half precision recommended |
| SDXL | ✅ | 6.0 | Full support |
| Wav2Vec2 | ❌ | - | Custom CUDA kernels |
| Deformable DETR | ❌ | - | Custom CUDA ops |

## Hardware Compatibility

| GPU | ROCm 5.5 | 5.6 | 5.7 | 6.0 |
|-----|----------|-----|-----|-----|
| MI100 | ✅ | ✅ | ✅ | ✅ |
| MI210 | ✅ | ✅ | ✅ | ✅ |
| MI250 | ✅ | ✅ | ✅ | ✅ |
| MI250X | ✅ | ✅ | ✅ | ✅ |
| MI300X | - | - | ✅ | ✅ |
| RX 7900 XTX | ✅ | ✅ | ✅ | ✅ |
