# Mamba SSM - MLX/MPS Compatible Version for macOS

This is a modified version of mamba_ssm that works on macOS without CUDA dependencies. It uses reference implementations (pure PyTorch) that are compatible with PyTorch's MPS backend for Apple Silicon.

## Installation

```bash
cd mamba_install_mps
pip install -e .
```

## Key Changes

1. **No CUDA Dependencies**: The setup.py has been modified to skip CUDA compilation on macOS
2. **Reference Implementations**: Uses pure PyTorch reference implementations instead of CUDA kernels
3. **MPS Compatible**: Works with PyTorch's Metal Performance Shaders (MPS) backend on Apple Silicon
4. **Automatic Fallback**: Automatically uses reference implementations when CUDA kernels are not available

## Usage

The API remains the same as the original mamba_ssm:

```python
import torch
from mamba_ssm import Mamba

# Works on macOS with MPS backend
device = "mps" if torch.backends.mps.is_available() else "cpu"
batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to(device)

model = Mamba(
    d_model=dim,
    d_state=16,
    d_conv=4,
    expand=2,
).to(device)

y = model(x)
assert y.shape == x.shape
```

## Performance Notes

- The reference implementations are slower than CUDA kernels but work on all platforms
- For best performance on macOS, use PyTorch with MPS backend
- The code automatically detects available backends and uses the best option

## Requirements

- Python >= 3.7
- PyTorch (with MPS support for Apple Silicon)
- einops
- transformers
- packaging

Optional:
- causal-conv1d (will use reference implementation if not available)

## Differences from Original

- No CUDA kernel compilation
- Uses `selective_scan_ref` and `mamba_inner_ref` instead of CUDA-accelerated versions
- Triton operations fall back to reference implementations on non-CUDA devices
- Setup.py skips CUDA build on macOS by default

