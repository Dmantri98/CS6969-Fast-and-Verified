"""
Test harness for auto-generated NKI matmul kernel.
Compares matmul_kernel_nki (from linalg-to-nki pipeline) against torch.matmul.
"""

import torch
from torch_xla.core import xla_model as xm

from matmul_kernel_generated import matmul_kernel_nki

if __name__ == "__main__":
    device = xm.xla_device()

    # Small workload
    M, K, N = 128, 64, 128
    lhs = torch.rand((M, K), dtype=torch.float32, device=device)
    rhs = torch.rand((K, N), dtype=torch.float32, device=device)

    output_nki = matmul_kernel_nki(lhs.T, rhs)
    output_torch = torch.matmul(lhs, rhs)

    print(f"Small test ({M}x{K} @ {K}x{N}):")
    if torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2):
        print("  PASS: NKI and Torch match")
    else:
        diff = (output_torch - output_nki).abs().max().item()
        print(f"  FAIL: max abs diff = {diff}")

    # Large workload
    M, K, N = 4096, 1024, 2048
    lhs = torch.rand((M, K), dtype=torch.float32, device=device)
    rhs = torch.rand((K, N), dtype=torch.float32, device=device)

    output_nki = matmul_kernel_nki(lhs.T, rhs)
    output_torch = torch.matmul(lhs, rhs)

    print(f"Large test ({M}x{K} @ {K}x{N}):")
    if torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2):
        print("  PASS: NKI and Torch match")
    else:
        diff = (output_torch - output_nki).abs().max().item()
        print(f"  FAIL: max abs diff = {diff}")
