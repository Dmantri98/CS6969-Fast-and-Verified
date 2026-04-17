"""
Test suite for the auto-generated NKI matmul+relu kernel.

Loads generated_relu/matmul_relu_kernel.py (produced by
generate_matmul_relu_kernels.py) and sweeps (M, K, N) shapes comparing
against torch.matmul(...).clamp_min(0).

Run:
    python generate_matmul_relu_kernels.py   # (re)build the kernel
    python test_matmul_relu.py               # run the shape sweep
"""
import importlib.util
import sys
from pathlib import Path

import torch
from torch_xla.core import xla_model as xm


HERE = Path(__file__).resolve().parent
GEN_DIR = HERE / "generated_relu"
KERNEL_PATH = GEN_DIR / "matmul_relu_kernel.py"


# (M, K, N, description). Inputs are centered around zero so relu actually
# clamps ~half of the outputs -- a pure positive input would give the same
# answer as a plain matmul and hide relu bugs.
CASES = [
    (128,  128,  512,  "single tile, exact"),
    (256,  128,  512,  "2 M-tiles"),
    (128,  256,  512,  "2 K-tiles"),
    (128,  128,  1024, "2 N-tiles"),
    (100,  128,  512,  "partial M"),
    (128,  128,  400,  "partial N"),
    (128,  100,  512,  "partial K"),
    (200,  150,  300,  "partial M, K, N (small)"),
    (1,    1,    1,    "degenerate 1x1x1"),
]


def load_kernel():
    if not KERNEL_PATH.exists():
        raise FileNotFoundError(
            f"{KERNEL_PATH} missing -- run "
            "`python generate_matmul_relu_kernels.py` first"
        )
    spec = importlib.util.spec_from_file_location(
        "generated_matmul_relu_kernel", KERNEL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generated_matmul_relu_kernel"] = mod
    spec.loader.exec_module(mod)
    return mod.matmul_relu_kernel_nki


def run_one(kernel, M, K, N, tag, device):
    lhs = torch.rand((M, K), dtype=torch.float32, device=device) - 0.5
    rhs = torch.rand((K, N), dtype=torch.float32, device=device) - 0.5

    nki_out = kernel(lhs.T, rhs)
    torch_out = torch.matmul(lhs, rhs).clamp_min(0.0)

    label = f"({M:>5d} x {K:>5d} x {N:>5d})  {tag}"
    if torch.allclose(torch_out, nki_out, atol=1e-4, rtol=1e-2):
        print(f"  PASS  {label}")
        return True
    max_abs = (torch_out - nki_out).abs().max().item()
    mean_abs = (torch_out - nki_out).abs().mean().item()
    print(f"  FAIL  {label}   max|Δ|={max_abs:.4e}  mean|Δ|={mean_abs:.4e}")
    return False


def main():
    device = xm.xla_device()
    print(f"=== NKI matmul_relu_kernel on {device} ===")
    kernel = load_kernel()

    passed = 0
    for (M, K, N, case_tag) in CASES:
        if run_one(kernel, M, K, N, case_tag, device):
            passed += 1

    total = len(CASES)
    print(f"\nOVERALL: {passed} / {total} passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
