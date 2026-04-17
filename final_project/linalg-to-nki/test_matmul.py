"""
Test suite for the auto-generated NKI matmul kernel.

Loads generated/matmul_kernel.py (produced by generate_kernels.py) and sweeps
a table of (M, K, N) shapes, comparing against torch.matmul. The emitted
kernel uses fixed TILE_M=128, TILE_N=512, TILE_K=128 and masks at every
load/store, so both aligned and ragged shapes -- including shapes smaller
than one tile -- must round-trip correctly.

Run:
    python generate_kernels.py        # once, to (re)build the kernel
    python test_matmul.py             # run the shape sweep
"""
import importlib.util
import sys
from pathlib import Path

import torch
from torch_xla.core import xla_model as xm


HERE = Path(__file__).resolve().parent
GEN_DIR = HERE / "generated"
KERNEL_PATH = GEN_DIR / "matmul_kernel.py"


# (M, K, N, description). Each row exercises a different masking regime.
CASES = [
    # Aligned to tile boundaries.
    (128,  128,  512,  "single tile, exact"),
    (256,  128,  512,  "2 M-tiles"),
    (128,  256,  512,  "2 K-tiles"),
    (128,  128,  1024, "2 N-tiles"),
    (512,  256,  1024, "multi-tile each axis"),

    # Ragged tails on one axis.
    (100,  128,  512,  "partial M (< TILE_M)"),
    (128,  128,  400,  "partial N (< TILE_N)"),
    (128,  100,  512,  "partial K (< TILE_K)"),

    # Ragged tails on every axis.
    (100,  63,   100,  "partial M, K, N (tiny)"),
    (200,  150,  300,  "partial M, K, N (small)"),
    (4090, 1020, 2040, "partial M, K, N (large; prior failing case)"),

    # Degenerate.
    (1,    1,    1,    "degenerate 1x1x1"),
    (1,    64,   1,    "degenerate 1xKx1"),
    (64,   1,    64,   "degenerate Mx1xN"),
]


def load_kernel():
    if not KERNEL_PATH.exists():
        raise FileNotFoundError(
            f"{KERNEL_PATH} missing -- run `python generate_kernels.py` first"
        )
    spec = importlib.util.spec_from_file_location(
        "generated_matmul_kernel", KERNEL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generated_matmul_kernel"] = mod
    spec.loader.exec_module(mod)
    return mod.matmul_kernel_nki


def run_one(kernel, M, K, N, tag, device):
    lhs = torch.rand((M, K), dtype=torch.float32, device=device)
    rhs = torch.rand((K, N), dtype=torch.float32, device=device)

    nki_out = kernel(lhs.T, rhs)
    torch_out = torch.matmul(lhs, rhs)

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
    print(f"=== NKI matmul_kernel on {device} ===")
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
