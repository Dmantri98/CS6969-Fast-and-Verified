"""
Test suite for auto-generated NKI matmul+relu kernels across tile configs.

Run:
    python generate_matmul_relu_kernels.py   # (re)build the kernels
    python test_matmul_relu.py               # run the shape sweep across configs
"""
import importlib.util
import sys
from pathlib import Path

import torch
from torch_xla.core import xla_model as xm


HERE = Path(__file__).resolve().parent
GEN_DIR = HERE / "generated_relu"


# Keep in sync with CONFIGS in generate_matmul_relu_kernels.py.
CONFIGS = [
    (64,  64,  64),
    (128, 128, 128),
    (64,  128, 64),
    (128, 128, 32),
]


CASES = [
    (64,   64,   64,   "all multiples, single tile"),
    (128,  128,  128,  "all multiples"),
    (256,  128,  512,  "rectangular"),
    (100,  64,   128,  "partial M only"),
    (128,  64,   100,  "partial N only"),
    (128,  100,  128,  "partial K only"),
    (200,  150,  300,  "partial M, K, N (small)"),
    (1,    1,    1,    "degenerate 1x1x1"),
]


def load_kernel(bm: int, bn: int, bk: int):
    tag = f"{bm}x{bn}x{bk}"
    path = GEN_DIR / f"matmul_relu_kernel_{tag}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python generate_matmul_relu_kernels.py` first"
        )
    mod_name = f"generated_matmul_relu_kernel_{tag}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.matmul_relu_kernel_nki


def run_one(kernel, M, K, N, tag, device):
    # Use signed values so relu actually clamps something to zero on the
    # comparison path.
    lhs = torch.rand((M, K), dtype=torch.float32, device=device) - 0.5
    rhs = torch.rand((K, N), dtype=torch.float32, device=device) - 0.5

    nki_out = kernel(lhs.T, rhs)
    torch_out = torch.matmul(lhs, rhs).clamp_min(0.0)

    label = f"({M:>5d} x {K:>5d} x {N:>5d})  {tag}"
    if torch.allclose(torch_out, nki_out, atol=1e-4, rtol=1e-2):
        print(f"    PASS  {label}")
        return True
    max_abs = (torch_out - nki_out).abs().max().item()
    mean_abs = (torch_out - nki_out).abs().mean().item()
    print(f"    FAIL  {label}   max|Δ|={max_abs:.4e}  mean|Δ|={mean_abs:.4e}")
    return False


def main():
    device = xm.xla_device()

    grand_pass = 0
    grand_total = 0
    failing_configs = []

    for (bm, bn, bk) in CONFIGS:
        tag = f"{bm}x{bn}x{bk}"
        print(f"\n=== Kernel BLOCK_SIZE={tag} on {device} ===")
        try:
            kernel = load_kernel(bm, bn, bk)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            failing_configs.append(tag)
            continue

        passed = 0
        for (M, K, N, case_tag) in CASES:
            if run_one(kernel, M, K, N, case_tag, device):
                passed += 1
        total = len(CASES)
        grand_pass += passed
        grand_total += total
        print(f"  -> {passed}/{total} passed for BLOCK_SIZE={tag}")
        if passed != total:
            failing_configs.append(tag)

    print()
    print(f"OVERALL: {grand_pass} / {grand_total} passed")
    if failing_configs:
        print("Configs with failures or skips:", ", ".join(failing_configs))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
