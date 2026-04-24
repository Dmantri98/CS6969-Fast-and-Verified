"""
Benchmark: auto-generated *fused* matmul+relu kernels vs. an unfused
reference (nki-samples tiled matmul + standalone NKI relu).

This is the benchmark that most directly measures the win from the
-nki-fuse-activation pass: the unfused reference writes the matmul result
to HBM, reloads it into SBUF, applies relu, and writes it back. The
generated fused kernel keeps the PSUM resident and emits one
`nisa.activation(op=nl.relu, ...)` in place of the PSUM->SBUF
tensor_copy, saving the full HBM roundtrip.

Run:
    python bench/bench_matmul_relu.py
"""
import importlib.util
import sys
from pathlib import Path

import torch
from torch_xla.core import xla_model as xm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from common import BenchRow, render, run_case                    # noqa: E402
from reference_kernels import ref_matmul_relu_unfused            # noqa: E402


GEN_DIR = ROOT / "generated_relu"

CONFIGS = [
    (64,  64,  64),
    (128, 128, 128),
    (64,  128, 64),
    (128, 128, 32),
]

CASES = [
    (128,  128,  512,  "min tiled"),
    (256,  128,  512,  "M=2 tiles"),
    (128,  256,  512,  "K=2 tiles"),
    (128,  128, 1024,  "N=2 tiles"),
    (512,  512, 1024,  "medium"),
    (1024, 512, 2048,  "larger"),
]


def load_gen_kernel(bm: int, bn: int, bk: int):
    path = GEN_DIR / f"matmul_relu_kernel_{bm}x{bn}x{bk}.py"
    spec = importlib.util.spec_from_file_location(
        f"generated_matmul_relu_kernel_{bm}x{bn}x{bk}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.matmul_relu_kernel_nki


def main():
    device = xm.xla_device()
    print(f"device = {device}\n")

    rows: list[BenchRow] = []
    for (bm, bn, bk) in CONFIGS:
        tag_cfg = f"{bm}x{bn}x{bk}"
        try:
            gen_kernel = load_gen_kernel(bm, bn, bk)
        except FileNotFoundError:
            print(f"SKIP CFG={tag_cfg} "
                  f"(regenerate with generate_matmul_relu_kernels.py)")
            continue

        for (M, K, N, case_tag) in CASES:
            lhs = torch.rand((M, K), dtype=torch.float32, device=device) - 0.5
            rhs = torch.rand((K, N), dtype=torch.float32, device=device) - 0.5
            lhsT = lhs.T.contiguous()

            gt = torch.matmul(lhs, rhs).clamp_min(0.0)

            rows.append(run_case(
                bench="matmul_relu",
                config=f"CFG={tag_cfg}",
                shape=f"MKN={M}x{K}x{N}",
                case_tag=case_tag,
                gen_kernel=gen_kernel,
                ref_kernel=ref_matmul_relu_unfused,
                args=(lhsT, rhs),
                ground_truth=gt,
                atol=1e-4, rtol=1e-2,
            ))
            print(f"  ran CFG={tag_cfg}  MKN={M}x{K}x{N}  ({case_tag})")

    render(rows, bench_name="matmul_relu")


if __name__ == "__main__":
    main()
