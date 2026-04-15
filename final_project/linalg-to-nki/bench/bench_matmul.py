"""
Benchmark: auto-generated matmul kernels vs. the nki-samples tiled matmul
reference.

Both kernels take lhsT (K, M) and rhs (K, N) and return (M, N). The
nki-samples reference requires M%128 == 0, K%128 == 0, N%512 == 0, so we
restrict the sweep to shapes that satisfy that (our generated kernels
accept arbitrary shapes, but a fair comparison needs both to run).

Run:
    python bench/bench_matmul.py
"""
import importlib.util
import sys
from pathlib import Path

import torch
from torch_xla.core import xla_model as xm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from common import compare, fmt_row, print_header  # noqa: E402
from reference_kernels import ref_matmul_tiled     # noqa: E402


GEN_DIR = ROOT / "generated"

CONFIGS = [
    (64,  64,  64),
    (128, 128, 128),
    (64,  128, 64),
    (128, 128, 32),
]

# (M, K, N, tag) -- all 128/128/512-aligned for the nki-samples reference.
CASES = [
    (128,  128,  512,  "minimum tiled"),
    (256,  128,  512,  "M=2 tiles"),
    (128,  256,  512,  "K=2 tiles"),
    (128,  128, 1024,  "N=2 tiles"),
    (512,  512, 1024,  "medium square-ish"),
    (1024, 512, 2048,  "larger"),
]


def load_gen_kernel(bm: int, bn: int, bk: int):
    path = GEN_DIR / f"matmul_kernel_{bm}x{bn}x{bk}.py"
    spec = importlib.util.spec_from_file_location(
        f"generated_matmul_kernel_{bm}x{bn}x{bk}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.matmul_kernel_nki


def main():
    device = xm.xla_device()
    print(f"device = {device}\n")

    print_header(["config/shape", "generated", "reference (nki-samples)"])
    for (bm, bn, bk) in CONFIGS:
        tag_cfg = f"{bm}x{bn}x{bk}"
        try:
            gen_kernel = load_gen_kernel(bm, bn, bk)
        except FileNotFoundError:
            print(f"SKIP CFG={tag_cfg} (regenerate with generate_kernels.py)")
            continue

        for (M, K, N, case_tag) in CASES:
            lhs = torch.rand((M, K), dtype=torch.float32, device=device) - 0.5
            rhs = torch.rand((K, N), dtype=torch.float32, device=device) - 0.5
            lhsT = lhs.T.contiguous()

            ref_out = torch.matmul(lhs, rhs)

            gen_res = compare(f"CFG={tag_cfg}", gen_kernel, (lhsT, rhs),
                              ref_out, device)
            ref_res = compare(f"CFG={tag_cfg}", ref_matmul_tiled, (lhsT, rhs),
                              ref_out, device)

            print(fmt_row(f"CFG={tag_cfg}  MKN={M}x{K}x{N}  ({case_tag})",
                          gen_res, ref_res))
        print()


if __name__ == "__main__":
    main()
