"""
Benchmark: auto-generated vector-add kernels vs. the nki-samples 2D tiled
tensor_add reference.

The generated kernels are 1D (BLOCK_SIZE parameterized); the nki-samples
reference is 2D (128 x 512 tile). To compare on the same underlying data
we reshape the 1D flat buffer into (rows, cols) where rows % 128 == 0 and
cols == 512, feed the 2D shape to the reference, and feed the flat shape
to the generated kernel. Total element count is identical.

Run:
    python bench/bench_add.py
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
from reference_kernels import ref_tensor_add       # noqa: E402


GEN_DIR = ROOT / "generated_add"

# Which generated BLOCK_SIZE to benchmark. 1024 tends to be the sweet spot
# for our kernel (single BIG tile per step); pick a range to show the curve.
CONFIGS = [512, 1024, 2048]

# Shapes chosen so the reference (2D, 128x512-tiled) can run them. We pick
# element counts that factor as rows*512 with rows % 128 == 0.
# (elements, rows, cols, tag)
CASES = [
    (128 * 512,       128,  512, "1 tile (smallest ref shape)"),
    (256 * 512,       256,  512, "2 tiles"),
    (1024 * 512,     1024,  512, "medium"),
    (4096 * 512,     4096,  512, "large"),
]


def load_gen_kernel(block_size: int):
    path = GEN_DIR / f"add_kernel_{block_size}.py"
    spec = importlib.util.spec_from_file_location(
        f"generated_add_kernel_{block_size}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.add_kernel_nki


def main():
    device = xm.xla_device()
    print(f"device = {device}\n")

    print_header(["config/shape", "generated", "reference (nki-samples)"])
    for bs in CONFIGS:
        try:
            gen_kernel = load_gen_kernel(bs)
        except FileNotFoundError:
            print(f"SKIP BS={bs} (regenerate with generate_add_kernels.py)")
            continue

        for (n_elements, rows, cols, tag) in CASES:
            a2d = torch.rand((rows, cols), dtype=torch.float32, device=device)
            b2d = torch.rand((rows, cols), dtype=torch.float32, device=device)
            a1d = a2d.reshape(-1).contiguous()
            b1d = b2d.reshape(-1).contiguous()

            ref_out = (a2d + b2d)  # torch ground truth

            gen_res = compare(f"BS={bs}", gen_kernel, (a1d, b1d),
                              ref_out.reshape(-1), device)
            ref_res = compare(f"BS={bs}", ref_tensor_add, (a2d, b2d),
                              ref_out, device)

            print(fmt_row(f"BS={bs}  n={n_elements}  ({tag})",
                          gen_res, ref_res))
        print()


if __name__ == "__main__":
    main()
