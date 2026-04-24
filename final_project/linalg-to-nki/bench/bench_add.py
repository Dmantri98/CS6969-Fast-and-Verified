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

from common import BenchRow, render, run_case                   # noqa: E402
from reference_kernels import ref_tensor_add                    # noqa: E402


GEN_DIR = ROOT / "generated_add"

CONFIGS = [512, 1024, 2048]

# (elements, rows, cols, tag) -- rows % 128 == 0, cols == 512 so the ref fits.
CASES = [
    (128 * 512,       128,  512, "1 tile"),
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

    rows: list[BenchRow] = []
    for bs in CONFIGS:
        try:
            gen_kernel = load_gen_kernel(bs)
        except FileNotFoundError:
            print(f"SKIP BS={bs} (regenerate with generate_add_kernels.py)")
            continue

        for (n_elements, nrows, ncols, tag) in CASES:
            a2d = torch.rand((nrows, ncols), dtype=torch.float32, device=device)
            b2d = torch.rand((nrows, ncols), dtype=torch.float32, device=device)
            a1d = a2d.reshape(-1).contiguous()
            b1d = b2d.reshape(-1).contiguous()

            gt2d = a2d + b2d  # torch ground truth, 2D

            # gen: (1D, 1D) -> 1D; wrap so the bench sees a 2D output for
            # ground-truth comparison.
            def gen_2d(a, b, _g=gen_kernel, _s=gt2d.shape):
                return _g(a.reshape(-1), b.reshape(-1)).reshape(_s)

            rows.append(run_case(
                bench="add",
                config=f"BS={bs}",
                shape=f"n={n_elements}",
                case_tag=tag,
                gen_kernel=gen_2d,
                ref_kernel=ref_tensor_add,
                args=(a2d, b2d),
                ground_truth=gt2d,
                atol=1e-5, rtol=1e-5,
            ))
            print(f"  ran BS={bs}  n={n_elements}  ({tag})")

    render(rows, bench_name="add")


if __name__ == "__main__":
    main()
