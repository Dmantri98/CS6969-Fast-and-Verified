"""
Hardware-side numerical check for the block-size sweep.

Loads every matmul_kernel_{BM}_{BN}_{BK}.py under generated_sweep/ (produced
offline by generate_sweep_kernels.py) and runs a short shape sweep against
torch.matmul on the XLA device. Meant to be run on the Trainium host after
pulling the committed generated_sweep/ tree.

Run (on Trainium):
    python test_sweep_kernels.py
"""
import importlib.util
import re
import sys
from pathlib import Path

import torch
from torch_xla.core import xla_model as xm


HERE = Path(__file__).resolve().parent
GEN_DIR = HERE / "generated_sweep"

KERNEL_RE = re.compile(r"matmul_kernel_(\d+)_(\d+)_(\d+)\.py$")


# Problem shapes: sanity-tier sweep (not full coverage -- test_matmul.py
# already exhausts the canonical config).
SHAPES = [
    (128, 128, 512, "aligned"),
    (100, 128, 400, "ragged M,N"),
    (200, 150, 300, "ragged all"),
]


def discover_kernels():
    """Return [(bm, bn, bk, py_path)] sorted by (bm, bn, bk)."""
    out = []
    for p in GEN_DIR.glob("matmul_kernel_*_*_*.py"):
        m = KERNEL_RE.search(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), p))
    out.sort()
    return out


def load_kernel(py_path: Path, unique_name: str):
    spec = importlib.util.spec_from_file_location(unique_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod.matmul_kernel_nki


def run_one(kernel, M, K, N, device):
    lhs = torch.rand((M, K), dtype=torch.float32, device=device)
    rhs = torch.rand((K, N), dtype=torch.float32, device=device)
    out = kernel(lhs.T, rhs)
    ref = torch.matmul(lhs, rhs)
    if torch.allclose(ref, out, atol=1e-4, rtol=1e-2):
        return True, ""
    diff = (ref - out).abs()
    return False, (
        f"max|Δ|={diff.max().item():.2e}  mean|Δ|={diff.mean().item():.2e}"
    )


def main():
    if not GEN_DIR.exists():
        raise SystemExit(
            f"{GEN_DIR} missing -- run `python generate_sweep_kernels.py` on a "
            "dev host with triton, then pull the generated tree here."
        )

    kernels = discover_kernels()
    if not kernels:
        raise SystemExit(f"no matmul_kernel_*.py found under {GEN_DIR}")

    device = xm.xla_device()
    print(f"=== Sweep hardware check on {device} "
          f"({len(kernels)} kernels × {len(SHAPES)} shapes) ===\n")

    total_ok = 0
    total = 0
    for (bm, bn, bk, py_path) in kernels:
        cfg = f"BLOCK=({bm:>3d},{bn:>3d},{bk:>3d})"
        try:
            kernel = load_kernel(py_path, f"swept_{bm}_{bn}_{bk}")
        except Exception as e:
            print(f"  LOAD-FAIL  {cfg}  {type(e).__name__}: {e}")
            continue
        for (M, K, N, tag) in SHAPES:
            total += 1
            label = f"{cfg}  ({M:>4d} x {K:>4d} x {N:>4d})  {tag}"
            try:
                ok, detail = run_one(kernel, M, K, N, device)
            except Exception as e:
                print(f"  ERR   {label}   {type(e).__name__}: {e}")
                continue
            if ok:
                total_ok += 1
                print(f"  PASS  {label}")
            else:
                print(f"  FAIL  {label}   {detail}")

    print(f"\nOVERALL: {total_ok} / {total} passed")
    if total_ok != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
