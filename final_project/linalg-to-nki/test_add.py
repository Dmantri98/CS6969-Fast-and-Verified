"""
Test suite for auto-generated NKI vector-add kernels across BLOCK_SIZEs.

For each BLOCK_SIZE in CONFIGS, dynamic-imports the kernel file written by
generate_add_kernels.py (generated_add/add_kernel_<BS>.py) and sweeps a table
of n_elements, comparing against torch element-wise addition.

Run:
    python generate_add_kernels.py    # once, to (re)build the kernels
    python test_add.py                # run the shape sweep across configs
"""
import importlib.util
import sys
from pathlib import Path

import torch
from torch_xla.core import xla_model as xm


HERE = Path(__file__).resolve().parent
GEN_DIR = HERE / "generated_add"


# Keep in sync with CONFIGS in generate_add_kernels.py.
CONFIGS = [
    128,
    256,
    512,
    1024,
    2048,
]


CASES = [
    # (n_elements, tag)
    (128,     "single tile, exact"),
    (1024,    "single tile (for BS=1024)"),
    (4096,    "multiple full tiles"),
    (100_000, "large, multiple full tiles + partial"),

    (127,     "partial, < one tile"),
    (129,     "partial, 1 tile + 1"),
    (1023,    "partial, just under BS=1024"),
    (1025,    "partial, just over BS=1024"),
    (4097,    "partial, multi-tile + 1"),
    (99_999,  "partial, large"),

    (1,       "degenerate: 1 element"),
    (2,       "degenerate: 2 elements"),
]


def load_kernel(block_size: int):
    path = GEN_DIR / f"add_kernel_{block_size}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python generate_add_kernels.py` first"
        )
    mod_name = f"generated_add_kernel_{block_size}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.add_kernel_nki


def run_one(kernel, n, tag, device):
    x = torch.rand((n,), dtype=torch.float32, device=device)
    y = torch.rand((n,), dtype=torch.float32, device=device)
    out = torch.zeros((n,), dtype=torch.float32, device=device)

    nki_out = kernel(x, y, out)
    torch_out = x + y

    label = f"(n={n:>7d})  {tag}"
    if torch.allclose(torch_out, nki_out, atol=1e-5, rtol=1e-5):
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

    for block_size in CONFIGS:
        print(f"\n=== Kernel BLOCK_SIZE={block_size} on {device} ===")
        try:
            kernel = load_kernel(block_size)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            failing_configs.append(block_size)
            continue

        passed = 0
        for (n, tag) in CASES:
            if run_one(kernel, n, tag, device):
                passed += 1
        total = len(CASES)
        grand_pass += passed
        grand_total += total
        print(f"  -> {passed}/{total} passed for BLOCK_SIZE={block_size}")
        if passed != total:
            failing_configs.append(block_size)

    print()
    print(f"OVERALL: {grand_pass} / {grand_total} passed")
    if failing_configs:
        print("Configs with failures or skips:",
              ", ".join(str(c) for c in failing_configs))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
