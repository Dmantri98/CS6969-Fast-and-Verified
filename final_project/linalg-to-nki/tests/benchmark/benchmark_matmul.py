"""
Benchmark the emitted matmul kernel against the AWS nki-samples
reference kernels on a shared set of shapes.

We use `neuronxcc.nki.benchmark`, which compiles each kernel to a NEFF
once and times pure on-device execute. This sidesteps torch_xla tracing
/ HLO-caching issues (the nki-samples kernels produce a fresh HLO hash
on every call through torch_xla, so every torch_xla call paid a full
neuronxcc compile -- what we were measuring there was compile time, not
device time).

Loaded kernels
--------------
  emitted    : generated/matmul_kernel.py  (our pipeline output)
  tiled      : nki_matmul_tiled_
  hoist_load : nki_matmul_hoist_load_
  block_free : nki_matmul_block_free_dimension_
  fully_opt  : nki_matmul_fully_optimized_  (default TILES_IN_BLOCK)

nki_matmul_basic_ is intentionally skipped: it is hardcoded to a
single 64x128x512 matmul and is not comparable across shapes.

Each reference kernel has its own alignment constraints; for each
shape we only run the kernels whose constraints are satisfied. The
emitted kernel masks at every load/store and runs on every shape.

Reference file path
-------------------
Default:
  /home/ubuntu/nki-samples/src/nki_samples/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py

Override:
  NKI_SAMPLES_PATH=/abs/path/to/matrix_multiplication_nki_kernels.py \
      python tests/benchmark/benchmark_matmul.py

Run (on Trainium):
    python tests/benchmark/benchmark_matmul.py
"""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch_xla
from torch_xla.core import xla_model as xm
from neuronxcc.nki import benchmark


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
EMITTED_PATH = PROJECT_ROOT / "generated" / "matmul_kernel.py"

DEFAULT_REF_PATH = Path(
    "/home/ubuntu/nki-samples/src/nki_samples/tutorials/"
    "matrix_multiplication/matrix_multiplication_nki_kernels.py"
)
REF_PATH = Path(os.environ.get("NKI_SAMPLES_PATH", str(DEFAULT_REF_PATH)))


# (attr name in reference module, short label, (M, K, N) -> bool).
# Predicates match the asserts at the top of each kernel in
# nki-samples/src/nki_samples/tutorials/matrix_multiplication/
# matrix_multiplication_nki_kernels.py:
#   tiled            lines 96-98   : M%128,   N%512,  K%128
#   hoist_load       lines 164-166 : M%128,   N%512,  K%128
#   block_free_dim   lines 251-252 : M%256,   N%1024  (K%128 implicit)
#   fully_optimized  lines 365-367 : M%2048,  N%1024, K%1024  (defaults)
REF_KERNELS = [
    ("nki_matmul_tiled_", "tiled",
     lambda M, K, N: M % 128 == 0 and K % 128 == 0 and N % 512 == 0),
    ("nki_matmul_hoist_load_", "hoist_load",
     lambda M, K, N: M % 128 == 0 and K % 128 == 0 and N % 512 == 0),
    ("nki_matmul_block_free_dimension_", "block_free",
     lambda M, K, N: M % 256 == 0 and K % 128 == 0 and N % 1024 == 0),
    ("nki_matmul_fully_optimized_", "fully_opt",
     lambda M, K, N: M % 2048 == 0 and K % 1024 == 0 and N % 1024 == 0),
]


# Shapes aligned to every reference kernel's tightest constraint
# (fully_optimized with defaults): M % 2048, K % 1024, N % 1024. That
# way all four references participate in every row -- the emitted
# kernel's shape-generality is discussed in the report.
SHAPES = [
    (2048, 1024, 1024, "smallest all-refs"),
    (2048, 2048, 2048, "square 2k"),
    (4096, 2048, 2048, "mid"),
    (4096, 4096, 4096, "square 4k"),
]

N_WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
N_ITERS = int(os.environ.get("BENCH_ITERS", "10"))

TOL_ATOL = 1e-4
TOL_RTOL = 1e-2


def load_module(py_path: Path, unique_name: str):
    if not py_path.exists():
        raise SystemExit(f"missing file: {py_path}")
    spec = importlib.util.spec_from_file_location(unique_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sync():
    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        xm.mark_step()
    xm.wait_device_ops()


def _xla_device():
    if hasattr(torch_xla, "device"):
        return torch_xla.device()
    return xm.xla_device()


def run_correctness(kernel, lhsT_np, rhs_np, ref_out_np):
    """Call the @nki.jit kernel once via torch_xla to get a real output.

    @nki.jit on numpy inputs attempts a baremetal fallback that is not
    supported in this environment ("Did not find torch or jax, fallback
    nki.baremetal not supported"). So we route the correctness probe
    through torch_xla: tensors on the XLA device -> kernel -> sync ->
    back to numpy. This is one call per (shape, kernel) and is not on
    the timing path.
    """
    device = _xla_device()
    lhsT = torch.from_numpy(lhsT_np).to(device)
    rhs = torch.from_numpy(rhs_np).to(device)
    out = kernel(lhsT, rhs)
    _sync()
    out_np = out.cpu().numpy()
    return np.allclose(out_np, ref_out_np, atol=TOL_ATOL, rtol=TOL_RTOL)


def time_kernel(kernel, lhsT_np, rhs_np):
    """Compile to NEFF once via nki.benchmark, warmup+iters on device.

    Returns p50 device latency in seconds. Output is not returned --
    nki.benchmark does not reliably expose the tensor (see the
    nki-samples contributed/matmul.py pattern: baremetal for correctness,
    benchmark for timing).
    """
    bench_fn = benchmark(warmup=N_WARMUP, iters=N_ITERS)(kernel)
    bench_fn(lhsT_np, rhs_np)
    latency = bench_fn.benchmark_result.nc_latency
    p50_us = latency.get_latency_percentile(50)
    return p50_us * 1e-6


def fmt_cell(t_s: float, ok: bool) -> str:
    mark = " " if ok else "!"
    return f"{t_s*1e3:>8.2f}ms{mark}"


def main():
    if not EMITTED_PATH.exists():
        raise SystemExit(
            f"{EMITTED_PATH} missing -- run `python generate_kernels.py` first"
        )
    if not REF_PATH.exists():
        raise SystemExit(
            f"reference file not found: {REF_PATH}\n"
            f"set NKI_SAMPLES_PATH to the absolute path of "
            f"matrix_multiplication_nki_kernels.py"
        )

    emitted_mod = load_module(EMITTED_PATH, "emitted_matmul_kernel")
    ref_mod = load_module(REF_PATH, "nki_samples_matmul_kernels")

    emitted_fn = emitted_mod.matmul_kernel_nki
    refs = []
    for attr, label, supports in REF_KERNELS:
        fn = getattr(ref_mod, attr, None)
        if fn is None:
            print(f"  WARN  reference kernel '{attr}' not found; skipping")
            continue
        refs.append((label, fn, supports))

    verbose = os.environ.get("BENCH_VERBOSE") == "1"
    print(f"=== matmul benchmark via nki.benchmark "
          f"(warmup={N_WARMUP}, iters={N_ITERS}) ===")
    print(f"    emitted: {EMITTED_PATH}")
    print(f"    refs   : {REF_PATH}\n")

    col_labels = ["emitted"] + [l for (l, _, _) in refs]
    head = f"  {'shape':<28s}  " + "  ".join(f"{c:>10s}" for c in col_labels)
    print(head)
    print("  " + "-" * (len(head) - 2))

    rows = []

    for (M, K, N, tag) in SHAPES:
        rng = np.random.default_rng(0)
        lhs_np = rng.random((M, K), dtype=np.float32)
        rhs_np = rng.random((K, N), dtype=np.float32)
        lhsT_np = np.ascontiguousarray(lhs_np.T)
        ref_out_np = lhs_np @ rhs_np

        shape_str = f"({M:>4d} x {K:>4d} x {N:>4d})"
        cells = []
        errors = []

        try:
            ok = run_correctness(emitted_fn, lhsT_np, rhs_np, ref_out_np)
            t = time_kernel(emitted_fn, lhsT_np, rhs_np)
            cells.append(fmt_cell(t, ok))
        except Exception as e:
            cells.append(f"{'ERR':>10s}")
            errors.append(("emitted", e, traceback.format_exc()))

        for (rlabel, fn, supports) in refs:
            if not supports(M, K, N):
                cells.append(f"{'n/a':>10s}")
                continue
            try:
                ok = run_correctness(fn, lhsT_np, rhs_np, ref_out_np)
                t = time_kernel(fn, lhsT_np, rhs_np)
                cells.append(fmt_cell(t, ok))
            except Exception as e:
                cells.append(f"{'ERR':>10s}")
                errors.append((rlabel, e, traceback.format_exc()))

        label = f"{shape_str} {tag}"
        print(f"  {label:<28s}  " + "  ".join(f"{c:>10s}" for c in cells))
        rows.append((label, cells))
        for (klabel, e, tb) in errors:
            print(f"      {klabel:>10s}: {type(e).__name__}: {e}")
            if verbose:
                for line in tb.splitlines():
                    print(f"        {line}")

    # Clean summary table, reprinted at the end so it's not buried in
    # the Neuron compiler's per-kernel log output.
    print("\n" + "=" * (len(head) - 2))
    print("SUMMARY (ms/iter, p50 device latency)")
    print("=" * (len(head) - 2))
    print(head)
    print("  " + "-" * (len(head) - 2))
    for (label, cells) in rows:
        print(f"  {label:<28s}  " + "  ".join(f"{c:>10s}" for c in cells))
    print("\n(! = diverges from numpy matmul within "
          f"atol={TOL_ATOL}, rtol={TOL_RTOL})")


if __name__ == "__main__":
    main()
