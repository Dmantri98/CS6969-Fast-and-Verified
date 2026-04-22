"""
Benchmark the emitted matmul kernel against the AWS nki-samples
reference kernels on a shared set of shapes.

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
emitted kernel masks at every load/store and runs on every shape we
include.

Reference file path
-------------------
Default:
  /home/ubuntu/nki-samples/src/nki_samples/tutorials/matrix_multiplication/matrix_multiplication_nki_kernels.py

Override:
  NKI_SAMPLES_PATH=/abs/path/to/matrix_multiplication_nki_kernels.py \
      python tests/benchmark/benchmark_matmul.py

Methodology
-----------
Each iteration is traced and synced independently (sync per call) so the
neuronxcc disk cache is consulted once per call, matching the warmup
graph shape. For the emitted kernel this means warmup primes the cache
and timed iters measure pure device execute. The nki-samples refs stamp
random temp paths into their generated AST, so each call re-compiles
regardless -- that cache-miss cost is part of what the benchmark is
meant to expose (and is what a user would experience on a real workload
with varying shapes). We report min across N_ITERS to suppress host
noise. Correctness is checked with `torch.allclose` against
`torch.matmul` using the same tolerance as the existing suites.

Run (on Trainium):
    python tests/benchmark/benchmark_matmul.py
"""
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch_xla
from torch_xla.core import xla_model as xm


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
EMITTED_PATH = PROJECT_ROOT / "generated" / "matmul_kernel.py"

DEFAULT_REF_PATH = Path(
    "/home/ubuntu/nki-samples/src/nki_samples/tutorials/"
    "matrix_multiplication/matrix_multiplication_nki_kernels.py"
)
REF_PATH = Path(os.environ.get("NKI_SAMPLES_PATH", str(DEFAULT_REF_PATH)))


# (attr name in reference module, short label, (M, K, N) -> bool).
# The predicates match the explicit asserts at the top of each kernel
# in nki-samples/src/nki_samples/tutorials/matrix_multiplication/
# matrix_multiplication_nki_kernels.py:
#   tiled            lines 96-98   : M%128,   N%512,  K%128
#   hoist_load       lines 164-166 : M%128,   N%512,  K%128
#   block_free_dim   lines 251-252 : M%256,   N%1024  (K%128 implicit)
#   fully_optimized  lines 365-367 : M%2048,  N%1024, K%1024  (defaults)
# Calling a reference outside its predicate raises AssertionError; we
# emit "n/a" in the table instead.
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
# way all four references participate in every row -- we talk about
# the emitted kernel's shape-generality in the report, not here.
SHAPES = [
    (2048, 1024, 1024, "smallest all-refs"),
    (2048, 2048, 2048, "square 2k"),
    (4096, 2048, 2048, "mid"),
    (4096, 4096, 4096, "square 4k"),
]

N_WARMUP = int(os.environ.get("BENCH_WARMUP", "1"))
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


def sync():
    # Prefer the new torch_xla.sync API if present; fall back to mark_step.
    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        xm.mark_step()
    xm.wait_device_ops()


def get_device():
    if hasattr(torch_xla, "device"):
        return torch_xla.device()
    return xm.xla_device()


def time_kernel(kernel, lhsT, rhs):
    # Sync per call so each iteration is its own traced graph (same HLO
    # shape as warmup). For the emitted kernel that means warmup primes
    # the neuronxcc cache and every timed iter is a pure exec. For the
    # nki-samples refs, each call still re-compiles because their tooling
    # stamps random temp paths into the generated AST -- that cache-miss
    # is part of what we want the benchmark to expose. We report min
    # across iters to suppress host-side noise.
    for _ in range(N_WARMUP):
        _ = kernel(lhsT, rhs)
        sync()
    times = []
    out = None
    for _ in range(N_ITERS):
        start = time.perf_counter()
        out = kernel(lhsT, rhs)
        sync()
        times.append(time.perf_counter() - start)
    return min(times), out


def is_close(out, ref):
    return torch.allclose(out, ref, atol=TOL_ATOL, rtol=TOL_RTOL)


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

    device = get_device()
    verbose = os.environ.get("BENCH_VERBOSE") == "1"
    print(f"=== matmul benchmark on {device} "
          f"(warmup={N_WARMUP}, iters={N_ITERS}) ===")
    print(f"    emitted: {EMITTED_PATH}")
    print(f"    refs   : {REF_PATH}\n")

    col_labels = ["emitted"] + [l for (l, _, _) in refs]
    head = f"  {'shape':<28s}  " + "  ".join(f"{c:>10s}" for c in col_labels)
    print(head)
    print("  " + "-" * (len(head) - 2))

    rows = []  # [(label, [cell_str, ...]), ...] for the clean summary.

    for (M, K, N, tag) in SHAPES:
        lhs = torch.rand((M, K), dtype=torch.float32, device=device)
        rhs = torch.rand((K, N), dtype=torch.float32, device=device)
        ref_out = torch.matmul(lhs, rhs)
        sync()

        shape_str = f"({M:>4d} x {K:>4d} x {N:>4d})"
        cells = []
        errors = []

        try:
            t, out = time_kernel(emitted_fn, lhs.T, rhs)
            cells.append(fmt_cell(t, is_close(out, ref_out)))
        except Exception as e:
            cells.append(f"{'ERR':>10s}")
            errors.append(("emitted", e, traceback.format_exc()))

        for (rlabel, fn, supports) in refs:
            if not supports(M, K, N):
                cells.append(f"{'n/a':>10s}")
                continue
            try:
                t, out = time_kernel(fn, lhs.T, rhs)
                cells.append(fmt_cell(t, is_close(out, ref_out)))
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
    print("SUMMARY (ms/iter)")
    print("=" * (len(head) - 2))
    print(head)
    print("  " + "-" * (len(head) - 2))
    for (label, cells) in rows:
        print(f"  {label:<28s}  " + "  ".join(f"{c:>10s}" for c in cells))
    print("\n(! = diverges from torch.matmul within "
          f"atol={TOL_ATOL}, rtol={TOL_RTOL})")


if __name__ == "__main__":
    main()
