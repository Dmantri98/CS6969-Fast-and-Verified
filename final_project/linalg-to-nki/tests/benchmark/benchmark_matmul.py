"""
Benchmark the emitted matmul kernel against the AWS nki-samples
reference kernels on a shared set of shapes.

We use `neuronxcc.nki.benchmark`, which compiles each kernel to a NEFF
once and times pure on-device execute. This sidesteps torch_xla tracing
/ HLO-caching issues (the nki-samples kernels produce a fresh HLO hash
on every call through torch_xla, so every torch_xla call paid a full
neuronxcc compile -- what we were measuring there was compile time, not
device time).

Correctness is NOT checked here: nki.benchmark takes over the Neuron
cores, so we can't also run a torch_xla probe in the same process
(NRT errors out with "Logical Neuron Core(s) not available"). The
emitted kernel's correctness is covered by tests/ (torch_xla against
torch.matmul); the refs are AWS-authored and assumed correct.

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
# Match the ref kernels' imports: the nki-samples matmul refs do
# `import nki.language as nl` / `@nki.jit`. If we import benchmark
# from `neuronxcc.nki` instead, `neuronxcc.nki.benchmark` sets up a
# different trace context than the one `@nki.jit` (from top-level
# `nki`) expects, and `nl.affine_range(...)` inside the ref kernel
# body returns None at re-trace time. Importing both from the same
# top-level `nki` package keeps the trace context consistent.
import nki.language as nl
from nki import benchmark


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


def load_module(py_path: Path, unique_name: str):
    if not py_path.exists():
        raise SystemExit(f"missing file: {py_path}")
    spec = importlib.util.spec_from_file_location(unique_name, py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _unwrap_nki_jit(kernel):
    """Return the raw Python function inside a @nki.jit TraceKernel.

    nki.benchmark wrapped on top of @nki.jit re-traces the function and
    the double-trace path breaks inside TraceKernel.expand_kernel_with_ctx
    for the nki-samples matmul refs (affine_range returns None). The
    contributed/matmul.py example applies @nki.benchmark to a raw function,
    not a pre-jit'd one; mirror that by unwrapping.
    """
    for attr in ("__wrapped__", "func", "py_func", "kernel_fn", "fn",
                 "_kernel", "_func"):
        inner = getattr(kernel, attr, None)
        if callable(inner) and inner is not kernel:
            return inner
    return kernel


def time_kernel(kernel, M: int, K: int, N: int):
    """Compile to NEFF once via nki.benchmark, warmup+iters on device.

    Returns p50 device latency in seconds. Inputs use the
    nl.static_cast(numpy, dtype) pattern from the nki-samples attention
    benchmark test (test_attention.py).
    """
    rng = np.random.default_rng(0)
    lhsT = rng.random((K, M)).astype(np.float32)
    rhs = rng.random((K, N)).astype(np.float32)
    bench_fn = benchmark(warmup=N_WARMUP, iters=N_ITERS)(_unwrap_nki_jit(kernel))
    bench_fn(lhsT, rhs)
    latency = bench_fn.benchmark_result.nc_latency
    p50_us = latency.get_latency_percentile(50)
    return p50_us * 1e-6


def fmt_cell(t_s: float) -> str:
    return f"{t_s*1e3:>8.2f}ms"


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
        shape_str = f"({M:>4d} x {K:>4d} x {N:>4d})"
        cells = []
        errors = []

        try:
            t = time_kernel(emitted_fn, M, K, N)
            cells.append(fmt_cell(t))
        except Exception as e:
            cells.append(f"{'ERR':>10s}")
            errors.append(("emitted", e, traceback.format_exc()))

        for (rlabel, fn, supports) in refs:
            if not supports(M, K, N):
                cells.append(f"{'n/a':>10s}")
                continue
            try:
                t = time_kernel(fn, M, K, N)
                cells.append(fmt_cell(t))
            except Exception as e:
                cells.append(f"{'ERR':>10s}")
                errors.append((rlabel, e, traceback.format_exc()))

        label = f"{shape_str} {tag}"
        print(f"  {label:<28s}  " + "  ".join(f"{c:>10s}" for c in cells))
        rows.append((label, cells))
        for (klabel, e, tb) in errors:
            print(f"      {klabel:>10s}: {type(e).__name__}: {e}")
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
    print("\n(correctness for the emitted kernel is covered by the "
          "torch_xla suites in tests/; this file measures device time only)")


if __name__ == "__main__":
    main()
