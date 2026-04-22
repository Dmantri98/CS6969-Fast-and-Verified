"""
Benchmark the emitted matmul kernel against an AWS nki-samples
reference on a shared set of shapes.

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
torch.matmul); the ref is AWS-authored and assumed correct.

Reference choice
----------------
The tutorial kernels under nki-samples/src/.../matrix_multiplication/
are written against a newer `nki.*` top-level API (nc_matmul with
dst=/stationary=/moving= kwargs, Python-list tile storage, memset with
different signature). That API is incompatible with the older
`neuronxcc.nki.*` that `neuronxcc.nki.benchmark` drives in this venv.

Instead we use nki-samples/contributed/matmul.py, which is written in
the `neuronxcc.nki` API (positional `ni.nc_matmul` that returns PSUM,
`nl.loop_reduce`, etc.) and is the sophisticated block-tiled variant --
the equivalent of the tutorial's `fully_optimized_`.

Loaded kernels
--------------
  emitted   : generated/matmul_kernel.py  (our pipeline output)
  fully_opt : contributed/matmul.py `matmul`  (default TILES_IN_BLOCK)

Reference file path
-------------------
Default:
  /home/ubuntu/nki-samples/contributed/matmul.py

Override:
  NKI_SAMPLES_PATH=/abs/path/to/matmul.py \
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
from neuronxcc.nki import benchmark
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as ni


# Inline reference kernels in the old neuronxcc.nki API (ni.nc_matmul
# is positional and returns a PSUM tensor; see contributed/matmul.py).
# These mirror the optimization levels of the nki-samples tutorial
# kernels (`nki_matmul_tiled_`, `nki_matmul_hoist_load_`) but against
# the API layer this venv actually supports.
def ref_tiled(lhsT, rhs):
    """Tile-by-tile matmul. Loads lhsT and rhs fresh for every (m, n)."""
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_
    TILE_K = nl.tile_size.pmax               # 128
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax      # 512

    Z = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TILE_K):
                a = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K,
                                 m * TILE_M:(m + 1) * TILE_M])
                b = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                n * TILE_N:(n + 1) * TILE_N])
                psum += ni.nc_matmul(a, b)
            nl.store(Z[m * TILE_M:(m + 1) * TILE_M,
                       n * TILE_N:(n + 1) * TILE_N], value=psum)
    return Z


def ref_hoist_load(lhsT, rhs):
    """Hoist the lhsT load out of the N loop: for each m, each lhsT tile
    is loaded once and reused across every n."""
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_
    TILE_K = nl.tile_size.pmax
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    NUM_K = K // TILE_K

    Z = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        lhsT_sbuf = nl.ndarray(
            (NUM_K, nl.par_dim(TILE_K), TILE_M),
            dtype=lhsT.dtype, buffer=nl.sbuf)
        for k in nl.affine_range(NUM_K):
            lhsT_sbuf[k] = nl.load(
                lhsT[k * TILE_K:(k + 1) * TILE_K,
                     m * TILE_M:(m + 1) * TILE_M])

        for n in nl.affine_range(N // TILE_N):
            psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(NUM_K):
                b = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                n * TILE_N:(n + 1) * TILE_N])
                psum += ni.nc_matmul(lhsT_sbuf[k], b)
            nl.store(Z[m * TILE_M:(m + 1) * TILE_M,
                       n * TILE_N:(n + 1) * TILE_N], value=psum)
    return Z


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
EMITTED_PATH = PROJECT_ROOT / "generated" / "matmul_kernel.py"

DEFAULT_REF_PATH = Path("/home/ubuntu/nki-samples/contributed/matmul.py")
REF_PATH = Path(os.environ.get("NKI_SAMPLES_PATH", str(DEFAULT_REF_PATH)))


# (callable, short label, (M, K, N) -> bool). For "fully_opt" the
# callable is a string that is resolved against the loaded ref module;
# for the inline kernels it is the function itself.
#
# Alignment predicates:
#   tiled / hoist_load : M % TILE_M(128),  K % TILE_K(128),  N % TILE_N(512)
#   fully_opt (contributed/matmul.py defaults 8/4/4):
#                        M % 512,  K % 1024,  N % 2048
REF_KERNELS = [
    (ref_tiled, "tiled",
     lambda M, K, N: M % 128 == 0 and K % 128 == 0 and N % 512 == 0),
    (ref_hoist_load, "hoist_load",
     lambda M, K, N: M % 128 == 0 and K % 128 == 0 and N % 512 == 0),
    ("matmul", "fully_opt",
     lambda M, K, N: M % 512 == 0 and K % 1024 == 0 and N % 2048 == 0),
]


# First shape hits tiled/hoist_load only (fully_opt is n/a there);
# remaining three satisfy all three refs.
SHAPES = [
    ( 512,  512,  512, "small"),
    (2048, 1024, 2048, "smallest all-refs"),
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


def time_kernel(kernel, M: int, K: int, N: int):
    """Compile to NEFF once via nki.benchmark, warmup+iters on device.

    Returns p50 device latency in seconds.
    """
    rng = np.random.default_rng(0)
    lhsT = rng.random((K, M)).astype(np.float32)
    rhs = rng.random((K, N)).astype(np.float32)
    bench_fn = benchmark(warmup=N_WARMUP, iters=N_ITERS)(kernel)
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
            f"nki-samples/contributed/matmul.py"
        )

    emitted_mod = load_module(EMITTED_PATH, "emitted_matmul_kernel")
    ref_mod = load_module(REF_PATH, "nki_samples_matmul_kernels")

    emitted_fn = emitted_mod.matmul_kernel_nki
    refs = []
    for entry, label, supports in REF_KERNELS:
        if callable(entry):
            fn = entry
        else:
            fn = getattr(ref_mod, entry, None)
            if fn is None:
                print(f"  WARN  reference kernel '{entry}' not found; skipping")
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
