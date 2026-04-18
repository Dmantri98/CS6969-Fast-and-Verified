"""
Offline block-size sweep for the Triton -> NKI toolchain.

The toolchain's value is that *any* valid Triton matmul should compile to NKI,
not just the canonical BLOCK={128,128,128} that generate_kernels.py uses by
default. This test parameterizes the pipeline over several (BM, BN, BK)
combinations and verifies each one round-trips cleanly, without touching any
Trainium runtime.

Two tiers per config:
  (a) pipeline: triton -> TTIR -> linalg -> lowered.mlir -> .py all succeed
      with zero exit code and non-empty output at each step
  (b) import:   emitted .py parses and exposes matmul_kernel_nki (the import
                is attempted but tolerates neuronxcc being absent)

Hardware-side numerical correctness lives in test_sweep_kernels.py, which
loads pre-generated kernels from generated_sweep/ on the Trainium host. Run
generate_sweep_kernels.py to produce that tree.

CONFIGS is the canonical list of BLOCK combinations the toolchain supports
and is re-used by generate_sweep_kernels.py.

Run (from linalg-to-nki/):
    conda activate triton
    source ../triton/.venv/bin/activate
    python test_block_sweep.py
"""
import ast
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

from generate_kernels import (
    OPT_BIN,
    TRANSLATE_BIN,
    TRITON_SHARED_OPT,
    compile_ttir,
)


# (BM, BN, BK, description). Triton requires powers of 2 for tl.dot operands.
# Sweep covers: sub-PE tiles, canonical, and each axis individually enlarged.
CONFIGS = [
    ( 32,  32,  32, "tiny cube (all < PE)"),
    ( 64,  64,  64, "small cube"),
    (128, 128, 128, "canonical"),
    (256, 128, 128, "wide M"),
    (128, 256, 128, "wide N"),
    (128, 128, 256, "wide K"),
    ( 64, 128,  32, "mixed small"),
]


def run_stage(cmd) -> tuple[bool, str, str]:
    res = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True
    )
    return res.returncode == 0, res.stdout, res.stderr


def compile_one(bm: int, bn: int, bk: int, out_dir: Path) -> tuple[bool, str]:
    """Run the full pipeline for one config. Returns (ok, error_detail)."""
    try:
        ttir = compile_ttir(bm, bn, bk)
    except Exception as e:
        return False, f"[triton] {type(e).__name__}: {e}"

    ttir_path = out_dir / "kernel.ttir"
    linalg_path = out_dir / "kernel.linalg"
    lowered_path = out_dir / "kernel.lowered.mlir"
    py_path = out_dir / "kernel.py"
    ttir_path.write_text(ttir)

    ok, stdout, stderr = run_stage([
        TRITON_SHARED_OPT, ttir_path,
        "--triton-to-linalg",
        "--linalg-specialize-generic-ops",
    ])
    if not ok:
        return False, f"[triton-shared-opt] {stderr.strip()[:240]}"
    linalg_path.write_text(stdout)

    ok, stdout, stderr = run_stage([
        OPT_BIN, linalg_path,
        "-nki-canonicalize-pid-loops",
        "-linalg-to-nki",
        "-nki-fuse-dma",
        "-nki-fuse-store",
        "-nki-fold-psum-init",
    ])
    if not ok:
        return False, f"[linalg-to-nki-opt] {stderr.strip()[:240]}"
    lowered_path.write_text(stdout)

    ok, stdout, stderr = run_stage([
        TRANSLATE_BIN, lowered_path, "-o", py_path,
    ])
    if not ok:
        return False, f"[linalg-to-nki-translate] {stderr.strip()[:240]}"
    if not py_path.exists() or py_path.stat().st_size == 0:
        return False, "[translate] produced empty .py"

    return True, ""


def check_importable(py_path: Path) -> tuple[bool, str]:
    """Syntactic + structural check. Tries to import too, but tolerates the
    neuronxcc runtime being absent (dev hosts without Trainium)."""
    src = py_path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"[syntax] {e}"

    fn_names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    if "matmul_kernel_nki" not in fn_names:
        return False, "[ast] missing matmul_kernel_nki"

    spec = importlib.util.spec_from_file_location("swept_kernel", py_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("neuronxcc"):
            return True, ""
        return False, f"[import] {type(e).__name__}: {e}"
    except Exception as e:
        return False, f"[import] {type(e).__name__}: {e}"
    if not hasattr(mod, "matmul_kernel_nki"):
        return False, "[import] missing matmul_kernel_nki"
    return True, ""


def main():
    for b in (TRITON_SHARED_OPT, OPT_BIN, TRANSLATE_BIN):
        if not b.exists():
            raise SystemExit(f"missing binary: {b}")

    print(f"=== Triton -> NKI block-size sweep ({len(CONFIGS)} configs) ===\n")

    passed = 0
    for (bm, bn, bk, tag) in CONFIGS:
        label = f"BLOCK=({bm:>3d},{bn:>3d},{bk:>3d})  {tag}"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ok, err = compile_one(bm, bn, bk, td)
            if not ok:
                print(f"  FAIL  {label}")
                print(f"        {err}")
                continue
            ok, err = check_importable(td / "kernel.py")
            if not ok:
                print(f"  FAIL  {label}")
                print(f"        {err}")
                continue
            print(f"  PASS  {label}")
            passed += 1

    total = len(CONFIGS)
    print(f"\nOVERALL: {passed} / {total} passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
