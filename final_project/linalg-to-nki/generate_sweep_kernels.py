"""
Emit one NKI kernel per BLOCK config in the sweep.

Reuses the pipeline from generate_kernels.py and the CONFIGS list from
test_block_sweep.py. Each config produces:

    generated_sweep/matmul_kernel_{BM}_{BN}_{BK}.py
    generated_sweep/_ir/matmul_kernel_{BM}_{BN}_{BK}.ttir
    generated_sweep/_ir/matmul_kernel_{BM}_{BN}_{BK}.linalg
    generated_sweep/_ir/matmul_kernel_{BM}_{BN}_{BK}.lowered.mlir

This script runs offline -- it doesn't need the Trainium runtime. Commit the
generated tree, pull it on the Trainium host, and run test_sweep_kernels.py
there to get numerical correctness against torch.matmul.

Run (from linalg-to-nki/):
    conda activate triton
    source ../triton/.venv/bin/activate
    python generate_sweep_kernels.py
"""
import subprocess
import sys
from pathlib import Path

from generate_kernels import (
    OPT_BIN,
    TRANSLATE_BIN,
    TRITON_SHARED_OPT,
    compile_ttir,
)
from sweep_configs import CONFIGS


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "generated_sweep"
IR_DIR = OUT_DIR / "_ir"


def run(cmd):
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(
            "\nCommand failed:\n  " + " ".join(str(c) for c in cmd) + "\n"
        )
        sys.stderr.write(res.stderr)
        raise SystemExit(res.returncode)
    return res.stdout


def emit_one(bm: int, bn: int, bk: int, tag: str) -> None:
    stem = f"matmul_kernel_{bm}_{bn}_{bk}"
    py_path = OUT_DIR / f"{stem}.py"
    ttir_path = IR_DIR / f"{stem}.ttir"
    linalg_path = IR_DIR / f"{stem}.linalg"
    lowered_path = IR_DIR / f"{stem}.lowered.mlir"

    print(f"\n--- BLOCK=({bm},{bn},{bk})  {tag} ---")
    print(f"  [1/4] triton -> TTIR")
    ttir = compile_ttir(bm, bn, bk)
    ttir_path.write_text(ttir)

    print("  [2/4] TTIR -> specialized linalg")
    linalg = run([
        TRITON_SHARED_OPT, ttir_path,
        "--triton-to-linalg",
        "--linalg-specialize-generic-ops",
    ])
    linalg_path.write_text(linalg)

    print("  [3/4] linalg -> lowered NKI IR")
    lowered = run([
        OPT_BIN, linalg_path,
        "-nki-canonicalize-pid-loops",
        "-linalg-to-nki",
        "-nki-fuse-dma",
        "-nki-fuse-store",
        "-nki-fold-psum-init",
    ])
    lowered_path.write_text(lowered)

    print(f"  [4/4] lowered IR -> NKI Python -> {py_path.name}")
    run([TRANSLATE_BIN, lowered_path, "-o", py_path])


def main():
    for b in (TRITON_SHARED_OPT, OPT_BIN, TRANSLATE_BIN):
        if not b.exists():
            raise SystemExit(f"missing binary: {b}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IR_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").touch()

    print(f"=== Emitting NKI kernels for {len(CONFIGS)} block configs ===")
    for (bm, bn, bk, tag) in CONFIGS:
        emit_one(bm, bn, bk, tag)

    print(f"\n=== Done. Output in {OUT_DIR} ===")


if __name__ == "__main__":
    main()
