"""
End-to-end pipeline driver: Triton vector-add kernel -> NKI Python kernel.

For each BLOCK_SIZE in CONFIGS:
  1. Compile the Triton add kernel with that constexpr -> TTIR
  2. Lower TTIR -> specialized linalg via triton-shared-opt
  3. Apply our MLIR pass pipeline via linalg-to-nki-opt
  4. Emit NKI Python via linalg-to-nki-translate
  5. Write to generated_add/add_kernel_<BLOCK_SIZE>.py

Run (from linalg-to-nki/):
    conda activate triton
    source ../triton/.venv/bin/activate
    python generate_add_kernels.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import triton
import triton.compiler as tc
import triton.language as tl


HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
TRITON_SHARED_OPT = (
    FINAL
    / "triton/build/cmake.linux-x86_64-cpython-3.10"
    / "third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
)
OPT_BIN = HERE / "build/bin/linalg-to-nki-opt"
TRANSLATE_BIN = HERE / "build/bin/linalg-to-nki-translate"
OUT_DIR = HERE / "generated_add"


CONFIGS = [
    # BLOCK_SIZE values to sweep
    128,
    256,
    512,
    1024,
    2048,
]


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def compile_ttir(block_size: int) -> str:
    src = tc.ASTSource(
        fn=add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": block_size},
    )
    try:
        target = triton.runtime.driver.active.get_current_target()
    except Exception:
        from triton.backends.compiler import GPUTarget
        target = GPUTarget("cuda", 80, 32)
    compiled = tc.compile(src, target=target)
    return compiled.asm["ttir"]


def run(cmd):
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write("\nCommand failed:\n  " + " ".join(str(c) for c in cmd) + "\n")
        sys.stderr.write(res.stderr)
        raise SystemExit(res.returncode)
    return res.stdout


def pipeline(block_size: int, out_py: Path) -> None:
    print(f"  [1/4] triton -> TTIR (BLOCK_SIZE = {block_size})")
    ttir = compile_ttir(block_size)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ttir_path = td / "kernel.ttir"
        linalg_path = td / "kernel.linalg"
        lowered_path = td / "kernel.lowered.mlir"
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
            "-linalg-to-nki",
            "-nki-fuse-dma",
            "-nki-fuse-store",
        ])
        lowered_path.write_text(lowered)

        print(f"  [4/4] lowered IR -> NKI Python -> {out_py}")
        out_py.parent.mkdir(parents=True, exist_ok=True)
        run([TRANSLATE_BIN, lowered_path, "-o", out_py])


def main():
    for b in (TRITON_SHARED_OPT, OPT_BIN, TRANSLATE_BIN):
        if not b.exists():
            raise SystemExit(f"missing binary: {b}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").touch()

    for block_size in CONFIGS:
        out_py = OUT_DIR / f"add_kernel_{block_size}.py"
        print(f"\n=== Generating kernel for BLOCK_SIZE = {block_size} ===")
        pipeline(block_size, out_py)
        print(f"  OK: {out_py}")

    print("\nAll kernels generated under", OUT_DIR)


if __name__ == "__main__":
    main()
