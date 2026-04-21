"""
End-to-end pipeline driver: Triton matmul+relu kernel -> NKI Python kernel.

  1. Compile the Triton matmul-relu kernel -> TTIR
  2. Lower TTIR -> specialized linalg via triton-shared-opt
  3. Apply the MLIR pass pipeline via linalg-to-nki-opt
     (includes -nki-fuse-activation to fold relu into the store)
  4. Emit NKI Python via linalg-to-nki-translate
  5. Write to generated_relu/matmul_relu_kernel.py

The emitter hardcodes TILE_M=128, TILE_N=512, TILE_K=128 regardless of the
upstream BLOCK_SIZE, so a single canonical Triton compile is all we need.

Run (from linalg-to-nki/):
    conda activate triton
    source ../triton/.venv/bin/activate
    python generate_matmul_relu_kernels.py
"""
import subprocess
import sys
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
OUT_DIR = HERE / "generated_relu"

# (128, 512, 128) matches the emitter's TILE_{M,K,N}=128/128/512, so the IR's
# ⌈N/512⌉ loop bound reads naturally end-to-end. The emitted Python is the
# same regardless of Triton BLOCK_SIZE.
BLOCK_M, BLOCK_N, BLOCK_K = 128, 512, 128


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    mask_m = rm < M
    mask_n = rn < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k * BLOCK_SIZE_K
        curr_a_ptrs = a_ptr + (rm[:, None] * stride_am + (rk[None, :] + k_offset) * stride_ak)
        curr_b_ptrs = b_ptr + ((rk[:, None] + k_offset) * stride_bk + rn[None, :] * stride_bn)
        a = tl.load(curr_a_ptrs, mask=mask_m[:, None])
        b = tl.load(curr_b_ptrs, mask=mask_n[None, :])
        accumulator += tl.dot(a, b)

    # Fused relu epilogue. triton-shared lowers this to a `linalg.max` against
    # a zero-filled tensor, which our `-nki-fuse-activation` pass folds into
    # the `nki.dma_store`'s `activation` attribute.
    accumulator = tl.maximum(accumulator, 0.0)

    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])


def compile_ttir() -> str:
    src = tc.ASTSource(
        fn=matmul_relu_kernel,
        signature={
            "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
            "M": "i32", "N": "i32", "K": "i32",
            "stride_am": "i32", "stride_ak": "i32",
            "stride_bk": "i32", "stride_bn": "i32",
            "stride_cm": "i32", "stride_cn": "i32",
        },
        constexprs={
            "BLOCK_SIZE_M": BLOCK_M,
            "BLOCK_SIZE_N": BLOCK_N,
            "BLOCK_SIZE_K": BLOCK_K,
        },
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


def pipeline(out_py: Path) -> None:
    print(f"  [1/4] triton -> TTIR (BLOCK_SIZE = {BLOCK_M}, {BLOCK_N}, {BLOCK_K})")
    ttir = compile_ttir()

    # Persist intermediates alongside the emitted Python so we can inspect
    # where fusion did or didn't fire.
    art_dir = out_py.parent / "_ir"
    art_dir.mkdir(parents=True, exist_ok=True)
    tag = out_py.stem
    ttir_path = art_dir / f"{tag}.ttir"
    linalg_path = art_dir / f"{tag}.linalg"
    lowered_path = art_dir / f"{tag}.lowered.mlir"
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
        "-nki-fuse-activation",
        "-nki-fold-psum-init",
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

    out_py = OUT_DIR / "matmul_relu_kernel.py"
    print(f"\n=== Generating NKI matmul+relu kernel ===")
    pipeline(out_py)
    print(f"  OK: {out_py}")


if __name__ == "__main__":
    main()
