"""
Hand-written NKI reference kernels used by the benchmarks. The matmul and
tensor-add kernels are adapted from the Amazon nki-samples tutorials
(see nki-samples/src/nki_samples/tutorials/matrix_multiplication and
.../tensor_addition) -- they represent an expert hand-tuned baseline.

Copyrights on the adapted code belong to Amazon.com (BSD-style license as
distributed with nki-samples).
"""
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


# ---------------------------------------------------------------------------
# tensor_add: nki-samples 2D tiled add (2D, requires shape%128==0 row,
# shape%512==0 col). Same structure as tensor_addition_nki_kernels.py but
# inlined so the benchmark file is self-contained.
# ---------------------------------------------------------------------------
@nki.jit
def ref_tensor_add_tile(a_input, b_input):
    """Single-tile add over a (128, 512) tile. Called as an SPMD grid."""
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)
    offset_i_x = nl.program_id(0) * 128
    offset_i_y = nl.program_id(1) * 512
    ix = offset_i_x + nl.arange(128)[:, None]
    iy = offset_i_y + nl.arange(512)[None, :]
    a_tile = nl.load(a_input[ix, iy])
    b_tile = nl.load(b_input[ix, iy])
    c_tile = a_tile + b_tile
    nl.store(c_output[ix, iy], value=c_tile)
    return c_output


def ref_tensor_add(a, b):
    """Lifted tile-restriction wrapper. a/b must be [N*128, M*512]."""
    grid_x = a.shape[0] // 128
    grid_y = a.shape[1] // 512
    return ref_tensor_add_tile[grid_x, grid_y](a, b)


# ---------------------------------------------------------------------------
# matmul: nki-samples tiled matmul (lhsT is (K, M), rhs is (K, N)).
# Requires M%128 == 0, K%128 == 0, N%512 == 0.
# ---------------------------------------------------------------------------
@nki.jit
def ref_matmul_tiled(lhsT, rhs):
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax                  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax      # 512

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TILE_K):
                lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                rhs_tile  = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype,  buffer=nl.sbuf)
                nisa.dma_copy(dst=lhsT_tile,
                              src=lhsT[k * TILE_K:(k + 1) * TILE_K,
                                       m * TILE_M:(m + 1) * TILE_M])
                nisa.dma_copy(dst=rhs_tile,
                              src=rhs[k * TILE_K:(k + 1) * TILE_K,
                                      n * TILE_N:(n + 1) * TILE_N])
                nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)
            res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum, dtype=result.dtype)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M,
                                      n * TILE_N:(n + 1) * TILE_N],
                          src=res_sb)
    return result


# ---------------------------------------------------------------------------
# matmul + relu (unfused reference): same tiled matmul, then a standalone
# NKI relu over the HBM result. This is the "obvious" baseline that our
# -nki-fuse-activation pass replaces with a single PSUM->SBUF activation.
# Extra cost here: one full HBM roundtrip (store result -> reload -> store).
# ---------------------------------------------------------------------------
@nki.jit
def ref_relu_inplace_kernel(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    M, N = x.shape
    TILE_M = 128
    TILE_N = 512
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            ix = m * TILE_M + nl.arange(TILE_M)[:, None]
            iy = n * TILE_N + nl.arange(TILE_N)[None, :]
            t = nl.load(x[ix, iy])
            t = nl.maximum(t, 0.0)
            nl.store(out[ix, iy], value=t)
    return out


def ref_matmul_relu_unfused(lhsT, rhs):
    """Reference = matmul then standalone relu (no fusion)."""
    mm = ref_matmul_tiled(lhsT, rhs)
    return ref_relu_inplace_kernel(mm)
