"""
Auto-generated NKI kernel.
Generator : linalg-to-nki-translate
Pipeline  : Triton -> TTIR -> triton-shared -> Linalg -> nki dialect -> Python

Op-pattern translation table (source -> target):
  func.func @kernel(...)               -> @nki.jit def matmul_relu_kernel_nki(lhsT, rhs)
  arith.constant 128/512/128               -> TILE_M/N/K constants (NC-v2 default 128/512/128)
  scf.for over pid_m / pid_n           -> ceil-div nl.affine_range(...)
  nki.psum_alloc                       -> nl.ndarray(buffer=nl.psum)
  scf.for over k iter_args(%psum)      -> ceil-div nl.affine_range(...)
  nki.dma_copy (A tile)                -> nl.load(lhsT[...], mask=...)
  nki.dma_copy (B tile)                -> nl.load(rhs[...], mask=...)
  nki.nc_matmul                        -> nisa.nc_matmul(...)
  nki.dma_store                        -> nisa.tensor_copy + nl.store(C[...], mask=...)

Tile sizes:  TILE_M=128  TILE_N=512  TILE_K=128  (full 128x128 PE array, N=512)
"""
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit
def matmul_relu_kernel_nki(lhsT, rhs):
    """lhsT is A transposed (shape K, M). Computes C = lhsT.T @ rhs."""
    TILE_M = 128
    TILE_N = 512
    TILE_K = 128

    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, f"Contraction mismatch: lhsT.K={K} != rhs.K={K_}"

    C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range((M + TILE_M - 1) // TILE_M):
        for n in nl.affine_range((N + TILE_N - 1) // TILE_N):
            res_psum = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for k in nl.affine_range((K + TILE_K - 1) // TILE_K):
                i_k, i_m = nl.mgrid[0:TILE_K, 0:TILE_M]
                mask_lhsT = (k * TILE_K + i_k < K) & (m * TILE_M + i_m < M)
                lhsT_tile = nl.zeros((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                lhsT_tile[i_k, i_m] = nl.load(
                    lhsT[k * TILE_K + i_k, m * TILE_M + i_m],
                    mask=mask_lhsT,
                )

                i_k, i_n = nl.mgrid[0:TILE_K, 0:TILE_N]
                mask_rhs = (k * TILE_K + i_k < K) & (n * TILE_N + i_n < N)
                rhs_tile = nl.zeros((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                rhs_tile[i_k, i_n] = nl.load(
                    rhs[k * TILE_K + i_k, n * TILE_N + i_n],
                    mask=mask_rhs,
                )

                res_psum[...] += nisa.nc_matmul(
                    stationary=lhsT_tile,
                    moving=rhs_tile,
                )

            i_m, i_n = nl.mgrid[0:TILE_M, 0:TILE_N]
            res_sbuf = nisa.activation(op=nl.relu, data=res_psum, dtype=lhsT.dtype)
            nl.store(
                C[m * TILE_M + i_m, n * TILE_N + i_n],
                value=res_sbuf,
                mask=(m * TILE_M + i_m < M) & (n * TILE_N + i_n < N),
            )
    return C
