"""
Auto-generated NKI kernel.
Generator : linalg-to-nki-translate
Pipeline  : Triton -> TTIR -> triton-shared -> Linalg -> nki dialect -> Python

Op-pattern translation table (source -> target):
  func.func @kernel(...)               -> @nki.jit def matmul_kernel_nki(A, B)
  arith.constant 64/64/32                 -> TILE_M/N/K constants
  scf.for over pid_m / pid_n           -> for m, n in nl.affine_range(...)
  nki.psum_alloc                       -> nl.ndarray(buffer=nl.psum)
  scf.for over k iter_args(%psum)      -> for k in nl.affine_range(...)
  nki.dma_copy (A tile)                -> nisa.dma_copy(dst=lhs, src=A[...])
  nki.dma_copy (B tile)                -> nisa.dma_copy(dst=rhs, src=B[...])
  nki.nc_matmul                        -> nisa.nc_matmul(tile_size=(64, 64))
  nki.dma_store                        -> nisa.tensor_copy + nisa.dma_copy

Tile sizes:  TILE_M=64  TILE_N=64  TILE_K=32
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def matmul_kernel_nki(A, B):
    TILE_M = 64
    TILE_N = 64
    TILE_K = 32

    M, K = A.shape
    K_, N = B.shape
    assert K == K_, f"Contraction mismatch: A.K={K} != B.K={K_}"
    assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0, \
        "M, N, K must be divisible by their respective tile sizes"

    C = nl.ndarray((M, N), dtype=A.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                lhs_tile = nl.ndarray((TILE_M, TILE_K), dtype=A.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=lhs_tile,
                    src=A[m * TILE_M : (m + 1) * TILE_M,
                          k * TILE_K : (k + 1) * TILE_K],
                )

                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=rhs_tile,
                    src=B[k * TILE_K : (k + 1) * TILE_K,
                          n * TILE_N : (n + 1) * TILE_N],
                )

                nisa.nc_matmul(
                    dst=res_psum,
                    stationary=lhs_tile,
                    moving=rhs_tile,
                    tile_size=(64, 64),
                )

            res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=A.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sbuf, src=res_psum, dtype=A.dtype)
            nisa.dma_copy(
                dst=C[m * TILE_M : (m + 1) * TILE_M,
                      n * TILE_N : (n + 1) * TILE_N],
                src=res_sbuf,
            )
    return C
