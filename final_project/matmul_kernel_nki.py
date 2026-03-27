"""
Auto-generated NKI kernel.
Source IR : matmul_kernel.linalg
Generator : linalg_to_nki.py  (MatmulConversionPattern)
Pipeline  : Triton → TTIR → triton-shared → Linalg → NKI

Op-pattern translation table (source → target):
  func.func @matmul_kernel(...)         → @nki.jit def matmul_kernel_nki(A, B)
  arith.constant 64/64/32            → TILE_M/N/K constants
  arith.divsi/remsi %pid                   → for m/n in nl.affine_range(...)
  tensor.empty + linalg.fill (zeros)       → nl.ndarray(..., buffer=nl.psum)
  scf.for %k = 0 to cdiv(K,32)          → for k in nl.affine_range(K // TILE_K)
  reinterpret_cast + copy (A [64×32]) → nisa.dma_copy → lhsT_tile [TILE_K, TILE_M]
  reinterpret_cast + copy (B [32×64]) → nisa.dma_copy → rhs_tile  [TILE_K, TILE_N]
  linalg.matmul + linalg.add               → nisa.nc_matmul(tile_size=(64,64))
  materialize_in_destination               → nisa.tensor_copy + nisa.dma_copy

Tile sizes:  TILE_M=64  TILE_N=64  TILE_K=32
PE-grid:     tile_size=(64,64)  →  2×2 parallel slots
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def matmul_kernel_nki(A, B):
    # arith.constant ops → tile-size Python constants
    TILE_M = 64   # %c64 = arith.constant 64 : index
    TILE_N = 64   # %c64 = arith.constant 64 : index
    TILE_K = 32   # %c32_i32 = arith.constant 32 : i32
    PE_ROW_SIZE = 64  # smallest valid PE row slot >= TILE_M
    PE_COL_SIZE = 64  # smallest valid PE col slot >= TILE_N
    
    M, K = A.shape
    K_, N = B.shape
    assert K == K_, f"Contraction mismatch: A.K={K} != B.K={K_}"
    assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0, \
        "M, N, K must be divisible by their respective tile sizes"
    assert TILE_M <= PE_ROW_SIZE and TILE_N <= PE_COL_SIZE, \
        "Tile dims must not exceed chosen PE-grid tile_size"
    
    # memref.reinterpret_cast %arg2 (c_ptr) → nl.shared_hbm tensor
    C = nl.ndarray((M, N), dtype=A.dtype, buffer=nl.shared_hbm)
    
    # arith.divsi/remsi %pid → (pid_m, pid_n)  →  explicit tile loops
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            
            # tensor.empty + linalg.fill (zeros) → PSUM accumulator
            res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            
            # scf.for %k = 0 to cdiv(K, TILE_K) step 1 → affine_range K-loop
            for k in nl.affine_range(K // TILE_K):
                
                # reinterpret_cast %arg0 + memref.copy (A tile [BM×BK])
                # → dma_copy A[k*BK:(k+1)*BK, m*BM:(m+1)*BM] as lhsT [TILE_K, TILE_M]
                lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=A.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=lhsT_tile,
                    src=A[k * TILE_K : (k + 1) * TILE_K,
                        m * TILE_M : (m + 1) * TILE_M],
                )
                
                # reinterpret_cast %arg1 + memref.copy (B tile [BK×BN])
                # → dma_copy B[k*BK:(k+1)*BK, n*BN:(n+1)*BN] as rhs [TILE_K, TILE_N]
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=B.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=rhs_tile,
                    src=B[k * TILE_K : (k + 1) * TILE_K,
                        n * TILE_N : (n + 1) * TILE_N],
                )
                
                # tensor.empty + linalg.fill + linalg.matmul + linalg.add + scf.yield
                # → nisa.nc_matmul (accumulates into PSUM across K iterations)
                # tile_size=(64,64): 2×2=4 parallel PE slots
                nisa.nc_matmul(
                    dst=res_psum,
                    stationary=lhsT_tile,   # lhsT_tile.T = A_tile  →  A_tile @ B_tile
                    moving=rhs_tile,
                    tile_size=(64, 64),
                )
            
            # tensor.extract_slice + memref.subview
            # + bufferization.materialize_in_destination (store C tile)
            # → nisa.tensor_copy PSUM→SBUF  then  nisa.dma_copy SBUF→HBM
            res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=A.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sbuf, src=res_psum, dtype=A.dtype)
            nisa.dma_copy(
                dst=C[m * TILE_M : (m + 1) * TILE_M,
                    n * TILE_N : (n + 1) * TILE_N],
                src=res_sbuf,
            )
    
    return C
