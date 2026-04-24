"""
Auto-generated NKI kernel.
Source IR : add_kernel_specialized_linalg.mlir
Generator : linalg_to_nki.py  (VecAddConversionPattern)
Pipeline  : Triton → TTIR → triton-shared → Linalg → NKI

Op-pattern translation table (source → target):
  func.func @add_kernel(...)         → @nki.jit def add_kernel_nki(x, y, out)
  arith.constant 1024 : index              → BLOCK_SIZE constant
  arith.muli %pid, %c1024                  → for block in nl.affine_range(...)
  reinterpret_cast + copy (x tile [1024])  → nisa.dma_copy → x_tile [BLOCK_SIZE]
  reinterpret_cast + copy (y tile [1024])  → nisa.dma_copy → y_tile [BLOCK_SIZE]
  linalg.add ins(%x, %y)                  → nl.add (Vector Engine)
  materialize_in_destination               → nisa.dma_copy result → HBM

Tile size:   BLOCK_SIZE=1024
Engine:      Vector Engine (nl.add)
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def add_kernel_nki(x, y, out):
    # arith.constant 1024 : index → BLOCK_SIZE Python constant
    BLOCK_SIZE = 1024   # %c1024 = arith.constant 1024 : index
    
    n_elements = x.shape[0]
    assert n_elements % BLOCK_SIZE == 0, \
        "n_elements must be divisible by BLOCK_SIZE"
    
    # arith.muli %pid, %c1024 → pid * BLOCK_SIZE  →  explicit block loop
    for block in nl.affine_range(n_elements // BLOCK_SIZE):
        offset = block * BLOCK_SIZE
        
        # reinterpret_cast %arg0 + memref.copy (x block [BLOCK_SIZE])
        # → dma_copy x[offset:offset+BLOCK_SIZE] → x_tile
        x_tile = nl.ndarray((BLOCK_SIZE,), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_tile, src=x[offset : offset + BLOCK_SIZE])
        
        # reinterpret_cast %arg1 + memref.copy (y block [BLOCK_SIZE])
        # → dma_copy y[offset:offset+BLOCK_SIZE] → y_tile
        y_tile = nl.ndarray((BLOCK_SIZE,), dtype=y.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=y_tile, src=y[offset : offset + BLOCK_SIZE])
        
        # linalg.add ins(%x, %y) outs(%x) → nl.add (Vector Engine)
        result_tile = nl.add(x_tile, y_tile)
        
        # tensor.extract_slice + bufferization.materialize_in_destination
        # → nisa.dma_copy result_tile → out[offset:offset+BLOCK_SIZE]
        nisa.dma_copy(dst=out[offset : offset + BLOCK_SIZE], src=result_tile)
    
    return out
