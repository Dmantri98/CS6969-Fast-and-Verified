"""
Auto-generated NKI kernel.
Generator : linalg-to-nki-translate
Pipeline  : Triton -> TTIR -> triton-shared -> Linalg -> nki dialect -> Python

Op-pattern translation table (source -> target):
  func.func @kernel(...)               -> @nki.jit def add_kernel_nki(x, y, out)
  arith.constant 1024                   -> BLOCK_SIZE constant
  arith.muli %pid, %c1024            -> for block in nl.affine_range(...)
  nki.dma_copy (x tile)                -> nl.load(x[...], mask=...)
  nki.dma_copy (y tile)                -> nl.load(y[...], mask=...)
  nki.tensor_tensor "add"             -> nl.add(...)
  nki.dma_store                        -> nl.store(out[...], mask=...)

Block size: BLOCK_SIZE=1024
Engine:     Vector Engine (nl.add)
"""
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit
def add_kernel_nki(x, y, out):
    """Computes out = nl.add(x, y) over BLOCK_SIZE-sized strips."""
    BLOCK_SIZE = 1024

    n_elements = x.shape[0]

    for block in nl.affine_range((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE):
        i = nl.mgrid[0:BLOCK_SIZE]
        mask = block * BLOCK_SIZE + i < n_elements

        x_tile = nl.zeros((BLOCK_SIZE,), dtype=x.dtype, buffer=nl.sbuf)
        x_tile[i] = nl.load(x[block * BLOCK_SIZE + i], mask=mask)

        y_tile = nl.zeros((BLOCK_SIZE,), dtype=y.dtype, buffer=nl.sbuf)
        y_tile[i] = nl.load(y[block * BLOCK_SIZE + i], mask=mask)

        z_tile = nl.add(x_tile, y_tile)
        nl.store(out[block * BLOCK_SIZE + i], value=z_tile, mask=mask)

    return out
