"""
Auto-generated NKI kernel.
Generator : linalg-to-nki-translate
Pipeline  : Triton -> TTIR -> triton-shared -> Linalg -> nki dialect -> Python

Block size: BLOCK_SIZE=128
Engine:     Vector Engine (nl.add)
"""
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl


@nki.jit
def add_kernel_nki(x, y):
    """Computes out = nl.add(x, y) over BLOCK_SIZE-sized strips."""
    BLOCK_SIZE = 128
    PAR = 128
    FREE = 1

    n_elements = x.shape[0]
    out = nl.ndarray((n_elements,), dtype=x.dtype, buffer=nl.shared_hbm)

    for block in nl.affine_range((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE):
        i_p, i_f = nl.mgrid[0:PAR, 0:FREE]
        lin = i_p * FREE + i_f
        mask = block * BLOCK_SIZE + lin < n_elements

        x_tile = nl.zeros((PAR, FREE), dtype=x.dtype, buffer=nl.sbuf)
        x_tile[i_p, i_f] = nl.load(x[block * BLOCK_SIZE + lin], mask=mask)

        y_tile = nl.zeros((PAR, FREE), dtype=y.dtype, buffer=nl.sbuf)
        y_tile[i_p, i_f] = nl.load(y[block * BLOCK_SIZE + lin], mask=mask)

        z_tile = nl.add(x_tile, y_tile)
        nl.store(out[block * BLOCK_SIZE + lin], value=z_tile, mask=mask)

    return out
