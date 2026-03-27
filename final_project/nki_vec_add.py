import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    """
    NKI kernel to compute element-wise addition of two input tensors.
    """
    # Check both input tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape

    # Allocate space for the input tensors in SBUF (on-chip memory) and copy from HBM
    a_tile = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)
    b_tile = nl.ndarray(dtype=b_input.dtype, shape=b_input.shape, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)

    # Allocate space for the result and use tensor_tensor to perform element-wise addition
    c_tile = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

    # Create a tensor in HBM (Host Buffer Memory) and copy the result into HBM
    c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.hbm)
    nisa.dma_copy(dst=c_output, src=c_tile)

    # Return kernel output
    return c_output
