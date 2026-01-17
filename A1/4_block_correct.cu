__global__ void four_block_right_shift_buffer(int* in,
                            int* buff,
                            int n)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    buff[tid] = (tid == 0) ? 0 : in[tid - 1];
  }
}