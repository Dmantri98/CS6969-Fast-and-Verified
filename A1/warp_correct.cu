#include <cuda_runtime.h>

__global__ void right_shift_buffer(const int* in,
                            int* buff,
                            int n)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    buff[gid] = (gid == 0) ? 0 : in[gid - 1];
  }
}