#include <cuda_runtime.h>

__global__ void right_shift_naive(int* in,
                            int n)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    in[gid] = (gid == 0) ? 0 : in[gid - 1];
  }
}