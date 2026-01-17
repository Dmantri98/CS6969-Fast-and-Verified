#include <cuda_runtime.h>

__global__ void right_shift_fast(int* a, int n)
{
  // Only one warp is assumed.
  const unsigned mask = __activemask() & ~1u;
  int lane = threadIdx.x; // 0..31
  
  if (lane >= n) return;

  // Load the old value of a[lane] into a register
  int x = a[lane];

  // Get old value from lane-1 (register-to-register), not from memory
  int left = __shfl_up_sync(mask, x, 1);

  // Write the shifted result in-place
  if (lane == 0) a[0] = 0;
  else          a[lane] = left;
}