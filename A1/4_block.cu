// right_shift.cu
// Implements: out[i] = (i==0 ? 0 : in[i-1])
// Correct (race-free) version uses separate input/output arrays.
//
// Compile:
//   nvcc -O3 -std=c++17 right_shift.cu -o right_shift
//
// Run:
//   ./right_shift [N]

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(_e));                                  \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

__global__ void right_shift_buffer(int* in,
                            int* buff,
                            int n)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    buff[tid] = (tid == 0) ? 0 : in[tid - 1];
  }
}

__global__ void right_shift_cg(int* a, int n, int* block_last)
{
  cg::grid_group grid = cg::this_grid();

  extern __shared__ int s[];               // 32-1
  const int tid  = threadIdx.x;
  const int gid  = blockIdx.x * blockDim.x + tid;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // If gid is out of range, this thread doesn't participate further.
  // Note: this creates divergence *after* we do the block boundary write.
  // So we must be careful about where we use warp intrinsics.
  // We'll gate later operations on (gid < n).
  const bool active = (gid < n);

  // ---- Phase 1: publish each block's "last old element" into global memory ----
  // We need the last valid element of *this* block so the *next* block can read it.
  // Handle partial last block safely.
  const int block_start = blockIdx.x * blockDim.x;
  const int last_gid    = min(block_start + blockDim.x - 1, n - 1);
  const int last_tid    = last_gid - block_start;

  // The thread that owns last_gid writes the boundary value (old a[last_gid]).
  if (tid == last_tid) {
    block_last[blockIdx.x] = a[last_gid];
  }

  // Grid-wide barrier so all blocks can safely read block_last.
  grid.sync();

  // ---- Phase 2: per-thread compute using shuffle + smem + cross-block boundary ----
  if (!active) return;

  // Load old value for this element.
  int x = a[gid];

  // Cross-warp handoff inside the block: publish last lane of each warp.
  if (lane == 31) {
    s[warp] = x;
  }
  __syncthreads();

  // Intra-warp shift (safe here because no divergence since "active" gate is above)
  const unsigned full_mask = 0xFFFFFFFFu;
  int left = __shfl_up_sync(full_mask, x, 1);

  // Write result in-place.
  if (gid == 0) {
    a[0] = 0;
  } else if (lane == 0) {
    // First lane of a warp:
    if (warp > 0) {
      // predecessor is last lane of previous warp in the same block
      a[gid] = s[warp - 1];
    } else {
      // warp==0 and lane==0 => tid==0 => first element of this block.
      // predecessor lives in previous block:
      // a[block_start] = old_a[block_start - 1] == block_last[blockIdx.x - 1]
      a[gid] = block_last[blockIdx.x - 1];
    }
  } else {
    a[gid] = left;
  }
}

static void cpu_right_shift(const std::vector<int>& in, std::vector<int>& out) {
  const int n = static_cast<int>(in.size());
  out.resize(n);
  if (n == 0) return;
  out[0] = 0;
  for (int i = 1; i < n; ++i) out[i] = in[i - 1];
}

/*
  type:
  0: naive
  1: buffer over warp
  2: buffer over block
  3: buffer across 4 thread blocks
*/
int main(int argc, char** argv) {
  int type = 0;
  if (argc >= 2) {
    type = std::atoi(argv[1]);
  }
  int N = 4096;
  int threads = 1024;
  int blocks  = (N + threads - 1) / threads;

  std::vector<int> h_in(N), h_out(N, -1), h_ref;
  int j = 0;
  for (int i = 0; i < N; ++i)
  {
    h_in[i] = j + 1;
    j++;
    if(j == 1024)
    {
      j = 0;
    }
  }
  
  int *d_in = nullptr;

  CUDA_CHECK(cudaMalloc(&d_in,  sizeof(int) * N));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
  int *d_buff = nullptr;

  if(type == 0) // buffered right shift
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_buff,  sizeof(int) * N));
    right_shift_buffer<<<blocks, threads>>>(d_in, d_buff, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Total time: %.3f ms\n", ms);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_buff, sizeof(int) * N, cudaMemcpyDeviceToHost));
  }
  else if(type == 1) // using edge buffer for block boundaries
  {
    int shared_mem_size = (32-1);
    void* args[] = {
    &d_in,
    &N,
    &d_buff
    };
    dim3 blockDim (threads);
    dim3 gridDim (blocks);
    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_buff,  sizeof(int) * (blocks-1))); //block boundaries
    cudaLaunchCooperativeKernel(
        (void*)right_shift_cg,  // kernel pointer
        gridDim,                // dim3
        blockDim,               // dim3
        args,                   // kernel arguments
        shared_mem_size,         // dynamic shared memory per block
        nullptr                 // stream (or a cudaStream_t)
    );
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Total time: %.3f ms\n", ms);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_in, sizeof(int) * N, cudaMemcpyDeviceToHost));
  }

  // Verify
  cpu_right_shift(h_in, h_ref);

  int mismatches = 0;
  for (int i = 0; i < N; ++i) {
    if (h_out[i] != h_ref[i]) {
      if (mismatches < 10) {
        std::fprintf(stderr, "Mismatch at %d: got %d, expected %d\n",
                     i, h_out[i], h_ref[i]);
      }
      ++mismatches;
    }
  }

  if (mismatches == 0) {
    std::printf("PASS \n");
  } else {
    std::printf("FAIL  mismatches=%d\n", mismatches);
  }

  CUDA_CHECK(cudaFree(d_in));
  if(type == 0)
    CUDA_CHECK(cudaFree(d_buff));
  return (mismatches == 0) ? 0 : 2;
}
