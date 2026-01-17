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

__global__ void right_shift_smem(int* a, int n)
{
  // Per-(block) warp count; dynamic shared memory not used here.
  // One slot per warp to hold the warp's last element (lane 31).
  extern __shared__ int s[];  // size must be (blockDim.x / 32) ints

  const int tid  = threadIdx.x;                 // thread id in block
  const int gid  = blockIdx.x * blockDim.x + tid; // global id
  const int lane = tid & 31;                    // lane id in warp [0..31]
  const int warp = tid >> 5;                    // warp id in block

  if (gid >= n) return;

  // Full-warp mask is fine here because there is no divergence before shuffle
  // (all threads that reach here execute the shuffle).
  const unsigned mask = 0xFFFFFFFFu;

  // Load old value to a register.
  int x = a[gid];

  // Publish last lane's old value for cross-warp handoff.
  if (lane == 31) {
    s[warp] = x;
  }
  __syncthreads();

  // Intra-warp shift candidate.
  int left = __shfl_up_sync(mask, x, 1);

  // Write result (in-place).
  if (gid == 0) {
    a[0] = 0;
  } else if (lane == 0) {
    // First lane of a warp: predecessor is last lane of previous warp *in the same block*.
    // For warp==0, predecessor is in previous block (not handled here).
    if (warp > 0) {
      a[gid] = s[warp - 1];
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
  int N = 1024;
  int threads = 1024;
  int blocks  = (N + threads - 1) / threads;

  std::vector<int> h_in(N), h_out(N, -1), h_ref;
  for (int i = 0; i < N; ++i) h_in[i] = i + 1;

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
  else if(type == 1) // shared memory warp-shuffle
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    right_shift_smem<<<blocks, threads>>>(d_in, N);
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
