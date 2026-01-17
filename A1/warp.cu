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


__global__ void right_shift_naive(int* in,
                            int n)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    in[gid] = (gid == 0) ? 0 : in[gid - 1];
  }
}

__global__ void right_shift_buffer(const int* in,
                            int* buff,
                            int n)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    buff[gid] = (gid == 0) ? 0 : in[gid - 1];
  }
}

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
  1: shadow buffer
  2: shuffle_sync
*/
int main(int argc, char** argv) {
  int type = 0;
  if (argc >= 2) {
    type = std::atoi(argv[1]);
  }
  int N = 32;
  int threads = 32;
  int blocks  = (N + threads - 1) / threads;

  std::vector<int> h_in(N), h_out(N, -1), h_ref;
  for (int i = 0; i < N; ++i) h_in[i] = i + 1;
  
  int *d_in = nullptr;

  CUDA_CHECK(cudaMalloc(&d_in,  sizeof(int) * N));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
  int *d_buff = nullptr;

  if(type == 0) // naive right shift
  {
    right_shift_naive<<<blocks, threads>>>(d_in, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_in, sizeof(int) * N, cudaMemcpyDeviceToHost));
  }
  else if(type == 1) // buffered right shift across warp
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
  else if(type == 2)
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    right_shift_fast<<<blocks, threads>>>(d_in, N); // shuffle sync
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
  if(type == 1)
    CUDA_CHECK(cudaFree(d_buff));
  return (mismatches == 0) ? 0 : 2;
}
