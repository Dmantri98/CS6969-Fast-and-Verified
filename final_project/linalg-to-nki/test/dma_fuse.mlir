// Smoke test for the nki-fuse-dma pass: a single masked-load chain
// (reinterpret_cast -> subview -> copy -> to_tensor) modeled exactly the way
// triton-shared emits it, modulo the surrounding scf.for. The pass should
// collapse it into one nki.dma_copy and erase all of the side-effecting
// memref machinery.
//
// Run as:
//   linalg-to-nki-opt dma_fuse.mlir -nki-fuse-dma
func.func @load_one_tile(%src: memref<*xf32>, %off: index, %s0: index, %s1: index,
                         %vrows: index) -> tensor<64x32xf32> {
  %rc  = memref.reinterpret_cast %src to
           offset: [%off], sizes: [64, 32], strides: [%s0, %s1]
           : memref<*xf32> to memref<64x32xf32, strided<[?, ?], offset: ?>>
  %buf = memref.alloc() : memref<64x32xf32>
  %ssrc = memref.subview %rc[0, 0] [%vrows, 32] [1, 1]
            : memref<64x32xf32, strided<[?, ?], offset: ?>>
              to memref<?x32xf32, strided<[?, ?], offset: ?>>
  %sdst = memref.subview %buf[0, 0] [%vrows, 32] [1, 1]
            : memref<64x32xf32> to memref<?x32xf32, strided<[32, 1]>>
  memref.copy %ssrc, %sdst
            : memref<?x32xf32, strided<[?, ?], offset: ?>>
              to memref<?x32xf32, strided<[32, 1]>>
  %tile = bufferization.to_tensor %buf restrict writable
            : memref<64x32xf32> to tensor<64x32xf32>
  return %tile : tensor<64x32xf32>
}
