// Smoke test for the nki-fuse-store pass: a single masked-store chain
// (extract_slice + reinterpret_cast + subview + materialize_in_destination)
// modeled exactly the way triton-shared emits it. The pass should collapse
// it into one nki.dma_store and erase the memref machinery.
//
// Run as:
//   linalg-to-nki-opt store_fuse.mlir -nki-fuse-store
func.func @store_one_tile(%tile: tensor<64x64xf32>,
                          %dst: memref<*xf32>,
                          %off: index, %s0: index, %s1: index,
                          %vrows: index, %vcols: index) {
  %rc = memref.reinterpret_cast %dst to
          offset: [%off], sizes: [64, 64], strides: [%s0, %s1]
          : memref<*xf32> to memref<64x64xf32, strided<[?, ?], offset: ?>>
  %slice = tensor.extract_slice %tile[0, 0] [%vrows, %vcols] [1, 1]
             : tensor<64x64xf32> to tensor<?x?xf32>
  %sdst = memref.subview %rc[0, 0] [%vrows, %vcols] [1, 1]
            : memref<64x64xf32, strided<[?, ?], offset: ?>>
              to memref<?x?xf32, strided<[?, ?], offset: ?>>
  bufferization.materialize_in_destination %slice in writable %sdst
            : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
  return
}
