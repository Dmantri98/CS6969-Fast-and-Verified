// Smoke test for nki-fuse-activation: a `linalg.max(acc, zeros)` (relu)
// feeding a `nki.dma_store` should be folded into the store's activation attr.
//
// Run as:
//   linalg-to-nki-opt relu_fuse.mlir -nki-fuse-activation
func.func @relu_epilogue(%acc: tensor<64x64xf32>,
                         %dst: memref<*xf32>,
                         %off: index, %s0: index, %s1: index) {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %zeros = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>)
             -> tensor<64x64xf32>
  %relu = linalg.max ins(%acc, %zeros : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  nki.dma_store %relu into %dst [%off] strides [%s0, %s1]
             : tensor<64x64xf32> into memref<*xf32>
  return
}
