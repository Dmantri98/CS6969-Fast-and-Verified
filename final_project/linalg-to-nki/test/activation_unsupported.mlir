// Negative tests: shapes the NKI emitter cannot handle. The pipeline is
// `-nki-fuse-activation -nki-check-unsupported-elementwise` and each case is
// expected to emit a readable diagnostic AND return a nonzero exit code.
//
// Diagnostics contain `file:line:col:` prefixes but NOT the `func.func @name`
// line, so we anchor on the error text itself (which is unique per case) and
// order the CHECK lines to match the walk order in the pass.
//
// RUN: not linalg-to-nki-opt %s -nki-fuse-activation \
// RUN:     -nki-check-unsupported-elementwise 2>&1 | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>

// Four independent failures, emitted in function-declaration order. We match
// them with CHECK (not CHECK-LABEL) because the diagnostic stream has no func
// header; each pattern below is unique enough to act as its own anchor.

// --- 1. Multi-op body (bias-add followed by max is not a single NISA op). ---
// CHECK: error: 'linalg.generic' op unsupported elementwise linalg.generic
// CHECK-SAME: survived -nki-fuse-activation
// CHECK: note: supported:
func.func @bias_add_relu_not_fusable(%acc: tensor<64x64xf32>,
                                     %bias: tensor<64x64xf32>,
                                     %dst: memref<*xf32>,
                                     %off: index, %s0: index, %s1: index) {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc, %bias : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%empty : tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %s = arith.addf %in, %in_0 : f32
    %m = arith.maxnumf %s, %cst : f32
    linalg.yield %m : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// --- 2. Plain add: body is single arith.addf, not in the registry. ---
// CHECK: error: 'linalg.generic' op unsupported elementwise linalg.generic
// CHECK: note: supported:
func.func @add_not_supported(%acc: tensor<64x64xf32>,
                             %bias: tensor<64x64xf32>,
                             %dst: memref<*xf32>,
                             %off: index, %s0: index, %s1: index) {
  %empty = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc, %bias : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%empty : tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %s = arith.addf %in, %in_0 : f32
    linalg.yield %s : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// --- 3. maxnumf against a non-zero constant: not relu. ---
// CHECK: error: 'linalg.generic' op unsupported elementwise linalg.generic
// CHECK: note: supported:
func.func @max_nonzero_not_relu(%acc: tensor<64x64xf32>,
                                %dst: memref<*xf32>,
                                %off: index, %s0: index, %s1: index) {
  %cst = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %ones = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>)
            -> tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc, %ones : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%acc : tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %m = arith.maxnumf %in, %in_0 : f32
    linalg.yield %m : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// --- 4. Recognized relu, but the generic has two consumers: cannot fuse into
// either store, so the check pass must flag it as "recognized but not fused".
// CHECK: error: 'linalg.generic' op recognized activation 'relu'
// CHECK-SAME: was not fused
// CHECK: note: likely cause:
func.func @relu_multi_use(%acc: tensor<64x64xf32>,
                          %dst0: memref<*xf32>, %dst1: memref<*xf32>,
                          %off: index, %s0: index, %s1: index) {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %zeros = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>)
             -> tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc, %zeros : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%acc : tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %m = arith.maxnumf %in, %in_0 : f32
    linalg.yield %m : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst0 [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  nki.dma_store %out into %dst1 [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}
