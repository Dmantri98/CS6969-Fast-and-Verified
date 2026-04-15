// Comprehensive fusion tests for -nki-fuse-activation. Each function pairs a
// specific linalg.generic (or structured) activation with a downstream
// nki.dma_store; after the pass the generic should be erased and the store
// should carry the expected `activation = "..."` attribute.
//
// RUN: linalg-to-nki-opt %s -nki-fuse-activation | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>

// -----------------------------------------------------------------------------
// Unary math ops (body = single math.* op).
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @exp_epilogue
// CHECK: nki.dma_store
// CHECK-SAME: activation = "exp"
// CHECK-NOT: linalg.generic
func.func @exp_epilogue(%acc: tensor<64x64xf32>,
                        %dst: memref<*xf32>,
                        %off: index, %s0: index, %s1: index) {
  %init = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// CHECK-LABEL: func.func @log_epilogue
// CHECK: activation = "log"
func.func @log_epilogue(%acc: tensor<64x64xf32>,
                        %dst: memref<*xf32>,
                        %off: index, %s0: index, %s1: index) {
  %init = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.log %in : f32
    linalg.yield %e : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// CHECK-LABEL: func.func @sqrt_epilogue
// CHECK: activation = "sqrt"
func.func @sqrt_epilogue(%acc: tensor<64x64xf32>,
                         %dst: memref<*xf32>,
                         %off: index, %s0: index, %s1: index) {
  %init = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.sqrt %in : f32
    linalg.yield %e : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// CHECK-LABEL: func.func @rsqrt_epilogue
// CHECK: activation = "rsqrt"
func.func @rsqrt_epilogue(%acc: tensor<64x64xf32>,
                          %dst: memref<*xf32>,
                          %off: index, %s0: index, %s1: index) {
  %init = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.rsqrt %in : f32
    linalg.yield %e : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// CHECK-LABEL: func.func @abs_epilogue
// CHECK: activation = "abs"
func.func @abs_epilogue(%acc: tensor<64x64xf32>,
                        %dst: memref<*xf32>,
                        %off: index, %s0: index, %s1: index) {
  %init = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.absf %in : f32
    linalg.yield %e : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// CHECK-LABEL: func.func @tanh_epilogue
// CHECK: activation = "tanh"
func.func @tanh_epilogue(%acc: tensor<64x64xf32>,
                         %dst: memref<*xf32>,
                         %off: index, %s0: index, %s1: index) {
  %init = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.tanh %in : f32
    linalg.yield %e : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// CHECK-LABEL: func.func @erf_epilogue
// CHECK: activation = "erf"
func.func @erf_epilogue(%acc: tensor<64x64xf32>,
                        %dst: memref<*xf32>,
                        %off: index, %s0: index, %s1: index) {
  %init = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.erf %in : f32
    linalg.yield %e : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// CHECK-LABEL: func.func @sin_epilogue
// CHECK: activation = "sin"
func.func @sin_epilogue(%acc: tensor<64x64xf32>,
                        %dst: memref<*xf32>,
                        %off: index, %s0: index, %s1: index) {
  %init = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.sin %in : f32
    linalg.yield %e : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// CHECK-LABEL: func.func @cos_epilogue
// CHECK: activation = "cos"
func.func @cos_epilogue(%acc: tensor<64x64xf32>,
                        %dst: memref<*xf32>,
                        %off: index, %s0: index, %s1: index) {
  %init = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.cos %in : f32
    linalg.yield %e : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// CHECK-LABEL: func.func @tan_epilogue
// CHECK: activation = "tan"
func.func @tan_epilogue(%acc: tensor<64x64xf32>,
                        %dst: memref<*xf32>,
                        %off: index, %s0: index, %s1: index) {
  %init = tensor.empty() : tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%acc : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %e = math.tan %in : f32
    linalg.yield %e : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// -----------------------------------------------------------------------------
// Binary relu forms.
// -----------------------------------------------------------------------------

// Generic form as triton-shared actually emits for `tl.maximum(x, 0)`.
// CHECK-LABEL: func.func @relu_generic_maxnumf_epilogue
// CHECK: activation = "relu"
// CHECK-NOT: linalg.generic
// CHECK-NOT: arith.maxnumf
func.func @relu_generic_maxnumf_epilogue(%acc: tensor<64x64xf32>,
                                         %dst: memref<*xf32>,
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
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// Operand order flipped (zeros is lhs): still fuses.
// CHECK-LABEL: func.func @relu_generic_maxnumf_flipped
// CHECK: activation = "relu"
func.func @relu_generic_maxnumf_flipped(%acc: tensor<64x64xf32>,
                                        %dst: memref<*xf32>,
                                        %off: index, %s0: index, %s1: index) {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %zeros = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>)
             -> tensor<64x64xf32>
  %out = linalg.generic {indexing_maps = [#id, #id, #id],
                         iterator_types = ["parallel", "parallel"]}
    ins(%zeros, %acc : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%acc : tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %o: f32):
    %m = arith.maxnumf %in, %in_0 : f32
    linalg.yield %m : f32
  } -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}

// Structured linalg.max form.
// CHECK-LABEL: func.func @relu_structured_epilogue
// CHECK: activation = "relu"
// CHECK-NOT: linalg.max
func.func @relu_structured_epilogue(%acc: tensor<64x64xf32>,
                                    %dst: memref<*xf32>,
                                    %off: index, %s0: index, %s1: index) {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %zeros = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>)
             -> tensor<64x64xf32>
  %out = linalg.max ins(%acc, %zeros : tensor<64x64xf32>, tensor<64x64xf32>)
                    outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  nki.dma_store %out into %dst [%off] strides [%s0, %s1]
      : tensor<64x64xf32> into memref<*xf32>
  return
}
