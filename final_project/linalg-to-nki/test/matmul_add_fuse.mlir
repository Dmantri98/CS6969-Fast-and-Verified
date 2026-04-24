// RUN: linalg-to-nki-opt %s -linalg-to-nki | FileCheck %s
//
// Smoke test for the linalg.matmul + linalg.add → nki.nc_matmul fusion.
// Mirrors the inner-loop pattern produced by triton-shared in
// ../matmul_kernel.linalg.

func.func @matmul_add(%A: tensor<64x32xf32>,
                      %B: tensor<32x64xf32>,
                      %acc: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<64x64xf32>
  %init = linalg.fill ins(%cst : f32)
                      outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %tmp = linalg.matmul ins(%A, %B : tensor<64x32xf32>, tensor<32x64xf32>)
                       outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  %r = linalg.add ins(%acc, %tmp : tensor<64x64xf32>, tensor<64x64xf32>)
                  outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %r : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @matmul_add
// CHECK-NOT:     linalg.matmul
// CHECK-NOT:     linalg.add
// CHECK:         %[[R:.*]] = nki.nc_matmul %{{.*}}, %{{.*}}, %{{.*}}
// CHECK-SAME:      : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
// CHECK:         return %[[R]]
