// RUN: linalg-to-nki-opt %s -linalg-to-nki | FileCheck %s
//
// Smoke test for the standalone linalg.add → nki.tensor_tensor "add" lowering
// used by the vector-add kernel path (mirrors the inner body of
// ../../add_kernel.linalg after DMA/store fusion).

func.func @vec_add(%x: tensor<1024xf32>,
                   %y: tensor<1024xf32>) -> tensor<1024xf32> {
  %r = linalg.add ins(%x, %y : tensor<1024xf32>, tensor<1024xf32>)
                  outs(%x : tensor<1024xf32>) -> tensor<1024xf32>
  return %r : tensor<1024xf32>
}

// CHECK-LABEL: func.func @vec_add
// CHECK-NOT:     linalg.add
// CHECK:         %[[R:.*]] = nki.tensor_tensor "add" %{{.*}}, %{{.*}}
// CHECK-SAME:      : tensor<1024xf32>
// CHECK:         return %[[R]]
