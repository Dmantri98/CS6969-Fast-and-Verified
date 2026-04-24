module {
  func.func @matmul_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) {
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.addi %arg3, %c63_i32 : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    %2 = nki.psum_alloc : tensor<64x64xf32>
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    scf.for %arg18 = %c0_i32 to %1 step %c1_i32  : i32 {
      scf.for %arg19 = %c0_i32 to %4 step %c1_i32  : i32 {
        %5 = arith.muli %arg18, %c64_i32 : i32
        %6 = arith.muli %arg19, %c64_i32 : i32
        %7 = arith.addi %arg5, %c31_i32 : i32
        %8 = arith.divsi %7, %c32_i32 : i32
        %9 = scf.for %arg20 = %c0_i32 to %8 step %c1_i32 iter_args(%arg21 = %2) -> (tensor<64x64xf32>)  : i32 {
          %17 = arith.muli %arg20, %c32_i32 : i32
          %18 = arith.index_cast %5 : i32 to index
          %19 = arith.index_cast %arg6 : i32 to index
          %20 = arith.muli %18, %19 : index
          %21 = arith.index_cast %17 : i32 to index
          %22 = arith.index_cast %arg7 : i32 to index
          %23 = arith.muli %21, %22 : index
          %24 = arith.addi %20, %23 : index
          %25 = arith.index_cast %17 : i32 to index
          %26 = arith.index_cast %arg8 : i32 to index
          %27 = arith.muli %25, %26 : index
          %28 = arith.index_cast %6 : i32 to index
          %29 = arith.index_cast %arg9 : i32 to index
          %30 = arith.muli %28, %29 : index
          %31 = arith.addi %27, %30 : index
          %32 = nki.dma_copy %arg0[%24] strides[%19, %22] : memref<*xf32> to tensor<64x32xf32>
          %33 = nki.dma_copy %arg1[%31] strides[%26, %29] : memref<*xf32> to tensor<32x64xf32>
          %34 = nki.nc_matmul %32, %33, %arg21 : (tensor<64x32xf32>, tensor<32x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
          scf.yield %34 : tensor<64x64xf32>
        }
        %10 = arith.index_cast %5 : i32 to index
        %11 = arith.index_cast %arg10 : i32 to index
        %12 = arith.muli %10, %11 : index
        %13 = arith.index_cast %6 : i32 to index
        %14 = arith.index_cast %arg11 : i32 to index
        %15 = arith.muli %13, %14 : index
        %16 = arith.addi %12, %15 : index
        nki.dma_store %9 into %arg2[%16] strides[%11, %14] : tensor<64x64xf32> into memref<*xf32>
      }
    }
    return
  }
}

