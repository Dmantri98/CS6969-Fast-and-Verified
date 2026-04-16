module {
  func.func @add_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg7, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xf32>
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.addi %2, %c1024 : index
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = arith.minsi %3, %4 : index
    %6 = arith.maxsi %5, %2 : index
    %7 = arith.subi %6, %2 : index
    %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%7] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32> to tensor<1024xf32>
    %9 = arith.index_cast %0 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%9], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    %10 = arith.index_cast %0 : i32 to index
    %11 = arith.addi %10, %c1024 : index
    %12 = arith.index_cast %arg3 : i32 to index
    %13 = arith.minsi %11, %12 : index
    %14 = arith.maxsi %13, %10 : index
    %15 = arith.subi %14, %10 : index
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%15] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%15] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %16 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32> to tensor<1024xf32>
    %17 = linalg.add ins(%8, %16 : tensor<1024xf32>, tensor<1024xf32>) outs(%8 : tensor<1024xf32>) -> tensor<1024xf32>
    %18 = arith.index_cast %0 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %19 = arith.index_cast %0 : i32 to index
    %20 = arith.addi %19, %c1024 : index
    %21 = arith.index_cast %arg3 : i32 to index
    %22 = arith.minsi %20, %21 : index
    %23 = arith.maxsi %22, %19 : index
    %24 = arith.subi %23, %19 : index
    %extracted_slice = tensor.extract_slice %17[0] [%24] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%24] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

