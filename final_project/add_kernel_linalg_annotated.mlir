// 1. ITERATION MAP: Defines a 1D identity mapping (index 'i' maps to position 'i')
#map = affine_map<(d0) -> (d0)>

module {
  // 2. SIGNATURE: %arg0 (x_ptr), %arg1 (y_ptr), %arg2 (out_ptr). 
  // 'memref<*xf32>' represents an unranked memory pointer to 32-bit floats.
  func.func @add_kernel(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    
    // 3. CONSTANTS: Set up the BLOCK_SIZE of 1024
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    
    // 4. BLOCK OFFSET: Calculate 'block_start = pid * BLOCK_SIZE'
    // Assuming %arg7 contains the program ID (pid)
    %0 = arith.muli %arg7, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index // Cast to MLIR 'index' type for memory addressing
    
    // 5. PROCESS 'X' ARRAY (Loading & Masking)
    // Shift the view of x_ptr so that index [0] starts at the 'block_start' offset
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%1], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    
    // Allocate a local buffer (like SRAM) to hold 1024 elements of X
    %alloc = memref.alloc() : memref<1024xf32>
    
    // Masking math: Calculate how many elements are valid to read 
    // equivalent to: valid_elements = min(block_start + 1024, n_elements) - block_start
    %2 = arith.index_cast %0 : i32 to index
    %3 = arith.addi %2, %c1024 : index
    %4 = arith.index_cast %arg3 : i32 to index // %arg3 is likely n_elements
    %5 = arith.minsi %3, %4 : index
    %6 = arith.maxsi %5, %2 : index
    %7 = arith.subi %6, %2 : index
    
    // Safely copy only the valid masked elements from global memory to local buffer
    %subview = memref.subview %reinterpret_cast[0] [%7] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%7] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    
    // Convert the local memory buffer into an immutable MLIR tensor for math operations
    %8 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32> to tensor<1024xf32>
    
    // 6. PROCESS 'Y' ARRAY (Loading & Masking)
    // This repeats the exact same pointer shifting, masking, and loading process for y_ptr
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
    %16 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32> to tensor<1024xf32> // Output is tensor %16
    
    // 7. CORE COMPUTATION (Vector Addition)
    // linalg.generic iterates in parallel over the 1024-element tensors (%8 and %16)
    %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %16 : tensor<1024xf32>, tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_7: f32, %out: f32):
      // For each element, add the float from X (%in) and the float from Y (%in_7)
      %25 = arith.addf %in, %in_7 : f32
      // Yield the result to the new output tensor
      linalg.yield %25 : f32
    } -> tensor<1024xf32> // The final result is stored in tensor %17
    
    // 8. PROCESS 'OUT' ARRAY (Masked Storing)
    // Shift the view of out_ptr to the block_start offset
    %18 = arith.index_cast %0 : i32 to index
    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    
    // Recalculate the mask size for the output bounds
    %19 = arith.index_cast %0 : i32 to index
    %20 = arith.addi %19, %c1024 : index
    %21 = arith.index_cast %arg3 : i32 to index
    %22 = arith.minsi %20, %21 : index
    %23 = arith.maxsi %22, %19 : index
    %24 = arith.subi %23, %19 : index
    
    // Slice off only the valid computed elements from the result tensor %17
    %extracted_slice = tensor.extract_slice %17[0] [%24] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%24] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    
    // Flush the immutable tensor data back into the actual physical memory pointer (out_ptr)
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    
    return
  }
}