#===------------------------------------------------------------------------===
# linalg-to-nki: an out-of-tree MLIR dialect + conversion pass that lowers
# `linalg`-level kernels to a small `nki` dialect designed to mirror NKI's
# tensor-engine ISA. The dialect is the precursor to a final translation step
# that emits NKI Python (`nisa.*`) calls.
#===------------------------------------------------------------------------===
cmake_minimum_required(VERSION 3.20.0)
project(LinalgToNKI LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pull in MLIR/LLVM. Configure with e.g.
#   cmake -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
#         -DLLVM_DIR=$PWD/../llvm-project/build/lib/cmake/llvm ..
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
# So that headers can be included as "IR/NKIDialect.h" / "Transforms/Passes.h"
# both for source headers and tablegen-generated `*.inc` files.
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(IR)
add_subdirectory(Transforms)
add_subdirectory(tools/linalg-to-nki-opt)
add_subdirectory(tools/linalg-to-nki-translate)
