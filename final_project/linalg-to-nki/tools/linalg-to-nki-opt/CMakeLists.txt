#===------------------------------------------------------------------------===
# linalg-to-nki-opt driver: links every upstream MLIR dialect/conversion lib
# plus the local nki dialect + linalg-to-nki conversion lib.
#===------------------------------------------------------------------------===
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# Drop MLIR's test-only libs (e.g. MLIRTosaTestPasses). These are only built
# when llvm-project is configured with -DMLIR_INCLUDE_TESTS=ON, and we don't
# need them for our opt driver.
list(FILTER dialect_libs EXCLUDE REGEX "Test")
list(FILTER conversion_libs EXCLUDE REGEX "Test")
list(FILTER translation_libs EXCLUDE REGEX "Test")

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${translation_libs}
  MLIRAnalysis
  MLIRIR
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  NKIDialect
  LinalgToNKITransforms
)

add_llvm_executable(linalg-to-nki-opt
  linalg-to-nki-opt.cpp

  DEPENDS
  NKIDialectIncGen
  LinalgToNKIPassIncGen
)
llvm_update_compile_flags(linalg-to-nki-opt)
target_link_libraries(linalg-to-nki-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(linalg-to-nki-opt)
