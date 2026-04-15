#===------------------------------------------------------------------------===
# linalg-to-nki-translate: emit Python NKI from a fully-lowered MLIR module.
#===------------------------------------------------------------------------===
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
list(FILTER dialect_libs EXCLUDE REGEX "Test")

add_llvm_executable(linalg-to-nki-translate
  linalg-to-nki-translate.cpp

  DEPENDS
  NKIDialectIncGen
)

llvm_update_compile_flags(linalg-to-nki-translate)

target_link_libraries(linalg-to-nki-translate PRIVATE
  ${dialect_libs}
  MLIRIR
  MLIRParser
  MLIRSupport
  NKIDialect
)

mlir_check_all_link_libraries(linalg-to-nki-translate)
