#===------------------------------------------------------------------------===
# NKI Dialect (IR) library + tablegen targets.
#===------------------------------------------------------------------------===
set(LLVM_TARGET_DEFINITIONS NKIDialect.td)
mlir_tablegen(NKIDialect.h.inc -gen-dialect-decls -dialect=nki)
mlir_tablegen(NKIDialect.cpp.inc -gen-dialect-defs -dialect=nki)
mlir_tablegen(NKIOps.h.inc -gen-op-decls)
mlir_tablegen(NKIOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(NKIDialectIncGen)

add_mlir_dialect_library(NKIDialect
  NKIDialect.cpp
  NKIOps.cpp

  DEPENDS
  NKIDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
