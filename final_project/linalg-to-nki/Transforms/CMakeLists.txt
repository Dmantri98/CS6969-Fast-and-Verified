#===------------------------------------------------------------------------===
# linalg-to-nki conversion pass library + tablegen targets.
#===------------------------------------------------------------------------===
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LinalgToNKI)
add_public_tablegen_target(LinalgToNKIPassIncGen)

add_mlir_library(LinalgToNKITransforms
  LinalgToNKI.cpp
  FuseDma.cpp
  FuseStore.cpp
  FoldPsumInit.cpp
  FuseActivation.cpp
  CanonicalizePidLoops.cpp

  DEPENDS
  LinalgToNKIPassIncGen
  NKIDialectIncGen

  LINK_LIBS PUBLIC
  NKIDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
)
