//===----------------------------------------------------------------------===//
// NKI Dialect implementation: dialect registration.
//===----------------------------------------------------------------------===//
#include "IR/NKIDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::nki;

void NKIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IR/NKIOps.cpp.inc"
      >();
}

#include "IR/NKIDialect.cpp.inc"
