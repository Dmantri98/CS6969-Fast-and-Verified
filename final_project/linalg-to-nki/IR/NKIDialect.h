//===----------------------------------------------------------------------===//
// NKI Dialect public C++ interface.
//===----------------------------------------------------------------------===//
#ifndef LINALG_TO_NKI_IR_NKIDIALECT_H
#define LINALG_TO_NKI_IR_NKIDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect class declaration (generated from NKIDialect.td).
#include "IR/NKIDialect.h.inc"

// Op class declarations (generated from NKIDialect.td).
#define GET_OP_CLASSES
#include "IR/NKIOps.h.inc"

#endif // LINALG_TO_NKI_IR_NKIDIALECT_H
