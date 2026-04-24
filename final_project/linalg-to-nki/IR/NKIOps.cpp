//===----------------------------------------------------------------------===//
// NKI Op implementations (verifiers, custom builders, ...).
//===----------------------------------------------------------------------===//
#include "IR/NKIDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#define GET_OP_CLASSES
#include "IR/NKIOps.cpp.inc"

using namespace mlir;
using namespace mlir::nki;

//===----------------------------------------------------------------------===//
// nki.nc_matmul
//===----------------------------------------------------------------------===//
LogicalResult NcMatmulOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto accType = dyn_cast<RankedTensorType>(getAcc().getType());
  auto resType = dyn_cast<RankedTensorType>(getResult().getType());

  if (!lhsType || !rhsType || !accType || !resType)
    return emitOpError("operands and result must be ranked tensors");

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      accType.getRank() != 2 || resType.getRank() != 2)
    return emitOpError("operands and result must be rank-2 tensors");

  const int64_t M = lhsType.getDimSize(0);
  const int64_t Klhs = lhsType.getDimSize(1);
  const int64_t Krhs = rhsType.getDimSize(0);
  const int64_t N = rhsType.getDimSize(1);

  // Only check static contraction-dim agreement; dynamic dims are skipped.
  if (!ShapedType::isDynamic(Klhs) && !ShapedType::isDynamic(Krhs) &&
      Klhs != Krhs)
    return emitOpError("contraction dim mismatch: lhs K=")
           << Klhs << " vs rhs K=" << Krhs;

  if (!ShapedType::isDynamic(M) && !ShapedType::isDynamic(accType.getDimSize(0)) &&
      accType.getDimSize(0) != M)
    return emitOpError("acc row count must equal lhs M");

  if (!ShapedType::isDynamic(N) && !ShapedType::isDynamic(accType.getDimSize(1)) &&
      accType.getDimSize(1) != N)
    return emitOpError("acc col count must equal rhs N");

  return success();
}
