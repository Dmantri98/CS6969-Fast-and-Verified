//===----------------------------------------------------------------------===//
// nki-fold-psum-init: rewrite linalg.fill 0.0 PSUM seeds to nki.psum_alloc.
//
//   %0 = tensor.empty() : tensor<MxNxT>
//   %1 = linalg.fill ins(%cst : T) outs(%0)        // %cst is +0.0
//   ...
//   scf.for ... iter_args(%acc = %1) {
//     %m = nki.nc_matmul %A, %B, %acc
//     scf.yield %m
//   }
//
//   ===>
//
//   %psum = nki.psum_alloc : tensor<MxNxT>
//   ...
//   scf.for ... iter_args(%acc = %psum) {
//     %m = nki.nc_matmul %A, %B, %acc
//     scf.yield %m
//   }
//
// The pass is intentionally narrow: it only fires when the fill's only use
// is an scf.for iter_args initializer AND the corresponding loop arg flows
// into an nki.nc_matmul acc operand.
//===----------------------------------------------------------------------===//
#include "Transforms/Passes.h"

#include "IR/NKIDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "nki-fold-psum-init"

namespace mlir {
namespace nki {

#define GEN_PASS_DEF_NKIFOLDPSUMINIT
#include "Transforms/Passes.h.inc"

namespace {

/// Returns true if `value` is a constant zero (any numeric type).
static bool isConstantZero(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return false;
  if (auto fa = dyn_cast<FloatAttr>(attr))
    return fa.getValue().isZero();
  if (auto ia = dyn_cast<IntegerAttr>(attr))
    return ia.getValue().isZero();
  return false;
}

/// Returns true if `loopArg` (a block argument of an scf.for) flows directly
/// into the `acc` operand of an nki.nc_matmul somewhere in its uses.
static bool flowsToNcMatmulAcc(BlockArgument loopArg) {
  for (Operation *user : loopArg.getUsers()) {
    if (auto mm = dyn_cast<NcMatmulOp>(user)) {
      if (mm.getAcc() == loopArg)
        return true;
    }
  }
  return false;
}

struct FillToPsumAlloc : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    // Single-result fill into a tensor (skip memref fills).
    if (fillOp.getNumResults() != 1)
      return rewriter.notifyMatchFailure(fillOp, "fill has no tensor result");
    Value result = fillOp.getResult(0);
    auto resTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resTy || !resTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          fillOp, "fill result is not a statically shaped tensor");

    // Single zero scalar input.
    if (fillOp.getInputs().size() != 1)
      return rewriter.notifyMatchFailure(fillOp, "fill has multiple inputs");
    if (!isConstantZero(fillOp.getInputs()[0]))
      return rewriter.notifyMatchFailure(fillOp,
                                         "fill scalar input is not zero");

    // The output operand must be a fresh tensor.empty() with no other users
    // (otherwise erasing it is unsafe).
    if (fillOp.getOutputs().size() != 1)
      return rewriter.notifyMatchFailure(fillOp, "fill has multiple outputs");
    auto emptyOp =
        fillOp.getOutputs()[0].getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp)
      return rewriter.notifyMatchFailure(
          fillOp, "fill destination is not tensor.empty()");
    if (!emptyOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          fillOp, "tensor.empty() has more than one use");

    // The fill result must have exactly one use, and that use must be an
    // scf.for iter_args initializer whose loop arg flows into nc_matmul.acc.
    if (!result.hasOneUse())
      return rewriter.notifyMatchFailure(fillOp,
                                         "fill result has more than one use");
    OpOperand &use = *result.getUses().begin();
    auto forOp = dyn_cast<scf::ForOp>(use.getOwner());
    if (!forOp)
      return rewriter.notifyMatchFailure(
          fillOp, "fill result is not consumed by scf.for");

    // Map the operand index to the corresponding region iter arg.
    unsigned operandIdx = use.getOperandNumber();
    unsigned firstInitIdx = forOp.getNumControlOperands();
    if (operandIdx < firstInitIdx)
      return rewriter.notifyMatchFailure(
          fillOp, "fill is bound to scf.for control, not iter_args");
    unsigned iterIdx = operandIdx - firstInitIdx;
    BlockArgument loopArg = forOp.getRegionIterArgs()[iterIdx];
    if (!flowsToNcMatmulAcc(loopArg))
      return rewriter.notifyMatchFailure(
          fillOp, "iter arg does not feed nki.nc_matmul.acc");

    // Replace fill with nki.psum_alloc.
    auto psum = PsumAllocOp::create(rewriter, fillOp.getLoc(), resTy);
    rewriter.replaceOp(fillOp, psum.getResult());

    // The empty op should now be dead; the rewriter erases SSA-dead
    // operations on its own only if they have no side effects, which is
    // true of tensor.empty.
    if (emptyOp->use_empty())
      rewriter.eraseOp(emptyOp);

    return success();
  }
};

struct NKIFoldPsumInitPass
    : public impl::NKIFoldPsumInitBase<NKIFoldPsumInitPass> {
  using impl::NKIFoldPsumInitBase<NKIFoldPsumInitPass>::NKIFoldPsumInitBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    populateNKIFoldPsumInitPatterns(patterns);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void populateNKIFoldPsumInitPatterns(RewritePatternSet &patterns) {
  patterns.add<FillToPsumAlloc>(patterns.getContext());
}

} // namespace nki
} // namespace mlir
