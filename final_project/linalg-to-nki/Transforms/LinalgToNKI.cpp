//===----------------------------------------------------------------------===//
// linalg-to-nki conversion pass.
//
// Pattern: fuse the linalg matmul/add accumulator pattern into nki.nc_matmul.
//
//   %init  = linalg.fill ins(%cst : f32) outs(%empty) -> tensor<MxNxf32>
//   %temp  = linalg.matmul ins(%A, %B)   outs(%init)  -> tensor<MxNxf32>
//   %nacc  = linalg.add    ins(%acc, %temp) outs(%acc) -> tensor<MxNxf32>
//
// becomes
//
//   %nacc  = nki.nc_matmul %A, %B, %acc : (...) -> tensor<MxNxf32>
//===----------------------------------------------------------------------===//
#include "Transforms/Passes.h"

#include "IR/NKIDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "linalg-to-nki"

namespace mlir {
namespace nki {

#define GEN_PASS_DEF_LINALGTONKI
#include "Transforms/Passes.h.inc"

namespace {

/// Returns true if `v` is produced by an `arith.constant` whose value is a
/// floating-point or integer zero.
static bool isZeroConstant(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  auto cst = dyn_cast<arith::ConstantOp>(def);
  if (!cst)
    return false;
  Attribute attr = cst.getValue();
  if (auto fa = dyn_cast<FloatAttr>(attr))
    return fa.getValue().isZero();
  if (auto ia = dyn_cast<IntegerAttr>(attr))
    return ia.getValue().isZero();
  return false;
}

/// Match the `linalg.matmul` + `linalg.add` accumulator pattern and rewrite to
/// a single `nki.nc_matmul`.
struct MatmulAddToNcMatmul : public OpRewritePattern<linalg::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    // Tensor-semantic linalg.add has exactly one result.
    if (addOp.getNumResults() != 1)
      return rewriter.notifyMatchFailure(addOp,
                                         "expected tensor-semantic linalg.add");

    if (addOp.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(addOp, "expected 2 inputs");

    Value addLhs = addOp.getInputs()[0];
    Value addRhs = addOp.getInputs()[1];

    // Find which side is produced by a linalg.matmul; the other side is the
    // pre-existing accumulator value.
    linalg::MatmulOp matmulOp;
    Value accumulator;
    if ((matmulOp = addLhs.getDefiningOp<linalg::MatmulOp>())) {
      accumulator = addRhs;
    } else if ((matmulOp = addRhs.getDefiningOp<linalg::MatmulOp>())) {
      accumulator = addLhs;
    } else {
      return rewriter.notifyMatchFailure(
          addOp, "neither linalg.add input is produced by a linalg.matmul");
    }

    // Refuse to fuse if the matmul has additional consumers; otherwise we
    // would silently drop those uses or duplicate work.
    if (!matmulOp->hasOneUse())
      return rewriter.notifyMatchFailure(
          addOp, "matmul has multiple uses; cannot fuse safely");

    // The matmul must produce a single tensor result.
    if (matmulOp.getNumResults() != 1)
      return rewriter.notifyMatchFailure(matmulOp, "matmul has no tensor result");

    // The matmul's `outs` should be a fresh zero-initialized buffer
    // (`linalg.fill 0 -> tensor.empty`). This is what tells us the matmul is
    // computing a fresh product, and so its result is exactly `lhs @ rhs`.
    if (matmulOp.getOutputs().size() != 1)
      return rewriter.notifyMatchFailure(matmulOp,
                                         "matmul has unexpected outs arity");
    auto fillOp = matmulOp.getOutputs()[0].getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return rewriter.notifyMatchFailure(addOp,
                                         "matmul output is not a linalg.fill");
    if (fillOp.getInputs().size() != 1 ||
        !isZeroConstant(fillOp.getInputs()[0]))
      return rewriter.notifyMatchFailure(addOp,
                                         "matmul fill init is not zero");

    // Element types must match across the fused chain.
    Value lhs = matmulOp.getInputs()[0];
    Value rhs = matmulOp.getInputs()[1];
    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    auto accTy = dyn_cast<RankedTensorType>(accumulator.getType());
    auto resTy = dyn_cast<RankedTensorType>(addOp.getResult(0).getType());
    if (!lhsTy || !rhsTy || !accTy || !resTy)
      return rewriter.notifyMatchFailure(addOp, "non-ranked tensor operands");

    // Build the fused op. The result type is the linalg.add result type
    // (which equals the accumulator type by DPS construction).
    auto fused = NcMatmulOp::create(rewriter, addOp.getLoc(), resTy, lhs, rhs,
                                    accumulator);

    rewriter.replaceOp(addOp, fused.getResult());

    // The matmul (now use-less) and the chain of linalg.fill / tensor.empty
    // become dead. Erase the matmul explicitly so the greedy driver does not
    // need an extra DCE round; the fill / empty will be cleaned up by
    // greedy folding.
    rewriter.eraseOp(matmulOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//
struct LinalgToNKIPass : public impl::LinalgToNKIBase<LinalgToNKIPass> {
  using impl::LinalgToNKIBase<LinalgToNKIPass>::LinalgToNKIBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    populateLinalgToNKIPatterns(patterns);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void populateLinalgToNKIPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulAddToNcMatmul>(patterns.getContext());
}

} // namespace nki
} // namespace mlir
