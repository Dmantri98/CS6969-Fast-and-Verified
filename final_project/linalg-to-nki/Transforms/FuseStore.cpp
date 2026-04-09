//===----------------------------------------------------------------------===//
// nki-fuse-store: collapse triton-shared masked-store chains into nki.dma_store.
//
//   %rc    = memref.reinterpret_cast %dst to
//              offset: [%off], sizes: [M, N], strides: [%s0, %s1]
//   %slice = tensor.extract_slice %tile[0, 0] [%vr, %vc] [1, 1]
//   %sdst  = memref.subview %rc[0, 0] [%vr, %vc] [1, 1]
//   bufferization.materialize_in_destination %slice in writable %sdst
//
//   ===>
//
//   nki.dma_store %tile into %dst [%off] strides [%s0, %s1]
//                : tensor<MxNxT> into memref<*xT>
//
// The dynamic mask extents are intentionally dropped; the source tile shape
// is the static result type and the upstream Triton kernel is the source of
// truth on bounds.
//===----------------------------------------------------------------------===//
#include "Transforms/Passes.h"

#include "IR/NKIDialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "nki-fuse-store"

namespace mlir {
namespace nki {

#define GEN_PASS_DEF_NKIFUSESTORE
#include "Transforms/Passes.h.inc"

namespace {

struct StoreChainToNkiDmaStore
    : public OpRewritePattern<bufferization::MaterializeInDestinationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  matchAndRewrite(bufferization::MaterializeInDestinationOp matOp,
                  PatternRewriter &rewriter) const override {
    // Source side: a tensor.extract_slice of the original (statically shaped)
    // tile -- we drop the mask and store the full extent.
    auto extractSlice =
        matOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractSlice)
      return rewriter.notifyMatchFailure(
          matOp, "source is not a tensor.extract_slice");

    auto tileTy =
        dyn_cast<RankedTensorType>(extractSlice.getSource().getType());
    if (!tileTy || !tileTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          matOp, "extract_slice source tile is not statically shaped");
    Value tile = extractSlice.getSource();

    // Destination side: a memref.subview of a memref.reinterpret_cast.
    Value destOperand = matOp.getDest();
    auto destSv = destOperand.getDefiningOp<memref::SubViewOp>();
    if (!destSv)
      return rewriter.notifyMatchFailure(
          matOp, "destination is not a memref.subview");

    auto reinterp =
        destSv.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterp)
      return rewriter.notifyMatchFailure(
          matOp, "subview source is not a memref.reinterpret_cast");

    // The original base must be an unranked memref so we can describe it as
    // a flat pointer + offset + strides in nki.dma_store.
    Value dest = reinterp.getSource();
    if (!isa<UnrankedMemRefType>(dest.getType()))
      return rewriter.notifyMatchFailure(
          matOp, "reinterpret_cast source is not an unranked memref");

    // Extract the dynamic offset and dynamic strides from the
    // reinterpret_cast. We require both to be dynamic operands.
    if (reinterp.getOffsets().size() != 1)
      return rewriter.notifyMatchFailure(
          matOp, "reinterpret_cast does not have exactly one dynamic offset");
    Value offset = reinterp.getOffsets()[0];

    SmallVector<Value> strides(reinterp.getStrides());
    if (strides.size() != static_cast<size_t>(tileTy.getRank()))
      return rewriter.notifyMatchFailure(
          matOp, "reinterpret_cast stride count does not match tile rank");

    // Build the fused store op (no result -- it's effectful).
    DmaStoreOp::create(rewriter, matOp.getLoc(), tile, dest, offset, strides);

    // The materialize op produces no result we need to replace; just erase.
    rewriter.eraseOp(matOp);

    // Clean up the side-effecting / dead memref + tensor machinery; the
    // greedy driver won't DCE side-effecting ops on its own.
    if (destSv->use_empty())
      rewriter.eraseOp(destSv);
    if (extractSlice->use_empty())
      rewriter.eraseOp(extractSlice);
    if (reinterp->use_empty())
      rewriter.eraseOp(reinterp);

    return success();
  }
};

struct NKIFuseStorePass : public impl::NKIFuseStoreBase<NKIFuseStorePass> {
  using impl::NKIFuseStoreBase<NKIFuseStorePass>::NKIFuseStoreBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    populateNKIFuseStorePatterns(patterns);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void populateNKIFuseStorePatterns(RewritePatternSet &patterns) {
  patterns.add<StoreChainToNkiDmaStore>(patterns.getContext());
}

} // namespace nki
} // namespace mlir
