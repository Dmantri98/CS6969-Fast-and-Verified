//===----------------------------------------------------------------------===//
// nki-fuse-dma: collapse triton-shared masked-load chains into nki.dma_copy.
//
//   %rc   = memref.reinterpret_cast %src to
//             offset: [%off], sizes: [M, N], strides: [%s0, %s1]
//   %buf  = memref.alloc()
//   %ssrc = memref.subview %rc[0,0][%vr, N][1,1]
//   %sdst = memref.subview %buf[0,0][%vr, N][1,1]
//   memref.copy %ssrc, %sdst
//   %tile = bufferization.to_tensor %buf restrict writable
//                : memref<MxNxT> to tensor<MxNxT>
//
//   ===>
//
//   %tile = nki.dma_copy %src [%off] strides [%s0, %s1]
//                : memref<*xT> to tensor<MxNxT>
//===----------------------------------------------------------------------===//
#include "Transforms/Passes.h"

#include "IR/NKIDialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "nki-fuse-dma"

namespace mlir {
namespace nki {

#define GEN_PASS_DEF_NKIFUSEDMA
#include "Transforms/Passes.h.inc"

namespace {

/// Walk the users of `alloc` looking for the (subview, copy) pair that writes
/// into it. Returns the matching ops or {nullptr, nullptr} if none found or
/// the structure is ambiguous.
static std::pair<memref::SubViewOp, memref::CopyOp>
findCopyDestSubview(memref::AllocOp alloc) {
  memref::SubViewOp foundSv;
  memref::CopyOp foundCopy;
  for (Operation *user : alloc->getUsers()) {
    auto sv = dyn_cast<memref::SubViewOp>(user);
    if (!sv)
      continue;
    for (Operation *svUser : sv.getResult().getUsers()) {
      auto copy = dyn_cast<memref::CopyOp>(svUser);
      if (!copy)
        continue;
      if (copy.getTarget() != sv.getResult())
        continue;
      // Refuse to commit if there's already a candidate -- ambiguous match.
      if (foundCopy)
        return {nullptr, nullptr};
      foundSv = sv;
      foundCopy = copy;
    }
  }
  return {foundSv, foundCopy};
}

struct DmaChainToNkiDmaCopy
    : public OpRewritePattern<bufferization::ToTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::ToTensorOp toTensor,
                                PatternRewriter &rewriter) const override {
    // Result must be a statically-shaped ranked tensor (we encode the tile
    // extent in the result type).
    auto resTy = dyn_cast<RankedTensorType>(toTensor.getType());
    if (!resTy || !resTy.hasStaticShape())
      return rewriter.notifyMatchFailure(toTensor,
                                         "result tensor not statically shaped");

    // Source must be a fresh memref.alloc. (`getBuffer()` returns a
    // TypedValue<BufferLikeType>, which converts to Value implicitly.)
    Value buffer = toTensor.getBuffer();
    auto alloc = buffer.getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return rewriter.notifyMatchFailure(toTensor,
                                         "source is not a memref.alloc");

    // The alloc must produce a statically shaped, identity-strided buffer.
    auto allocTy = dyn_cast<MemRefType>(alloc.getType());
    if (!allocTy || !allocTy.hasStaticShape())
      return rewriter.notifyMatchFailure(toTensor,
                                         "alloc is not statically shaped");

    // Find the copy that writes into the alloc.
    auto [destSv, copy] = findCopyDestSubview(alloc);
    if (!copy)
      return rewriter.notifyMatchFailure(
          toTensor, "no unambiguous memref.copy writes into the alloc");

    // Source side: subview of a reinterpret_cast.
    auto srcSv = copy.getSource().getDefiningOp<memref::SubViewOp>();
    if (!srcSv)
      return rewriter.notifyMatchFailure(
          toTensor, "memref.copy source is not a memref.subview");
    auto reinterp =
        srcSv.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterp)
      return rewriter.notifyMatchFailure(
          toTensor, "subview source is not a memref.reinterpret_cast");

    // The original base must be an unranked memref so we can describe it as
    // a flat pointer + offset + strides in nki.dma_copy.
    Value src = reinterp.getSource();
    if (!isa<UnrankedMemRefType>(src.getType()))
      return rewriter.notifyMatchFailure(
          toTensor, "reinterpret_cast source is not an unranked memref");

    // Extract the dynamic offset and dynamic strides from the
    // reinterpret_cast. We require all of them to be dynamic operands (the
    // shape we get from triton-shared) -- bail out if anything is encoded as
    // a static attribute.
    if (reinterp.getOffsets().size() != 1)
      return rewriter.notifyMatchFailure(
          toTensor, "reinterpret_cast does not have exactly one dynamic offset");
    Value offset = reinterp.getOffsets()[0];

    SmallVector<Value> strides(reinterp.getStrides());
    if (strides.size() != static_cast<size_t>(resTy.getRank()))
      return rewriter.notifyMatchFailure(
          toTensor, "reinterpret_cast stride count does not match tile rank");

    // Build the fused op.
    auto dma = DmaCopyOp::create(rewriter, toTensor.getLoc(), resTy, src,
                                 offset, strides);

    rewriter.replaceOp(toTensor, dma.getResult());

    // Erase the side-effecting memref machinery; the greedy driver won't DCE
    // it on its own. Order matters: copy first (consumer), then the subviews,
    // then the alloc / reinterpret_cast.
    rewriter.eraseOp(copy);
    if (destSv->use_empty())
      rewriter.eraseOp(destSv);
    if (srcSv->use_empty())
      rewriter.eraseOp(srcSv);
    if (alloc->use_empty())
      rewriter.eraseOp(alloc);
    if (reinterp->use_empty())
      rewriter.eraseOp(reinterp);

    return success();
  }
};

struct NKIFuseDmaPass : public impl::NKIFuseDmaBase<NKIFuseDmaPass> {
  using impl::NKIFuseDmaBase<NKIFuseDmaPass>::NKIFuseDmaBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    populateNKIFuseDmaPatterns(patterns);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void populateNKIFuseDmaPatterns(RewritePatternSet &patterns) {
  patterns.add<DmaChainToNkiDmaCopy>(patterns.getContext());
}

} // namespace nki
} // namespace mlir
