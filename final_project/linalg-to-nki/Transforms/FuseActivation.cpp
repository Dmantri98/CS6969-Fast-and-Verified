//===----------------------------------------------------------------------===//
// nki-fuse-activation: fold a pointwise activation sitting between the matmul
// K-loop result and the final `nki.dma_store` into the store's `activation`
// attribute, which the Python emitter renders as
// `nisa.activation(op=nl.<name>, data=psum, dtype=...)` in place of the
// PSUM->SBUF `nisa.tensor_copy`.
//
// Supported activation families (body shape -> NISA op name):
//
//   Unary math ops (linalg.generic with 1 input, body = single math op):
//     math.exp   -> "exp"
//     math.log   -> "log"
//     math.sqrt  -> "sqrt"
//     math.rsqrt -> "rsqrt"
//     math.absf  -> "abs"
//     math.tanh  -> "tanh"
//     math.erf   -> "erf"
//     math.sin   -> "sin"
//     math.cos   -> "cos"
//     math.tan   -> "tan"
//
//   Binary ops with a zero-filled tensor (one of the two inputs must be
//   `linalg.fill ins(0.0) outs(tensor.empty)`):
//     arith.maxnumf -> "relu"
//
//   Structured linalg ops (legacy pattern, specialized form):
//     linalg.max ins(x, zeros) -> "relu"
//
// Every other `linalg.generic` left in the IR after this pass runs is flagged
// by the companion `-nki-check-unsupported-elementwise` pass with a readable
// diagnostic listing the supported set (this is the "graceful fallback" --
// the alternative is silent downstream translator failure).
//===----------------------------------------------------------------------===//
#include "Transforms/Passes.h"

#include "IR/NKIDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "nki-fuse-activation"

namespace mlir {
namespace nki {

#define GEN_PASS_DEF_NKIFUSEACTIVATION
#define GEN_PASS_DEF_NKICHECKUNSUPPORTEDELEMENTWISE
#include "Transforms/Passes.h.inc"

namespace {

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

/// Returns true if `v` is produced by `linalg.fill ins(%zero) outs(tensor.empty)`.
static bool isZeroFilledTensor(Value v) {
  auto fill = v.getDefiningOp<linalg::FillOp>();
  if (!fill)
    return false;
  if (fill.getInputs().size() != 1 || !isZeroConstant(fill.getInputs()[0]))
    return false;
  return true;
}

/// One-line summary of supported activations used by the check-unsupported
/// pass diagnostic note.
static StringRef supportedActivationsNote() {
  return "supported: exp, log, sqrt, rsqrt, abs, tanh, erf, sin, cos, tan "
         "(single-op math.* bodies), relu (arith.maxnumf(x, 0)), "
         "and linalg.max(x, zeros)";
}

/// Map a body-local unary math op -> NISA activation name.
static std::optional<StringRef> unaryBodyOpToName(Operation *bodyOp) {
  if (isa<math::ExpOp>(bodyOp))   return StringRef("exp");
  if (isa<math::LogOp>(bodyOp))   return StringRef("log");
  if (isa<math::SqrtOp>(bodyOp))  return StringRef("sqrt");
  if (isa<math::RsqrtOp>(bodyOp)) return StringRef("rsqrt");
  if (isa<math::AbsFOp>(bodyOp))  return StringRef("abs");
  if (isa<math::TanhOp>(bodyOp))  return StringRef("tanh");
  if (isa<math::ErfOp>(bodyOp))   return StringRef("erf");
  if (isa<math::SinOp>(bodyOp))   return StringRef("sin");
  if (isa<math::CosOp>(bodyOp))   return StringRef("cos");
  if (isa<math::TanOp>(bodyOp))   return StringRef("tan");
  return std::nullopt;
}

/// Recognize `g` as a supported activation. On success, returns the NISA
/// activation name and sets `input` to the tensor value whose contents
/// flow through the activation (i.e. the value the downstream `nki.dma_store`
/// should consume directly once fused).
static std::optional<StringRef> recognizeActivation(linalg::GenericOp g,
                                                    Value &input) {
  // All iterators must be parallel, all maps identity, single init.
  for (auto it : g.getIteratorTypesArray())
    if (it != utils::IteratorType::parallel)
      return std::nullopt;
  for (AffineMap m : g.getIndexingMapsArray())
    if (!m.isIdentity())
      return std::nullopt;
  if (g.getNumDpsInits() != 1)
    return std::nullopt;

  // Body must be exactly one op + linalg.yield of its result.
  Block &body = g.getRegion().front();
  auto bodyOps = body.without_terminator();
  auto begin = bodyOps.begin();
  auto end = bodyOps.end();
  if (begin == end || std::next(begin) != end)
    return std::nullopt;
  Operation *bodyOp = &*begin;
  auto yield = cast<linalg::YieldOp>(body.getTerminator());
  if (yield.getNumOperands() != 1 ||
      yield.getOperand(0) != bodyOp->getResult(0))
    return std::nullopt;

  int numInputs = g.getNumDpsInputs();

  // ---- Unary: single-op math body on block arg 0.
  if (numInputs == 1) {
    if (bodyOp->getNumOperands() != 1 ||
        bodyOp->getOperand(0) != body.getArgument(0))
      return std::nullopt;
    auto name = unaryBodyOpToName(bodyOp);
    if (!name)
      return std::nullopt;
    input = g.getDpsInputs()[0];
    return *name;
  }

  // ---- Binary: currently just arith.maxnumf(x, zero) => relu.
  if (numInputs == 2) {
    if (bodyOp->getNumOperands() != 2)
      return std::nullopt;
    Value l = bodyOp->getOperand(0);
    Value r = bodyOp->getOperand(1);
    BlockArgument ba0 = body.getArgument(0);
    BlockArgument ba1 = body.getArgument(1);
    bool operandsAreArgs = (l == ba0 && r == ba1) || (l == ba1 && r == ba0);
    if (!operandsAreArgs)
      return std::nullopt;

    if (!isa<arith::MaxNumFOp>(bodyOp))
      return std::nullopt;

    Value a = g.getDpsInputs()[0];
    Value b = g.getDpsInputs()[1];
    if (isZeroFilledTensor(a))
      input = b;
    else if (isZeroFilledTensor(b))
      input = a;
    else
      return std::nullopt;
    return StringRef("relu");
  }

  return std::nullopt;
}

/// Legacy pattern: fold `linalg.max ins(%x, %zeros)` consumed by
/// `nki.dma_store` into `nki.dma_store %x ... {activation = "relu"}`.
/// Kept for the structured-op form produced by `-linalg-specialize-generic-ops`
/// when it succeeds (triton-shared's output does not currently get specialized
/// for arith.maxnumf, but a future linalg upgrade might emit linalg.max
/// directly).
struct MaxZeroToReluStore : public OpRewritePattern<DmaStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DmaStoreOp store,
                                PatternRewriter &rewriter) const override {
    if (store.getActivation() != "none")
      return rewriter.notifyMatchFailure(store, "activation already set");

    auto maxOp = store.getTile().getDefiningOp<linalg::MaxOp>();
    if (!maxOp)
      return rewriter.notifyMatchFailure(store,
                                         "tile is not produced by linalg.max");

    if (maxOp.getInputs().size() != 2)
      return rewriter.notifyMatchFailure(maxOp, "linalg.max not binary");

    if (!maxOp->hasOneUse())
      return rewriter.notifyMatchFailure(maxOp,
                                         "linalg.max has multiple consumers");

    Value a = maxOp.getInputs()[0];
    Value b = maxOp.getInputs()[1];
    Value nonZero;
    if (isZeroFilledTensor(a))
      nonZero = b;
    else if (isZeroFilledTensor(b))
      nonZero = a;
    else
      return rewriter.notifyMatchFailure(
          maxOp, "neither linalg.max input is a zero-filled tensor");

    rewriter.setInsertionPoint(store);
    DmaStoreOp::create(rewriter, store.getLoc(), nonZero, store.getDestination(),
                       store.getOffset(), store.getStrides(),
                       rewriter.getStringAttr("relu"));
    rewriter.eraseOp(store);
    if (maxOp->use_empty())
      rewriter.eraseOp(maxOp);
    return success();
  }
};

/// Registry-driven pattern: fold a recognized activation `linalg.generic` into
/// the downstream `nki.dma_store`'s `activation` attribute.
struct GenericActivationToStore : public OpRewritePattern<DmaStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DmaStoreOp store,
                                PatternRewriter &rewriter) const override {
    if (store.getActivation() != "none")
      return rewriter.notifyMatchFailure(store, "activation already set");

    auto g = store.getTile().getDefiningOp<linalg::GenericOp>();
    if (!g)
      return rewriter.notifyMatchFailure(store,
                                         "tile not produced by linalg.generic");

    if (!g->hasOneUse())
      return rewriter.notifyMatchFailure(g,
                                         "linalg.generic has multiple consumers");

    Value input;
    auto maybeName = recognizeActivation(g, input);
    if (!maybeName)
      return rewriter.notifyMatchFailure(
          g, "body is not a recognized NISA activation");

    rewriter.setInsertionPoint(store);
    DmaStoreOp::create(rewriter, store.getLoc(), input, store.getDestination(),
                       store.getOffset(), store.getStrides(),
                       rewriter.getStringAttr(*maybeName));
    rewriter.eraseOp(store);
    if (g->use_empty())
      rewriter.eraseOp(g);
    return success();
  }
};

struct NKIFuseActivationPass
    : public impl::NKIFuseActivationBase<NKIFuseActivationPass> {
  using impl::NKIFuseActivationBase<NKIFuseActivationPass>::NKIFuseActivationBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    populateNKIFuseActivationPatterns(patterns);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

/// Post-fusion verifier: after `-nki-fuse-activation` has run, any surviving
/// `linalg.generic` / `linalg.max` that the Python emitter cannot handle is
/// flagged here with a readable diagnostic. Without this pass, an unsupported
/// activation silently survives until the translator later fails in a much
/// less informative way.
struct NKICheckUnsupportedElementwisePass
    : public impl::NKICheckUnsupportedElementwiseBase<
          NKICheckUnsupportedElementwisePass> {
  using impl::NKICheckUnsupportedElementwiseBase<
      NKICheckUnsupportedElementwisePass>::NKICheckUnsupportedElementwiseBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    bool fail = false;

    func.walk([&](linalg::GenericOp g) {
      Value dummy;
      auto recognized = recognizeActivation(g, dummy);
      if (recognized) {
        // Recognized but not fused (e.g. multi-use, or no dma_store consumer).
        InFlightDiagnostic d = g.emitOpError()
            << "recognized activation '" << *recognized
            << "' was not fused into an nki.dma_store";
        d.attachNote() << "likely cause: the linalg.generic has multiple "
                          "consumers, or the downstream op is not nki.dma_store";
        fail = true;
        return;
      }
      InFlightDiagnostic d = g.emitOpError()
          << "unsupported elementwise linalg.generic survived "
             "-nki-fuse-activation; the linalg-to-nki-translate Python emitter "
             "has no rule for this body";
      d.attachNote() << supportedActivationsNote();
      fail = true;
    });

    func.walk([&](linalg::MaxOp m) {
      InFlightDiagnostic d = m.emitOpError()
          << "linalg.max survived -nki-fuse-activation";
      d.attachNote() << "expected shape: linalg.max ins(%x, %zeros) with a "
                        "single consumer that is an nki.dma_store";
      fail = true;
    });

    // Any other linalg structured op (e.g. linalg.add used outside the
    // matmul-accumulator pattern) is equally unsupported. `linalg.fill` is
    // exempt -- it's a legitimate helper that materializes zero/const seeds,
    // and if it's feeding an unsupported op, that op itself is already
    // flagged above.
    func.walk([&](Operation *op) {
      if (isa<linalg::GenericOp, linalg::MaxOp, linalg::FillOp>(op))
        return;
      if (!isa<linalg::LinalgOp>(op))
        return;
      InFlightDiagnostic d = op->emitOpError()
          << "unsupported structured linalg op survived the conversion "
             "pipeline";
      d.attachNote() << supportedActivationsNote();
      fail = true;
    });

    if (fail)
      signalPassFailure();
  }
};

} // namespace

void populateNKIFuseActivationPatterns(RewritePatternSet &patterns) {
  patterns.add<MaxZeroToReluStore, GenericActivationToStore>(
      patterns.getContext());
}

} // namespace nki
} // namespace mlir
