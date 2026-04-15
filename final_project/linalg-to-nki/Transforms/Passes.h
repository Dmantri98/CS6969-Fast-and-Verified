//===----------------------------------------------------------------------===//
// Public header for linalg-to-nki conversion passes.
//===----------------------------------------------------------------------===//
#ifndef LINALG_TO_NKI_TRANSFORMS_PASSES_H
#define LINALG_TO_NKI_TRANSFORMS_PASSES_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace nki {

#define GEN_PASS_DECL
#include "Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Transforms/Passes.h.inc"

/// Populate the rewrite-pattern set with linalg → nki rewrites.
void populateLinalgToNKIPatterns(RewritePatternSet &patterns);

/// Populate patterns that fuse triton-shared masked-load chains into
/// `nki.dma_copy`.
void populateNKIFuseDmaPatterns(RewritePatternSet &patterns);

/// Populate patterns that fuse triton-shared masked-store chains into
/// `nki.dma_store`.
void populateNKIFuseStorePatterns(RewritePatternSet &patterns);

/// Populate patterns that fold linalg.fill PSUM seeds into nki.psum_alloc.
void populateNKIFoldPsumInitPatterns(RewritePatternSet &patterns);

/// Populate patterns that fold a pointwise activation (e.g. relu expressed as
/// linalg.max against zeros) into the `activation` attribute of the
/// downstream `nki.dma_store`.
void populateNKIFuseActivationPatterns(RewritePatternSet &patterns);

} // namespace nki
} // namespace mlir

#endif // LINALG_TO_NKI_TRANSFORMS_PASSES_H
