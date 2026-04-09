//===----------------------------------------------------------------------===//
// nki-canonicalize-pid-loops: recover m/n tile loops from triton's program_id.
//
// Triton lowers `tl.program_id(0)` plus the standard
//   pid_m = pid // num_pid_n
//   pid_n = pid % num_pid_n
// into a flat divsi/remsi pair on a function argument. After triton-shared
// the IR looks like:
//
//   %2 = arith.addi %N,   %c63       : i32        // (N + BLOCK_N - 1)
//   %3 = arith.divsi %2,  %cBLOCK_N  : i32        // num_pid_n
//   %4 = arith.divsi %pid,%3         : i32        // pid_m
//   %5 = arith.remsi %pid,%3         : i32        // pid_n
//   %6 = arith.muli  %4,  %cBLOCK_M  : i32        // m_off
//   %7 = arith.muli  %5,  %cBLOCK_N  : i32        // n_off
//   ... rest of body uses %6, %7 ...
//
// After this pass:
//
//   %2 = arith.addi %N, %c63
//   %3 = arith.divsi %2, %cBLOCK_N            // num_pid_n (kept as bound)
//   %newAdd = arith.addi %M, %c(BLOCK_M - 1)
//   %nm     = arith.divsi %newAdd, %cBLOCK_M  // num_pid_m
//   scf.for %pid_m = %c0 to %nm step %c1 : i32 {
//     scf.for %pid_n = %c0 to %3 step %c1 : i32 {
//       %newM = arith.muli %pid_m, %cBLOCK_M : i32
//       %newN = arith.muli %pid_n, %cBLOCK_N : i32
//       ... rest of body, with %6 -> %newM, %7 -> %newN ...
//     }
//   }
//
// IMPORTANT: this pass MUST run BEFORE `nki-fuse-dma`. We use the load-side
// bounds-check `arith.minsi` chain inside the k-loop to identify which i32
// function argument is `M`. Once `nki-fuse-dma` collapses the bounds-check
// arith into a single `nki.dma_copy`, that information is lost.
//===----------------------------------------------------------------------===//
#include "Transforms/Passes.h"

#include "IR/NKIDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "nki-canonicalize-pid-loops"

namespace mlir {
namespace nki {

#define GEN_PASS_DEF_NKICANONICALIZEPIDLOOPS
#include "Transforms/Passes.h.inc"

namespace {

struct PidPattern {
  arith::DivSIOp divsiOp; // pid_m = pid / num_pid_n
  arith::RemSIOp remsiOp; // pid_n = pid % num_pid_n
  arith::MulIOp mulMOp;   // m_off = pid_m * BLOCK_M
  arith::MulIOp mulNOp;   // n_off = pid_n * BLOCK_N
  Value pid;              // %pid (i32 func arg)
  Value divisor;          // num_pid_n SSA value (kept as inner loop bound)
  Value nVal;             // %N i32 SSA value
  int64_t blockN;         // BLOCK_N constant
  int64_t blockM;         // BLOCK_M constant
};

static std::optional<int64_t> matchConstantInt(Value v) {
  IntegerAttr attr;
  if (matchPattern(v, m_Constant(&attr)))
    return attr.getInt();
  return std::nullopt;
}

/// Recover (N, BLOCK_N) from `divisor = arith.divsi (arith.addi %N, c) c'`
/// where `c == c' - 1`.
static LogicalResult parseDivisor(Value divisor, Value &nVal,
                                  int64_t &blockN) {
  auto divOp = divisor.getDefiningOp<arith::DivSIOp>();
  if (!divOp)
    return failure();
  auto blockNConst = matchConstantInt(divOp.getRhs());
  if (!blockNConst || *blockNConst <= 0)
    return failure();
  blockN = *blockNConst;
  auto addOp = divOp.getLhs().getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return failure();
  Value lhs = addOp.getLhs();
  Value rhs = addOp.getRhs();
  auto lhsConst = matchConstantInt(lhs);
  auto rhsConst = matchConstantInt(rhs);
  Value nCandidate;
  int64_t addConst;
  if (lhsConst && !rhsConst) {
    addConst = *lhsConst;
    nCandidate = rhs;
  } else if (rhsConst && !lhsConst) {
    addConst = *rhsConst;
    nCandidate = lhs;
  } else {
    return failure();
  }
  if (addConst != blockN - 1)
    return failure();
  nVal = nCandidate;
  return success();
}

/// Walk forward from %m_off through index_cast -> addi -> minsi to find the
/// i32 function argument that the load-side bounds-check uses as `M`.
static BlockArgument findMArgFromBoundsCheck(Value mOffI32) {
  for (Operation *u : mOffI32.getUsers()) {
    auto cast1 = dyn_cast<arith::IndexCastOp>(u);
    if (!cast1)
      continue;
    for (Operation *u2 : cast1.getResult().getUsers()) {
      auto add = dyn_cast<arith::AddIOp>(u2);
      if (!add)
        continue;
      for (Operation *u3 : add.getResult().getUsers()) {
        auto minOp = dyn_cast<arith::MinSIOp>(u3);
        if (!minOp)
          continue;
        Value other = (minOp.getLhs() == add.getResult()) ? minOp.getRhs()
                                                          : minOp.getLhs();
        auto castM = other.getDefiningOp<arith::IndexCastOp>();
        if (!castM)
          continue;
        if (auto barg = dyn_cast<BlockArgument>(castM.getIn()))
          if (barg.getType().isInteger(32))
            return barg;
      }
    }
  }
  return BlockArgument();
}

static LogicalResult detectPidPattern(func::FuncOp func, PidPattern &out) {
  arith::DivSIOp foundDivsi;
  arith::RemSIOp foundRemsi;

  // Look for a divsi whose lhs is an i32 function argument and which has a
  // matching remsi consumer of the same (lhs, rhs).
  func.walk([&](arith::DivSIOp d) {
    if (foundDivsi)
      return;
    if (!d.getType().isInteger(32))
      return;
    auto barg = dyn_cast<BlockArgument>(d.getLhs());
    if (!barg)
      return;
    for (Operation *u : barg.getUsers()) {
      auto r = dyn_cast<arith::RemSIOp>(u);
      if (!r)
        continue;
      if (r.getLhs() != d.getLhs() || r.getRhs() != d.getRhs())
        continue;
      foundDivsi = d;
      foundRemsi = r;
      return;
    }
  });
  if (!foundDivsi || !foundRemsi)
    return failure();

  Value nVal;
  int64_t blockN;
  if (failed(parseDivisor(foundDivsi.getRhs(), nVal, blockN)))
    return failure();

  // The divsi result must have exactly one consumer, an arith.muli (mulM).
  if (!foundDivsi.getResult().hasOneUse())
    return failure();
  auto mulMOp = dyn_cast<arith::MulIOp>(*foundDivsi.getResult().user_begin());
  if (!mulMOp)
    return failure();
  if (!foundRemsi.getResult().hasOneUse())
    return failure();
  auto mulNOp = dyn_cast<arith::MulIOp>(*foundRemsi.getResult().user_begin());
  if (!mulNOp)
    return failure();

  Value mulMOther = (mulMOp.getLhs() == foundDivsi.getResult())
                        ? mulMOp.getRhs()
                        : mulMOp.getLhs();
  auto blockMConst = matchConstantInt(mulMOther);
  if (!blockMConst || *blockMConst <= 0)
    return failure();

  Value mulNOther = (mulNOp.getLhs() == foundRemsi.getResult())
                        ? mulNOp.getRhs()
                        : mulNOp.getLhs();
  auto blockNFromMul = matchConstantInt(mulNOther);
  if (!blockNFromMul || *blockNFromMul != blockN)
    return failure();

  out.divsiOp = foundDivsi;
  out.remsiOp = foundRemsi;
  out.mulMOp = mulMOp;
  out.mulNOp = mulNOp;
  out.pid = foundDivsi.getLhs();
  out.divisor = foundDivsi.getRhs();
  out.nVal = nVal;
  out.blockN = blockN;
  out.blockM = *blockMConst;
  return success();
}

struct NKICanonicalizePidLoopsPass
    : public impl::NKICanonicalizePidLoopsBase<NKICanonicalizePidLoopsPass> {
  using impl::NKICanonicalizePidLoopsBase<
      NKICanonicalizePidLoopsPass>::NKICanonicalizePidLoopsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    PidPattern p;
    if (failed(detectPidPattern(func, p)))
      return; // No pattern; nothing to do.

    // Identify M.
    BlockArgument mArg = findMArgFromBoundsCheck(p.mulMOp.getResult());
    if (!mArg) {
      // Fall back: M is the i32 arg immediately preceding N (triton-shared
      // convention). This only triggers if the bounds-check is missing,
      // e.g. someone ran nki-fuse-dma first.
      auto nBarg = dyn_cast<BlockArgument>(p.nVal);
      if (!nBarg || nBarg.getArgNumber() == 0) {
        func.emitWarning()
            << "nki-canonicalize-pid-loops: could not identify M function "
               "argument; pass not applied (run before nki-fuse-dma)";
        return;
      }
      auto cand = func.getArgument(nBarg.getArgNumber() - 1);
      if (!cand.getType().isInteger(32)) {
        func.emitWarning() << "nki-canonicalize-pid-loops: M arg fallback "
                              "is not i32; pass not applied";
        return;
      }
      mArg = cand;
    }

    OpBuilder builder(&getContext());
    Location loc = func.getLoc();
    Block &entryBlock = func.getBody().front();

    // Hoist all loop-control constants and the M cdiv computation to the top
    // of the entry block (they need to dominate the new outer scf.for).
    builder.setInsertionPointToStart(&entryBlock);
    Value c0 =
        builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(0));
    Value c1 =
        builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
    Value cBlockMm1 = builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(p.blockM - 1));
    Value cBlockM = builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(p.blockM));
    Value cBlockN = builder.create<arith::ConstantOp>(
        loc, builder.getI32IntegerAttr(p.blockN));
    Value mPlusBlockMm1 =
        builder.create<arith::AddIOp>(loc, mArg, cBlockMm1);
    Value numPidM =
        builder.create<arith::DivSIOp>(loc, mPlusBlockMm1, cBlockM);

    // Create the outer + inner scf.for loops just BEFORE mulMOp. Both loops
    // produce no results (the body operates by side effect / iter_args of
    // the inner k-loop).
    builder.setInsertionPoint(p.mulMOp);
    auto outerLoop = builder.create<scf::ForOp>(loc, c0, numPidM, c1);
    Block *outerBody = outerLoop.getBody();
    builder.setInsertionPointToStart(outerBody);
    auto innerLoop =
        builder.create<scf::ForOp>(loc, c0, p.divisor, c1);
    Block *innerBody = innerLoop.getBody();

    // Inside the inner loop, recompute m_off / n_off from the IVs.
    builder.setInsertionPointToStart(innerBody);
    Value newMOff = builder.create<arith::MulIOp>(
        loc, outerLoop.getInductionVar(), cBlockM);
    Value newNOff = builder.create<arith::MulIOp>(
        loc, innerLoop.getInductionVar(), cBlockN);

    // RAUW the original m_off / n_off values with the in-loop versions.
    p.mulMOp.getResult().replaceAllUsesWith(newMOff);
    p.mulNOp.getResult().replaceAllUsesWith(newNOff);

    // Move every operation from "after mulNOp" up to (but not including) the
    // function terminator into the inner loop body, before its terminator.
    Operation *terminator = entryBlock.getTerminator();
    Block::iterator moveStart = std::next(Block::iterator(p.mulNOp));
    Block::iterator moveEnd = Block::iterator(terminator);
    if (moveStart != moveEnd) {
      Block::iterator innerInsert =
          innerBody->getTerminator()->getIterator();
      innerBody->getOperations().splice(innerInsert,
                                        entryBlock.getOperations(),
                                        moveStart, moveEnd);
    }

    // Erase the now-dead pid math.
    p.mulMOp.erase();
    p.mulNOp.erase();
    p.divsiOp.erase();
    p.remsiOp.erase();
  }
};

} // namespace

} // namespace nki
} // namespace mlir
