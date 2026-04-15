//===----------------------------------------------------------------------===//
// linalg-to-nki-translate: emit Python NKI from a fully-lowered MLIR module.
//
// Reads an MLIR file (expected to be the output of running linalg-to-nki-opt
// with the full canonicalization pipeline:
//   -nki-canonicalize-pid-loops
//   -linalg-to-nki
//   -nki-fuse-dma
//   -nki-fuse-store
//   -nki-fold-psum-init
// ), walks the single `func.func` inside, and prints the corresponding NKI
// Python kernel as a 1:1 shadow of `matmul_kernel_nki.py`.
//
// The emitter is intentionally specific to the (m, n, k)-loop matmul shape
// produced by the canonicalization pipeline above. It is NOT a general
// MLIR-to-Python translator: it recognizes the exact op skeleton
//
//   func.func @kernel(%A: memref<*>, %B: memref<*>, %C: memref<*>, ...) {
//     ... pid bound math ...
//     %psum = nki.psum_alloc : tensor<MxNxT>
//     ... pid bound math ...
//     scf.for %m = ... {                       // outer M loop
//       scf.for %n = ... {                     // inner N loop
//         ... offset math ...
//         scf.for %k = ... iter_args(%acc = %psum) {
//           %a = nki.dma_copy %A[...] : ... to tensor<MxKxT>
//           %b = nki.dma_copy %B[...] : ... to tensor<KxNxT>
//           %r = nki.nc_matmul %a, %b, %acc
//           scf.yield %r
//         }
//         nki.dma_store %r into %C[...]
//       }
//     }
//     return
//   }
//
// and emits the canonical NKI Python idiom for it. Different kernel shapes
// would need additional emitter logic.
//===----------------------------------------------------------------------===//
#include "IR/NKIDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::nki;

namespace {

enum class KernelKind { Matmul, Pointwise };

class PythonEmitter {
public:
  PythonEmitter(func::FuncOp func, llvm::raw_ostream &os)
      : func(func), os(os) {}

  LogicalResult emit() {
    if (failed(classify()))
      return failure();
    if (kind == KernelKind::Matmul) {
      if (failed(inferTileSizes()))
        return failure();
      emitHeader();
      return emitFunction();
    }
    // KernelKind::Pointwise
    if (failed(inferPointwise()))
      return failure();
    emitPointwiseHeader();
    return emitPointwiseFunction();
  }

private:
  func::FuncOp func;
  llvm::raw_ostream &os;
  int indent = 0;
  KernelKind kind = KernelKind::Matmul;
  int64_t tileM = 0, tileN = 0, tileK = 0;
  // Pointwise state.
  int64_t blockSize = 0;
  StringRef pointwiseOp; // "add", "sub", ...
  // PE-slot parallel unroll factors along the M (stationary-free) and N
  // (moving-free) output axes. When the stationary tile is smaller than
  // the 128×128 PE array, the free slots can each run an independent
  // nc_matmul, so we unroll the outer (m, n) loops to dispatch them.
  int64_t unrollM = 1, unrollN = 1;

  void writeIndent() {
    for (int i = 0; i < indent; ++i)
      os << "    ";
  }

  LogicalResult classify() {
    bool hasMatmul = false;
    bool hasPointwise = false;
    func.walk([&](Operation *op) {
      if (isa<NcMatmulOp>(op))
        hasMatmul = true;
      else if (isa<TensorTensorOp>(op))
        hasPointwise = true;
    });
    if (hasMatmul) {
      kind = KernelKind::Matmul;
      return success();
    }
    if (hasPointwise) {
      kind = KernelKind::Pointwise;
      return success();
    }
    func.emitError() << "no nki.nc_matmul or nki.tensor_tensor in function -- "
                        "nothing to emit";
    return failure();
  }

  LogicalResult inferPointwise() {
    TensorTensorOp tt;
    func.walk([&](TensorTensorOp t) {
      if (!tt)
        tt = t;
    });
    auto resTy = dyn_cast<RankedTensorType>(tt.getResult().getType());
    if (!resTy || resTy.getRank() != 1) {
      func.emitError() << "pointwise emitter only supports rank-1 tiles";
      return failure();
    }
    blockSize = resTy.getDimSize(0);
    if (blockSize <= 0) {
      func.emitError() << "pointwise tile has non-static shape";
      return failure();
    }
    pointwiseOp = tt.getOp();
    return success();
  }

  LogicalResult inferTileSizes() {
    NcMatmulOp mm;
    func.walk([&](NcMatmulOp m) {
      if (!mm)
        mm = m;
    });
    if (!mm) {
      func.emitError() << "no nki.nc_matmul in function -- nothing to emit";
      return failure();
    }
    auto lhsTy = dyn_cast<RankedTensorType>(mm.getLhs().getType());
    auto rhsTy = dyn_cast<RankedTensorType>(mm.getRhs().getType());
    if (!lhsTy || !rhsTy || lhsTy.getRank() != 2 || rhsTy.getRank() != 2) {
      func.emitError() << "nc_matmul operands are not rank-2 ranked tensors";
      return failure();
    }
    tileM = lhsTy.getDimSize(0);
    tileK = lhsTy.getDimSize(1);
    tileN = rhsTy.getDimSize(1);
    if (tileM <= 0 || tileN <= 0 || tileK <= 0) {
      func.emitError() << "nc_matmul has non-static tile dimensions";
      return failure();
    }
    // Snap each IR tile to a valid NC-v2 PE-array slot (64 or 128).
    // The IR tile is a lower bound on how much contiguous data we want to
    // push through the PE array per iteration; we always round up to the
    // nearest valid slot and mask/pad out-of-bounds elements at load/store.
    auto snap = [](int64_t dim) -> int64_t {
      return dim >= 128 ? 128 : 64;
    };
    tileM = snap(tileM);
    tileN = snap(tileN);
    tileK = snap(tileK);
    // PE-slot unroll. A 64-slot along the K (partition) axis leaves a
    // second K-slot free -- we claim it by unrolling the N output axis.
    // A 64-slot along the M (stationary-free) axis leaves a second M-slot
    // free -- we claim it by unrolling the M output axis. Both small =>
    // 4-way parallel matmuls per inner-k step.
    //
    // TEMPORARILY DISABLED while diagnosing a neuronx-cc SB_Allocator
    // assertion (`ml_base == 0`) that fires with the 4-way fan-out. Set
    // back to the dynamic rule below once the underlying layout issue
    // is understood.
    //   unrollM = (tileM < 128) ? 2 : 1;
    //   unrollN = (tileK < 128) ? 2 : 1;
    unrollM = 1;
    unrollN = 1;
    return success();
  }

  void emitHeader() {
    os << "\"\"\"\n";
    os << "Auto-generated NKI kernel.\n";
    os << "Generator : linalg-to-nki-translate\n";
    os << "Pipeline  : Triton -> TTIR -> triton-shared -> Linalg -> nki dialect"
          " -> Python\n";
    os << "\n";
    os << "Op-pattern translation table (source -> target):\n";
    os << "  func.func @kernel(...)               -> @nki.jit def "
       << func.getName() << "_nki(lhsT, rhs)\n";
    os << "  arith.constant " << tileM << "/" << tileN << "/" << tileK
       << "                 -> TILE_M/N/K constants (snapped to 64/128)\n";
    os << "  scf.for over pid_m / pid_n           -> ceil-div "
          "nl.affine_range(...)\n";
    os << "  nki.psum_alloc                       -> nl.ndarray(buffer=nl.psum)"
          "\n";
    os << "  scf.for over k iter_args(%psum)      -> ceil-div "
          "nl.affine_range(...)\n";
    os << "  nki.dma_copy (A tile)                -> nl.load(lhsT[...], "
          "mask=...)\n";
    os << "  nki.dma_copy (B tile)                -> nl.load(rhs[...], "
          "mask=...)\n";
    os << "  nki.nc_matmul                        -> nisa.nc_matmul(...)\n";
    os << "  nki.dma_store                        -> nisa.tensor_copy + "
          "nl.store(C[...], mask=...)\n";
    os << "\n";
    os << "Tile sizes:  TILE_M=" << tileM << "  TILE_N=" << tileN
       << "  TILE_K=" << tileK << "\n";
    os << "PE-slot unroll: M x N = " << unrollM << " x " << unrollN
       << " => " << (unrollM * unrollN)
       << " concurrent nc_matmul per inner k step\n";
    os << "\"\"\"\n";
    os << "import neuronxcc.nki as nki\n";
    os << "import neuronxcc.nki.isa as nisa\n";
    os << "import neuronxcc.nki.language as nl\n\n\n";
  }

  LogicalResult emitFunction() {
    os << "@nki.jit\n";
    os << "def " << func.getName() << "_nki(lhsT, rhs):\n";
    indent++;
    writeIndent();
    os << "\"\"\"lhsT is A transposed (shape K, M). Computes C = lhsT.T @ rhs."
          "\"\"\"\n";
    writeIndent();
    os << "TILE_M = " << tileM << "\n";
    writeIndent();
    os << "TILE_N = " << tileN << "\n";
    writeIndent();
    os << "TILE_K = " << tileK << "\n\n";

    writeIndent();
    os << "K, M = lhsT.shape\n";
    writeIndent();
    os << "K_, N = rhs.shape\n";
    writeIndent();
    os << "assert K == K_, "
          "f\"Contraction mismatch: lhsT.K={K} != rhs.K={K_}\"\n\n";

    writeIndent();
    os << "C = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)\n\n";

    // Find the outer scf.for in the function body.
    scf::ForOp outer;
    for (Operation &op : func.getBody().front().getOperations()) {
      if (auto f = dyn_cast<scf::ForOp>(&op)) {
        outer = f;
        break;
      }
    }
    if (!outer) {
      func.emitError() << "no outer scf.for found (run "
                          "-nki-canonicalize-pid-loops first)";
      return failure();
    }
    if (failed(emitMLoop(outer)))
      return failure();

    writeIndent();
    os << "return C\n";
    indent--;
    return success();
  }

  void emitPointwiseHeader() {
    os << "\"\"\"\n";
    os << "Auto-generated NKI kernel.\n";
    os << "Generator : linalg-to-nki-translate\n";
    os << "Pipeline  : Triton -> TTIR -> triton-shared -> Linalg -> nki dialect"
          " -> Python\n";
    os << "\n";
    os << "Op-pattern translation table (source -> target):\n";
    os << "  func.func @kernel(...)               -> @nki.jit def "
       << func.getName() << "_nki(x, y, out)\n";
    os << "  arith.constant " << blockSize
       << "                   -> BLOCK_SIZE constant\n";
    os << "  arith.muli %pid, %c" << blockSize
       << "            -> for block in nl.affine_range(...)\n";
    os << "  nki.dma_copy (x tile)                -> nl.load(x[...], "
          "mask=...)\n";
    os << "  nki.dma_copy (y tile)                -> nl.load(y[...], "
          "mask=...)\n";
    os << "  nki.tensor_tensor \"" << pointwiseOp
       << "\"             -> nl." << pointwiseOp << "(...)\n";
    os << "  nki.dma_store                        -> nl.store(out[...], "
          "mask=...)\n";
    os << "\n";
    os << "Block size: BLOCK_SIZE=" << blockSize << "\n";
    os << "Engine:     Vector Engine (nl." << pointwiseOp << ")\n";
    os << "\"\"\"\n";
    os << "import neuronxcc.nki as nki\n";
    os << "import neuronxcc.nki.isa as nisa\n";
    os << "import neuronxcc.nki.language as nl\n\n\n";
  }

  LogicalResult emitPointwiseFunction() {
    os << "@nki.jit\n";
    os << "def " << func.getName() << "_nki(x, y):\n";
    indent++;

    writeIndent();
    os << "\"\"\"Computes out = nl." << pointwiseOp
       << "(x, y) over BLOCK_SIZE-sized strips.\"\"\"\n";
    // Split BLOCK_SIZE into a 2D (par, free) SBUF tile so it satisfies NKI's
    // partition-dim-first layout requirement. par_dim = min(BLOCK_SIZE, 128).
    int64_t parDim = std::min<int64_t>(blockSize, 128);
    int64_t freeDim = blockSize / parDim;
    writeIndent();
    os << "BLOCK_SIZE = " << blockSize << "\n";
    writeIndent();
    os << "PAR = " << parDim << "\n";
    writeIndent();
    os << "FREE = " << freeDim << "\n\n";

    writeIndent();
    os << "n_elements = x.shape[0]\n";
    writeIndent();
    os << "out = nl.ndarray((n_elements,), dtype=x.dtype, "
          "buffer=nl.shared_hbm)\n\n";

    writeIndent();
    os << "for block in nl.affine_range((n_elements + BLOCK_SIZE - 1) "
          "// BLOCK_SIZE):\n";
    indent++;

    writeIndent();
    os << "i_p, i_f = nl.mgrid[0:PAR, 0:FREE]\n";
    writeIndent();
    os << "lin = i_p * FREE + i_f\n";
    writeIndent();
    os << "mask = block * BLOCK_SIZE + lin < n_elements\n\n";

    writeIndent();
    os << "x_tile = nl.zeros((PAR, FREE), dtype=x.dtype, buffer=nl.sbuf)\n";
    writeIndent();
    os << "x_tile[i_p, i_f] = nl.load(x[block * BLOCK_SIZE + lin], "
          "mask=mask)\n\n";

    writeIndent();
    os << "y_tile = nl.zeros((PAR, FREE), dtype=y.dtype, buffer=nl.sbuf)\n";
    writeIndent();
    os << "y_tile[i_p, i_f] = nl.load(y[block * BLOCK_SIZE + lin], "
          "mask=mask)\n\n";

    writeIndent();
    os << "z_tile = nl." << pointwiseOp << "(x_tile, y_tile)\n";
    writeIndent();
    os << "nl.store(out[block * BLOCK_SIZE + lin], value=z_tile, mask=mask)\n";

    indent--;
    os << "\n";
    writeIndent();
    os << "return out\n";
    indent--;
    return success();
  }

  // Emit an M-strip coefficient "(UM * m + mm)" or "m" if unrollM == 1.
  std::string mCoef(int64_t mm) {
    if (unrollM == 1)
      return "m";
    return "(" + std::to_string(unrollM) + " * m + " + std::to_string(mm) + ")";
  }
  std::string nCoef(int64_t nn) {
    if (unrollN == 1)
      return "n";
    return "(" + std::to_string(unrollN) + " * n + " + std::to_string(nn) + ")";
  }

  LogicalResult emitMLoop(scf::ForOp outer) {
    writeIndent();
    if (unrollM == 1) {
      os << "for m in nl.affine_range((M + TILE_M - 1) // TILE_M):\n";
    } else {
      os << "for m in nl.affine_range("
            "(M + " << unrollM << " * TILE_M - 1) // ("
         << unrollM << " * TILE_M)):\n";
    }
    indent++;

    scf::ForOp inner;
    for (Operation &op : outer.getBody()->getOperations()) {
      if (auto f = dyn_cast<scf::ForOp>(&op)) {
        inner = f;
        break;
      }
    }
    if (!inner) {
      func.emitError() << "no inner scf.for inside outer scf.for";
      return failure();
    }
    if (failed(emitNLoop(inner)))
      return failure();

    indent--;
    return success();
  }

  LogicalResult emitNLoop(scf::ForOp inner) {
    writeIndent();
    if (unrollN == 1) {
      os << "for n in nl.affine_range((N + TILE_N - 1) // TILE_N):\n";
    } else {
      os << "for n in nl.affine_range("
            "(N + " << unrollN << " * TILE_N - 1) // ("
         << unrollN << " * TILE_N)):\n";
    }
    indent++;

    // One PSUM accumulator per parallel PE slot.
    for (int64_t mm = 0; mm < unrollM; ++mm) {
      for (int64_t nn = 0; nn < unrollN; ++nn) {
        writeIndent();
        os << "res_psum_" << mm << nn
           << " = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, "
              "buffer=nl.psum)\n";
      }
    }
    os << "\n";

    scf::ForOp kLoop;
    DmaStoreOp store;
    for (Operation &op : inner.getBody()->getOperations()) {
      if (auto f = dyn_cast<scf::ForOp>(&op))
        kLoop = f;
      else if (auto s = dyn_cast<DmaStoreOp>(&op))
        store = s;
    }
    if (!kLoop) {
      func.emitError() << "no k-loop inside (m, n) loops";
      return failure();
    }
    if (failed(emitKLoop(kLoop)))
      return failure();
    if (store)
      emitStore(store);

    indent--;
    return success();
  }

  LogicalResult emitKLoop(scf::ForOp kLoop) {
    writeIndent();
    os << "for k in nl.affine_range((K + TILE_K - 1) // TILE_K):\n";
    indent++;

    DmaCopyOp aLoad, bLoad;
    NcMatmulOp mm;
    for (Operation &op : kLoop.getBody()->getOperations()) {
      if (auto d = dyn_cast<DmaCopyOp>(&op)) {
        if (!aLoad)
          aLoad = d;
        else if (!bLoad)
          bLoad = d;
      } else if (auto m = dyn_cast<NcMatmulOp>(&op)) {
        mm = m;
      }
    }
    if (!aLoad || !bLoad || !mm) {
      func.emitError() << "k-loop body missing expected dma_copy / nc_matmul";
      return failure();
    }

    // A tile loads. lhsT is already transposed (K, M). We zero-init an
    // SBUF tile then do a masked indexed assignment from an nl.load --
    // OOB lanes stay at 0, which is critical for the K-boundary tile
    // because nc_matmul would otherwise sum garbage into the PSUM and
    // corrupt every output element of this (m, n) tile.
    writeIndent();
    os << "i_k, i_m = nl.mgrid[0:TILE_K, 0:TILE_M]\n";
    for (int64_t mm = 0; mm < unrollM; ++mm) {
      std::string mc = mCoef(mm);
      writeIndent();
      os << "mask_lhsT_" << mm
         << " = (k * TILE_K + i_k < K) & (" << mc << " * TILE_M + i_m < M)\n";
      writeIndent();
      os << "lhsT_tile_" << mm
         << " = nl.zeros((TILE_K, TILE_M), dtype=lhsT.dtype, "
            "buffer=nl.sbuf)\n";
      writeIndent();
      os << "lhsT_tile_" << mm << "[i_k, i_m] = nl.load(\n";
      indent++;
      writeIndent();
      os << "lhsT[k * TILE_K + i_k, " << mc << " * TILE_M + i_m],\n";
      writeIndent();
      os << "mask=mask_lhsT_" << mm << ",\n";
      indent--;
      writeIndent();
      os << ")\n";
    }
    os << "\n";

    // B tile loads. rhs is (K, N); same zero-init + mask pattern.
    writeIndent();
    os << "i_k, i_n = nl.mgrid[0:TILE_K, 0:TILE_N]\n";
    for (int64_t nn = 0; nn < unrollN; ++nn) {
      std::string nc = nCoef(nn);
      writeIndent();
      os << "mask_rhs_" << nn
         << " = (k * TILE_K + i_k < K) & (" << nc << " * TILE_N + i_n < N)\n";
      writeIndent();
      os << "rhs_tile_" << nn
         << " = nl.zeros((TILE_K, TILE_N), dtype=rhs.dtype, "
            "buffer=nl.sbuf)\n";
      writeIndent();
      os << "rhs_tile_" << nn << "[i_k, i_n] = nl.load(\n";
      indent++;
      writeIndent();
      os << "rhs[k * TILE_K + i_k, " << nc << " * TILE_N + i_n],\n";
      writeIndent();
      os << "mask=mask_rhs_" << nn << ",\n";
      indent--;
      writeIndent();
      os << ")\n";
    }
    os << "\n";

    // nc_matmul dispatch. stationary=lhsT_tile_mm because lhsT^T == A.
    // Each (mm, nn) output strip gets its own PE slot at tile position
    // (mm*TILE_K, nn*TILE_M). When the stationary tile already fills
    // the full 128×128 PE array (both unrolls == 1), neuronx-cc doesn't
    // want tile_size/tile_position at all.
    bool useTiling = (tileK < 128) || (tileM < 128);
    for (int64_t mm = 0; mm < unrollM; ++mm) {
      for (int64_t nn = 0; nn < unrollN; ++nn) {
        writeIndent();
        os << "res_psum_" << mm << nn << "[...] += nisa.nc_matmul(\n";
        indent++;
        writeIndent();
        os << "stationary=lhsT_tile_" << mm << ",\n";
        writeIndent();
        os << "moving=rhs_tile_" << nn << ",\n";
        if (useTiling) {
          writeIndent();
          os << "tile_position=(" << (mm * tileK) << ", " << (nn * tileM)
             << "),\n";
          writeIndent();
          os << "tile_size=(" << tileK << ", " << tileM << "),\n";
        }
        indent--;
        writeIndent();
        os << ")\n";
      }
    }

    indent--;
    return success();
  }

  void emitStore(DmaStoreOp store) {
    os << "\n";
    // PSUM -> SBUF drain. If the store carries an `activation` attr, fold the
    // epilogue activation into the drain via `nisa.activation`; otherwise fall
    // back to a plain tensor_copy.
    StringRef activation = store.getActivation();
    bool hasActivation = activation != "none";
    writeIndent();
    os << "i_m, i_n = nl.mgrid[0:TILE_M, 0:TILE_N]\n";
    for (int64_t mm = 0; mm < unrollM; ++mm) {
      for (int64_t nn = 0; nn < unrollN; ++nn) {
        std::string mc = mCoef(mm);
        std::string nc = nCoef(nn);
        writeIndent();
        if (hasActivation) {
          os << "res_sbuf_" << mm << nn
             << " = nisa.activation(op=nl." << activation
             << ", data=res_psum_" << mm << nn
             << ", dtype=lhsT.dtype)\n";
        } else {
          os << "res_sbuf_" << mm << nn
             << " = nisa.tensor_copy(res_psum_" << mm << nn
             << ", dtype=lhsT.dtype)\n";
        }
        writeIndent();
        os << "nl.store(\n";
        indent++;
        writeIndent();
        os << "C[" << mc << " * TILE_M + i_m, " << nc << " * TILE_N + i_n],\n";
        writeIndent();
        os << "value=res_sbuf_" << mm << nn << ",\n";
        writeIndent();
        os << "mask=(" << mc << " * TILE_M + i_m < M) & (" << nc
           << " * TILE_N + i_n < N),\n";
        indent--;
        writeIndent();
        os << ")\n";
      }
    }
  }
};

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                       llvm::cl::desc("<input mlir file>"),
                                       llvm::cl::init("-"));
  llvm::cl::opt<std::string> outputFile("o", llvm::cl::desc("Output file"),
                                        llvm::cl::init("-"));
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Translate fully-lowered nki MLIR to NKI Python\n");

  // Set up an MLIR context with just the dialects this tool actually parses.
  // We don't need every upstream dialect because the canonicalization
  // pipeline produces a narrow op set.
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, bufferization::BufferizationDialect,
                  func::FuncDialect, linalg::LinalgDialect,
                  memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect,
                  nki::NKIDialect>();

  std::string errorMsg;
  auto file = openInputFile(inputFile, &errorMsg);
  if (!file) {
    llvm::errs() << errorMsg << "\n";
    return 1;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &ctx);
  if (!module) {
    llvm::errs() << "failed to parse module\n";
    return 1;
  }

  func::FuncOp func;
  module->walk([&](func::FuncOp f) {
    if (!func)
      func = f;
  });
  if (!func) {
    llvm::errs() << "no func.func in module\n";
    return 1;
  }

  auto outFile = openOutputFile(outputFile, &errorMsg);
  if (!outFile) {
    llvm::errs() << errorMsg << "\n";
    return 1;
  }
  PythonEmitter emitter(func, outFile->os());
  if (failed(emitter.emit())) {
    llvm::errs() << "Python emission failed\n";
    return 1;
  }
  outFile->keep();
  return 0;
}
