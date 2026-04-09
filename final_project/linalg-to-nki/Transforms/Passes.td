//===----------------------------------------------------------------------===//
// linalg-to-nki conversion passes.
//===----------------------------------------------------------------------===//
#ifndef LINALG_TO_NKI_PASSES
#define LINALG_TO_NKI_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToNKI : Pass<"linalg-to-nki", "::mlir::func::FuncOp"> {
  let summary = "Lower selected linalg patterns into the nki dialect";
  let description = [{
    Currently fuses the destination-passing-style
    `linalg.matmul + linalg.add` accumulator pattern emitted by triton-shared
    into a single `nki.nc_matmul` operation.

    Match (within a single `scf.for` body):
      %tmp_init = linalg.fill ins(%cst_zero) outs(tensor.empty()) -> M x N
      %tmp      = linalg.matmul ins(%A, %B) outs(%tmp_init)       -> M x N
      %new_acc  = linalg.add ins(%acc, %tmp) outs(%acc)           -> M x N

    Rewrite to:
      %new_acc = nki.nc_matmul %A, %B, %acc : (...) -> M x N
  }];
  let dependentDialects = [
    "::mlir::nki::NKIDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def NKIFuseDma : Pass<"nki-fuse-dma", "::mlir::func::FuncOp"> {
  let summary = "Collapse triton-shared DMA-load chains into nki.dma_copy";
  let description = [{
    Matches the pattern emitted by triton-shared for masked global-memory tile
    loads:

      %rc    = memref.reinterpret_cast %src to
                  offset: [%off], sizes: [M, N], strides: [%s0, %s1]
      %buf   = memref.alloc() : memref<MxNxT>
      %ssrc  = memref.subview %rc [0,0]  [%vrows, N] [1,1]
      %sdst  = memref.subview %buf[0,0]  [%vrows, N] [1,1]
      memref.copy %ssrc, %sdst
      %tile  = bufferization.to_tensor %buf restrict writable
                  : memref<MxNxT> to tensor<MxNxT>

    and rewrites to a single semantic DMA load:

      %tile = nki.dma_copy %src [%off] strides [%s0, %s1]
                  : memref<*xT> to tensor<MxNxT>

    The mask-clamping subview (`%vrows`) is intentionally dropped: the
    destination tile shape is the static `tensor<MxNxT>` extent, and any
    out-of-bounds rows are simply garbage in this lowering -- the upstream
    Triton kernel's bounds-handling is the source of truth that callers know
    not to touch.

    Side-effecting memref ops (`memref.copy`, `memref.alloc`,
    `memref.reinterpret_cast`, `memref.subview`) are explicitly erased after
    the rewrite -- the greedy driver will not DCE them on its own.
  }];
  let dependentDialects = [
    "::mlir::nki::NKIDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::bufferization::BufferizationDialect"
  ];
}

def NKIFuseStore : Pass<"nki-fuse-store", "::mlir::func::FuncOp"> {
  let summary = "Collapse triton-shared DMA-store chains into nki.dma_store";
  let description = [{
    Matches the pattern emitted by triton-shared for masked global-memory tile
    stores:

      %rc        = memref.reinterpret_cast %dst to
                      offset: [%off], sizes: [M, N], strides: [%s0, %s1]
      %slice     = tensor.extract_slice %tile[0, 0] [%vrows, %vcols] [1, 1]
      %sdst      = memref.subview %rc[0, 0] [%vrows, %vcols] [1, 1]
      bufferization.materialize_in_destination %slice in writable %sdst

    and rewrites to a single semantic DMA store:

      nki.dma_store %tile into %dst [%off] strides [%s0, %s1]
                  : tensor<MxNxT> into memref<*xT>

    The dynamic `extract_slice` mask extents are intentionally dropped: the
    source tile shape is the static `tensor<MxNxT>` extent, and any
    out-of-bounds rows are simply garbage in this lowering -- the upstream
    Triton kernel's bounds-handling is the source of truth.

    Side-effecting memref ops (`memref.subview`, `memref.reinterpret_cast`)
    and the dead `tensor.extract_slice` are explicitly erased after the
    rewrite -- the greedy driver will not DCE them on its own.
  }];
  let dependentDialects = [
    "::mlir::nki::NKIDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::bufferization::BufferizationDialect"
  ];
}

def NKIFoldPsumInit : Pass<"nki-fold-psum-init", "::mlir::func::FuncOp"> {
  let summary = "Fold linalg.fill 0.0 PSUM seeds into nki.psum_alloc";
  let description = [{
    Matches the destination-passing PSUM accumulator initializer pattern
    emitted by triton-shared once the matmul body has been lowered:

      %0 = tensor.empty() : tensor<MxNxT>
      %1 = linalg.fill ins(%cst : T) outs(%0)        // %cst is +0.0
      %r = scf.for ... iter_args(%acc = %1) -> (tensor<MxNxT>) {
        ...
        %m = nki.nc_matmul %A, %B, %acc : ...
        scf.yield %m : tensor<MxNxT>
      }

    and rewrites to:

      %psum = nki.psum_alloc : tensor<MxNxT>
      %r = scf.for ... iter_args(%acc = %psum) -> (tensor<MxNxT>) {
        ...
        %m = nki.nc_matmul %A, %B, %acc : ...
        scf.yield %m : tensor<MxNxT>
      }

    The pass is intentionally narrow: it only fires when the fill's only use
    is an `scf.for` `iter_args` initializer AND the corresponding loop arg
    flows into an `nki.nc_matmul` `acc` operand. This is precisely the PSUM
    accumulator shape and nothing else.

    The fill is replaced; the now-dead `tensor.empty` is erased.
  }];
  let dependentDialects = [
    "::mlir::nki::NKIDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def NKICanonicalizePidLoops : Pass<"nki-canonicalize-pid-loops",
                                   "::mlir::func::FuncOp"> {
  let summary = "Recover m/n tile loops from triton's program_id math";
  let description = [{
    Triton lowers `tl.program_id(0)` plus the standard

      pid_m = pid // num_pid_n
      pid_n = pid % num_pid_n

    re-indexing into a flat `arith.divsi`/`arith.remsi` pair on a function
    argument. After triton-shared's lowering it shows up as:

      %2 = arith.addi %N, %c63       : i32          // (N + BLOCK_N - 1)
      %3 = arith.divsi %2, %cBLOCK_N : i32          // num_pid_n
      %4 = arith.divsi %pid, %3      : i32          // pid_m
      %5 = arith.remsi %pid, %3      : i32          // pid_n
      %6 = arith.muli  %4, %cBLOCK_M : i32          // m_off (BLOCK_M not %3!)
      %7 = arith.muli  %5, %cBLOCK_N : i32          // n_off

    This pass recognizes that pattern, wraps the entire function body in
    two `scf.for` loops over `(pid_m, pid_n)`, and rewrites the divsi/remsi
    results to the new induction variables. The flat `program_id` function
    argument is removed.

    Reusing the bounds: the n loop bound is the existing `num_pid_n` SSA
    value (`%3`); the m loop bound is constructed analogously from the M
    function argument and the recovered `BLOCK_M`. The M arg is identified
    as the i32 arg used in the load-side bounds-check `minsi` chain inside
    the k-loop, so this pass MUST run BEFORE `nki-fuse-dma` (which deletes
    that arith).
  }];
  let dependentDialects = [
    "::mlir::nki::NKIDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect"
  ];
}

#endif // LINALG_TO_NKI_PASSES
