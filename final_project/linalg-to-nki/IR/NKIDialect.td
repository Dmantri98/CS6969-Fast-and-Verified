//===----------------------------------------------------------------------===//
// NKI Dialect
//
// Models a small slice of the NKI tensor-engine ISA so that linalg-level IR
// can be lowered to a representation that maps 1:1 onto NKI Python `nisa.*`
// calls (see ../matmul_kernel_nki.py for the eventual emission target).
//===----------------------------------------------------------------------===//
#ifndef NKI_DIALECT
#define NKI_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//
def NKI_Dialect : Dialect {
  let name = "nki";
  let cppNamespace = "::mlir::nki";
  let summary = "NKI tensor-engine ISA dialect";
  let description = [{
    The `nki` dialect is a low-level target dialect that models the tensor /
    scalar engine operations exposed by the Neuron Kernel Interface (NKI).

    It is designed to be the lowering target of an MLIR pipeline that starts
    from `linalg` IR (e.g. produced by triton-shared) and to subsequently be
    translated into NKI Python ISA calls of the form `nisa.<op>(...)`.
  }];
}

//===----------------------------------------------------------------------===//
// Op base class
//===----------------------------------------------------------------------===//
class NKI_Op<string mnemonic, list<Trait> traits = []> :
    Op<NKI_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// nki.nc_matmul
//===----------------------------------------------------------------------===//
def NKI_NcMatmulOp : NKI_Op<"nc_matmul", [
    Pure,
    AllElementTypesMatch<["lhs", "rhs", "acc", "result"]>,
    AllShapesMatch<["acc", "result"]>
  ]> {
  let summary = "Tensor-engine matmul with PSUM accumulation";
  let description = [{
    Computes `result = acc + (lhs @ rhs)` on a single Neuron Core tensor
    engine tile. This op fuses the destination-passing-style
    `linalg.matmul + linalg.add` accumulator pattern emitted by triton-shared
    into the single physical operation expressed by NKI's
    `nisa.nc_matmul(dst=..., stationary=..., moving=...)`.

    Operands:
      * `lhs` : `tensor<MxKxT>` -- left-hand-side tile. At a later lowering
                                   step it becomes the (transposed)
                                   `stationary` operand of `nisa.nc_matmul`.
      * `rhs` : `tensor<KxNxT>` -- right-hand-side tile, the `moving` operand.
      * `acc` : `tensor<MxNxT>` -- prior PSUM accumulator value.

    Result:
      * `result` : `tensor<MxNxT>` -- updated PSUM accumulator value.
  }];

  let arguments = (ins AnyRankedTensor:$lhs,
                       AnyRankedTensor:$rhs,
                       AnyRankedTensor:$acc);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict
    `:` `(` type($lhs) `,` type($rhs) `,` type($acc) `)`
    `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// nki.psum_alloc
//===----------------------------------------------------------------------===//
def NKI_PsumAllocOp : NKI_Op<"psum_alloc", [Pure]> {
  let summary = "Allocate a zero-initialized PSUM accumulator tile";
  let description = [{
    Models `nl.ndarray((M, N), dtype=..., buffer=nl.psum)`. Produces a
    statically-shaped, zero-initialized tile that is meant to be threaded
    through `scf.for` `iter_args` and consumed as the `acc` operand of
    `nki.nc_matmul`.

    This op is the target of the `fold-psum-init` pass, which recognizes the
    `tensor.empty` + `linalg.fill 0.0` pair emitted by triton-shared as a
    PSUM accumulator initializer and rewrites it to a single `nki.psum_alloc`.
    The Python emitter recognizes this op (and the `iter_args` value chain it
    feeds) and prints `res_psum = nl.ndarray(..., buffer=nl.psum)` *outside*
    the surrounding loop.
  }];

  let arguments = (ins);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// nki.dma_copy
//===----------------------------------------------------------------------===//
def NKI_DmaCopyOp : NKI_Op<"dma_copy", [Pure]> {
  let summary = "DMA copy from a strided unranked source into a tile tensor";
  let description = [{
    Models `nisa.dma_copy(dst=tile, src=source[...])`. The source is described
    by an unranked memref base plus a single linearized `offset` (in elements)
    and one `stride` per result-tensor dimension -- i.e. the same triple that
    `memref.reinterpret_cast` carries. The result is a freshly-allocated tile
    tensor whose shape determines the copy extent.

    This op is the target of the `nki-fuse-dma` pass, which collapses the
    `memref.reinterpret_cast` + `memref.subview` + `memref.copy` +
    `bufferization.to_tensor` chain emitted by triton-shared into a single
    semantic DMA load.
  }];

  let arguments = (ins AnyUnrankedMemRef:$source,
                       Index:$offset,
                       Variadic<Index>:$strides);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `[` $offset `]` `strides` `[` $strides `]`
    attr-dict `:` type($source) `to` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// nki.dma_store
//===----------------------------------------------------------------------===//
def NKI_DmaStoreOp : NKI_Op<"dma_store"> {
  let summary = "DMA copy a tile tensor into a strided unranked destination";
  let description = [{
    Models `nisa.dma_copy(dst=destination[...], src=tile)` -- the store-side
    counterpart of `nki.dma_copy`. The destination is described by an unranked
    memref base plus a single linearized `offset` (in elements) and one
    `stride` per source-tensor dimension; this is exactly the triple that
    `memref.reinterpret_cast` carries.

    This op is the target of the `nki-fuse-store` pass, which collapses the
    `tensor.extract_slice` + `memref.reinterpret_cast` + `memref.subview` +
    `bufferization.materialize_in_destination` chain emitted by triton-shared
    into a single semantic DMA store. Like the load side, the dynamic mask
    extents are intentionally dropped: the source tile shape is the
    statically known tensor extent, and the upstream Triton kernel's
    bounds-handling is the source of truth.

    The op has memory side effects (writes through `destination`) and is
    intentionally not `Pure` -- the greedy driver must not DCE it.
  }];

  let arguments = (ins AnyRankedTensor:$tile,
                       AnyUnrankedMemRef:$destination,
                       Index:$offset,
                       Variadic<Index>:$strides,
                       DefaultValuedStrAttr<StrAttr, "none">:$activation);

  let assemblyFormat = [{
    $tile `into` $destination `[` $offset `]` `strides` `[` $strides `]`
    attr-dict `:` type($tile) `into` type($destination)
  }];
}

//===----------------------------------------------------------------------===//
// nki.tensor_tensor
//===----------------------------------------------------------------------===//
def NKI_TensorTensorOp : NKI_Op<"tensor_tensor", [
    Pure,
    SameOperandsAndResultType
  ]> {
  let summary = "Elementwise tensor-tensor binary op";
  let description = [{
    Performs an elementwise binary op on two equally shaped tensors. Models
    NKI's `nisa.tensor_tensor` family. The specific binary op is selected by
    the string-valued `op` attribute (e.g. `"add"`, `"sub"`, `"mul"`, ...).

    No conversion currently produces this op; it is defined here so that
    subsequent passes (e.g. lowering of `linalg.add` outside of an
    accumulator pattern) can target it.
  }];

  let arguments = (ins AnyRankedTensor:$lhs,
                       AnyRankedTensor:$rhs,
                       StrAttr:$op);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $op $lhs `,` $rhs attr-dict `:` type($result)
  }];
}

#endif // NKI_DIALECT
