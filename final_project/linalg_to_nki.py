#!/usr/bin/env python3
"""
linalg_to_nki.py — MLIR ConversionPass: Linalg Dialect → NKI Python Kernel

Reads a .linalg/.mlir MLIR file produced by triton-shared and emits a @nki.jit
Python kernel targeting AWS Trainium / Inferentia.

Pipeline context:
    Triton (.py)
      → triton-shared (ttir)
        → triton-shared lowering
          → Linalg dialect (.linalg / .mlir)   ← THIS FILE READS THIS
            → NKI kernel (.py)                 ← THIS FILE WRITES THIS

Supported source patterns (each handled by one ConversionPattern):
  ┌──────────────────────────────────────────────────────────────────────┐
  │ MatmulConversionPattern                                              │
  │   func containing linalg.matmul + scf.for K-loop                    │
  │   → @nki.jit kernel using nisa.nc_matmul + nl.psum accumulator       │
  ├──────────────────────────────────────────────────────────────────────┤
  │ VecAddConversionPattern                                              │
  │   func containing linalg.add on 1-D tensor<Nxf32>                   │
  │   → @nki.jit kernel using nl.add (Vector Engine)                     │
  └──────────────────────────────────────────────────────────────────────┘

Architecture (mirrors MLIR ConversionPass infrastructure):

  ConversionTarget
    Declares which ops are "illegal" (must be rewritten by a pattern) and
    which are "legal" (already in the target NKI dialect).  Equivalent to
    mlir::ConversionTarget.

  CodeEmitter
    Shared indented-line buffer used by all patterns to emit Python source.
    Equivalent to the mlir::PatternRewriter role in code generation.

  RewritePattern  (abstract)
    Base class for a single source-op → target-op translation.
    match(kernel) → bool      — equivalent to PatternRewriter::matchOp
    rewrite(kernel, emitter)  — equivalent to PatternRewriter::rewrite

  MatmulConversionPattern(RewritePattern)
    Handles linalg.matmul kernels.  Internal _emit_* methods each translate
    exactly one recognisable group of source ops:

      Source op group                      _emit_* method
      ─────────────────────────────────    ───────────────────────────
      func.func @name(...)                 _emit_func_signature
      arith.constant tile dims             _emit_tile_constants
      reinterpret_cast %arg2 (output ptr)  _emit_output_alloc
      arith.divsi/remsi %pid → pid_m,n     _emit_outer_loops  (loop synthesis)
      tensor.empty + linalg.fill (zero)    _emit_psum_init
      scf.for (K-reduction loop)           _emit_k_loop
      reinterpret_cast+copy (A tile)       _emit_a_tile_load
      reinterpret_cast+copy (B tile)       _emit_b_tile_load
      linalg.matmul + linalg.add           _emit_nc_matmul
      materialize_in_destination           _emit_matmul_store

  VecAddConversionPattern(RewritePattern)
    Handles linalg.add kernels.  Same discipline:

      Source op group                      _emit_* method
      ─────────────────────────────────    ───────────────────────────
      func.func @name(...)                 _emit_func_signature
      arith.constant 1024 (BLOCK_SIZE)     _emit_tile_constant
      arith.muli %pid, %c1024 → offset     _emit_outer_loop  (loop synthesis)
      reinterpret_cast+copy (x tile)       _emit_x_tile_load
      reinterpret_cast+copy (y tile)       _emit_y_tile_load
      linalg.add ins(%x, %y)               _emit_vec_add
      materialize_in_destination           _emit_vec_add_store

  LinalgToNKIConversionPass
    Registers patterns against a ConversionTarget, parses the IR to extract
    a ParsedKernel, selects the matching pattern, and runs it.

Usage:
    python linalg_to_nki.py matmul_kernel.linalg       [matmul_kernel_nki.py]
    python linalg_to_nki.py add_kernel_specialized.mlir [add_kernel_nki.py]
"""

from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARSED KERNEL TYPES
#     Data extracted from the source IR, kind-tagged so patterns can match.
# ─────────────────────────────────────────────────────────────────────────────

class KernelKind(Enum):
    MATMUL  = auto()   # func contains linalg.matmul + scf.for K-loop
    VEC_ADD = auto()   # func contains linalg.add on 1-D tensors, no K-loop


@dataclass
class MatmulInfo:
    """
    Structural parameters extracted from a linalg matmul kernel.

    Populated by the IR parser from:
      linalg.matmul  ins(A : tensor<BM×BK×f32>, B : tensor<BK×BN×f32>)
      scf.for        (K-reduction loop presence)
      memref.subview (bounds-check masking presence)
    """
    kind:             KernelKind = field(default=KernelKind.MATMUL, init=False)
    func_name:        str  = "matmul_kernel"
    block_m:          int  = 64    # BLOCK_SIZE_M  (rows of A tile)
    block_n:          int  = 64    # BLOCK_SIZE_N  (cols of B tile)
    block_k:          int  = 32    # BLOCK_SIZE_K  (cols of A = rows of B)
    has_k_loop:       bool = True
    has_bounds_check: bool = True  # memref.subview masking pattern


@dataclass
class VecAddInfo:
    """
    Structural parameters extracted from a linalg vector-add kernel.

    Populated by the IR parser from:
      arith.constant 1024 : index  (BLOCK_SIZE)
      linalg.add ins(x : tensor<1024xf32>, y : tensor<1024xf32>)
      memref.subview               (bounds-check masking presence)
    """
    kind:             KernelKind = field(default=KernelKind.VEC_ADD, init=False)
    func_name:        str  = "add_kernel"
    block_size:       int  = 1024  # BLOCK_SIZE (elements per tile)
    has_bounds_check: bool = True  # memref.subview masking pattern


# Union type for any kernel that this pass can handle.
ParsedKernel = Union[MatmulInfo, VecAddInfo]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  IR PARSER
#     Extracts a ParsedKernel from the linalg text IR.
#     Preferred path: mlir.ir Python bindings (type-system exact).
#     Fallback path:  regex (works without the bindings package installed).
# ─────────────────────────────────────────────────────────────────────────────

def _walk_ops(op):
    """
    Recursively yield every operation nested inside *op* (DFS, pre-order).

    Works with any mlir.ir.Operation, including the module op itself.
    Yields child ops before descending, so the first match for any op name
    is the outermost (earliest in textual order) occurrence.
    """
    for region in op.regions:
        for block in region.blocks:
            for child_op in block.operations:
                yield child_op
                yield from _walk_ops(child_op)


def _parse_via_bindings(ir_text: str) -> ParsedKernel:
    """
    Parse using the mlir.ir Python bindings (preferred path).

    Detection strategy:
      1. Walk the func.func body looking for:
           linalg.matmul  → MatmulInfo (kind=MATMUL)
           linalg.add     → VecAddInfo (kind=VEC_ADD) if operands are 1-D
      2. Extract tile sizes from operand RankedTensorType shapes — exact,
         no heuristic needed.
      3. Set structural flags (has_k_loop, has_bounds_check) from op presence.
    """
    from mlir.ir import Context, Module, StringAttr, RankedTensorType  # type: ignore[import]

    with Context() as ctx:
        # Importing each dialect module registers it in the active Context so
        # Module.parse() can parse their ops by name.  The names are not used
        # directly here; the import is the registration side-effect.
        try:
            import mlir.dialects.arith          # type: ignore[import]  # noqa: F401
            import mlir.dialects.func           # type: ignore[import]  # noqa: F401
            import mlir.dialects.linalg         # type: ignore[import]  # noqa: F401
            import mlir.dialects.scf            # type: ignore[import]  # noqa: F401
            import mlir.dialects.tensor         # type: ignore[import]  # noqa: F401
            import mlir.dialects.memref         # type: ignore[import]  # noqa: F401
            import mlir.dialects.bufferization  # type: ignore[import]  # noqa: F401
        except ImportError:
            pass
        ctx.allow_unregistered_dialects = True

        module = Module.parse(ir_text)

        for top_op in module.body.operations:
            if top_op.name != "func.func":
                continue

            func_name = StringAttr(top_op.attributes["sym_name"]).value

            has_matmul      = False
            has_vec_add_1d  = False
            has_k_loop      = False
            has_bounds_check = False
            matmul_info     = MatmulInfo(func_name=func_name)
            vec_add_info    = VecAddInfo(func_name=func_name)

            for op in _walk_ops(top_op):

                if op.name == "linalg.matmul":
                    # ins(%A : tensor<BM×BK×f32>, %B : tensor<BK×BN×f32>)
                    # operands order: [A, B, C_out]
                    a_type = RankedTensorType(op.operands[0].type)
                    b_type = RankedTensorType(op.operands[1].type)
                    matmul_info.block_m = a_type.shape[0]
                    matmul_info.block_k = a_type.shape[1]
                    matmul_info.block_n = b_type.shape[1]
                    has_matmul = True

                elif op.name == "linalg.add":
                    # Check if operands are 1-D (vec add) vs 2-D (matmul add)
                    try:
                        operand_type = RankedTensorType(op.operands[0].type)
                        if operand_type.rank == 1:
                            vec_add_info.block_size = operand_type.shape[0]
                            has_vec_add_1d = True
                    except Exception:
                        pass

                elif op.name == "scf.for":
                    has_k_loop = True

                elif op.name == "memref.subview":
                    has_bounds_check = True

            break  # only the first func.func matters

        # Decide kernel kind: matmul takes priority if both are present
        if has_matmul:
            matmul_info.has_k_loop       = has_k_loop
            matmul_info.has_bounds_check = has_bounds_check
            return matmul_info
        else:
            vec_add_info.has_bounds_check = has_bounds_check
            return vec_add_info


def _parse_via_regex(ir_text: str) -> ParsedKernel:
    """
    Regex-based fallback parser used when mlir.ir is not installed.

    Detection:
      - linalg.matmul present → MatmulInfo
      - linalg.add on 1-D tensor present (no matmul) → VecAddInfo
    """
    # ── Function name ────────────────────────────────────────────────────────
    m = re.search(r'func\.func\s+@(\w+)\s*\(', ir_text)
    func_name = m.group(1) if m else "kernel"

    has_k_loop       = bool(re.search(r'\bscf\.for\b', ir_text))
    has_bounds_check = bool(re.search(r'\bmemref\.subview\b', ir_text))

    # ── Detect matmul ─────────────────────────────────────────────────────────
    matmul_m = re.search(
        r'linalg\.matmul\s+ins\([^:]+:\s*tensor<(\d+)x(\d+)xf32>,\s*tensor<(\d+)x(\d+)xf32>',
        ir_text,
    )
    if matmul_m:
        info = MatmulInfo(func_name=func_name)
        info.block_m          = int(matmul_m.group(1))
        info.block_k          = int(matmul_m.group(2))
        info.block_n          = int(matmul_m.group(4))
        info.has_k_loop       = has_k_loop
        info.has_bounds_check = has_bounds_check
        return info

    # ── Detect 1-D vector add ─────────────────────────────────────────────────
    vec_add_m = re.search(
        r'linalg\.add\s+ins\([^:]+:\s*tensor<(\d+)xf32>,\s*tensor<(\d+)xf32>',
        ir_text,
    )
    if vec_add_m:
        info = VecAddInfo(func_name=func_name)
        info.block_size       = int(vec_add_m.group(1))
        info.has_bounds_check = has_bounds_check
        return info

    # ── Fallback: assume matmul from shape hints ──────────────────────────────
    info = MatmulInfo(func_name=func_name)
    acc = re.search(r'tensor\.empty\(\)\s*:\s*tensor<(\d+)x(\d+)xf32>', ir_text)
    if acc:
        info.block_m = int(acc.group(1))
        info.block_n = int(acc.group(2))
    alloc_shapes = re.findall(r'memref\.alloc\(\)\s*:\s*memref<(\d+)x(\d+)xf32>', ir_text)
    bk_candidates: set[int] = set()
    for a, b in alloc_shapes:
        for dim in (int(a), int(b)):
            if dim not in (info.block_m, info.block_n):
                bk_candidates.add(dim)
    if bk_candidates:
        info.block_k = min(bk_candidates)
    info.has_k_loop       = has_k_loop
    info.has_bounds_check = has_bounds_check
    return info


def parse_linalg_ir(ir_text: str) -> ParsedKernel:
    """
    Parse a linalg MLIR module and return its structural parameters.

    Tries the mlir.ir Python bindings first (precise, type-system-aware).
    Falls back to the regex parser if the bindings package is not installed.
    """
    try:
        return _parse_via_bindings(ir_text)
    except ImportError:
        import warnings as _warnings
        _warnings.warn(
            "mlir Python bindings not found; falling back to regex parser.",
            stacklevel=2,
        )
        return _parse_via_regex(ir_text)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NKI HARDWARE CONSTRAINT VALIDATOR + TILE-SIZE SELECTOR  (matmul only)
# ─────────────────────────────────────────────────────────────────────────────

_NKI_PMAX           = 128   # max partition dimension (TILE_K)
_NKI_GEMM_STAT_FMAX = 128   # max stationary free dimension (TILE_M)
_NKI_GEMM_MOV_FMAX  = 512   # max moving free dimension (TILE_N)

# Valid PE-grid tile sizes for nc_matmul's tile_size=(row, col) parameter.
#
# The Tensor Engine is a 128×128 systolic array.  tile_size slices that grid
# so that smaller stationary tiles can run in parallel:
#
#   row_size options  : 128 (full), 64 (2× row tiles, NCV2+), 32 (4× row tiles, NCV3+)
#   col_size options  : 128 (full), 64 (2× col tiles, NCV2+), 32 (4× col tiles, NCV3+)
#
# We choose the *smallest* valid size >= the kernel's tile dimension so that
# the hardware can schedule the maximum number of parallel tiles.
_VALID_PE_TILE_SIZES = (32, 64, 128)
_PE_GRID_ROWS = 128
_PE_GRID_COLS = 128


def _fit_pe_tile_size(tile_dim: int, pe_axis_size: int, comp_max: int,
                      axis: str) -> tuple[int, list[str]]:
    """
    Return (pe_tile_size, warnings) for one axis of the nc_matmul tile_size.

    Rules:
      1. If tile_dim > comp_max → hard error; the computation itself exceeds hw.
      2. tile_size is a partition of the pe_axis_size (128) grid.
         We select the smallest value in _VALID_PE_TILE_SIZES that is
         >= min(tile_dim, pe_axis_size).  This gives the tightest PE slice
         that still fits the tile, maximising the number of parallel slots.
    """
    warnings: list[str] = []

    if tile_dim > comp_max:
        warnings.append(
            f"TILE_{axis}={tile_dim} exceeds the NKI {axis}-axis computation "
            f"maximum ({comp_max}).  The generated kernel will not be valid."
        )
        return pe_axis_size, warnings

    target = min(tile_dim, pe_axis_size)
    for candidate in _VALID_PE_TILE_SIZES:
        if candidate >= target:
            return candidate, warnings

    return pe_axis_size, warnings  # unreachable: max(_VALID_PE_TILE_SIZES) == 128


def select_tile_size(info: MatmulInfo) -> tuple[tuple[int, int], list[str]]:
    """
    Compute the nc_matmul tile_size=(row_size, col_size) for this matmul kernel.

    row_size  (PE rows)  constrains TILE_M — stationary free dim
    col_size  (PE cols)  constrains TILE_N — moving free dim
    """
    row_size, row_warns = _fit_pe_tile_size(
        info.block_m, _PE_GRID_ROWS, _NKI_GEMM_STAT_FMAX, "M")
    col_size, col_warns = _fit_pe_tile_size(
        info.block_n, _PE_GRID_COLS, _NKI_GEMM_MOV_FMAX,  "N")
    return (row_size, col_size), row_warns + col_warns


def validate_nki_constraints(info: MatmulInfo) -> list[str]:
    """Check extracted tile sizes against NKI nc_matmul hardware limits."""
    warnings: list[str] = []
    if info.block_k > _NKI_PMAX:
        warnings.append(
            f"BLOCK_K={info.block_k} exceeds NKI pmax={_NKI_PMAX}."
        )
    _, tile_warns = select_tile_size(info)
    warnings.extend(tile_warns)
    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CONVERSION PASS INFRASTRUCTURE
#
#     ConversionTarget  — mirrors mlir::ConversionTarget
#     CodeEmitter       — shared indented-line buffer (mirrors PatternRewriter
#                         in code-generation mode)
#     RewritePattern    — abstract base (mirrors mlir::ConversionPattern)
# ─────────────────────────────────────────────────────────────────────────────

class ConversionTarget:
    """
    Declares which ops are 'legal' (already in the target NKI dialect) and
    which are 'illegal' (must be converted by a registered RewritePattern).

    Mirrors mlir::ConversionTarget.

    A conversion pass is considered successful if all illegal ops in the input
    IR have been handled by a matching pattern.
    """

    def __init__(self) -> None:
        self._legal:   set[str] = set()
        self._illegal: set[str] = set()

    def add_legal_op(self, *op_names: str) -> "ConversionTarget":
        """Mark ops as legal (no conversion needed)."""
        for n in op_names:
            self._legal.add(n)
        return self

    def add_illegal_op(self, *op_names: str) -> "ConversionTarget":
        """Mark ops as illegal (must be handled by a pattern)."""
        for n in op_names:
            self._illegal.add(n)
        return self

    def is_legal(self, op_name: str) -> bool:
        return op_name in self._legal

    def is_illegal(self, op_name: str) -> bool:
        return op_name in self._illegal

    def illegal_ops(self) -> frozenset[str]:
        return frozenset(self._illegal)


class CodeEmitter:
    """
    Shared indented Python-source-line buffer used by all RewritePatterns.

    Mirrors the role of mlir::PatternRewriter in code-generation passes:
    it provides the mechanism for patterns to emit target code without
    knowing about each other's output.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._depth = 0

    def emit(self, line: str = "") -> None:
        self._lines.append("    " * self._depth + line)

    def indent(self) -> None:
        self._depth += 1

    def dedent(self) -> None:
        self._depth -= 1

    def get_source(self) -> str:
        return "\n".join(self._lines) + "\n"


class RewritePattern(ABC):
    """
    Abstract base for a single source-op-group → target-op translation.

    Mirrors mlir::ConversionPattern (or RewritePattern with a TypeConverter).

    Subclasses implement:
      match(kernel)          — return True if this pattern applies to *kernel*
      rewrite(kernel, emitter, source_path)
                             — emit NKI Python source into *emitter*

    The _emit_* naming convention inside subclasses mirrors how MLIR C++ passes
    break matchAndRewrite into per-op helper methods for readability.
    """

    @abstractmethod
    def match(self, kernel: ParsedKernel) -> bool:
        """Return True if this pattern handles *kernel*."""

    @abstractmethod
    def rewrite(self, kernel: ParsedKernel, emitter: CodeEmitter,
                source_path: str) -> None:
        """Emit NKI Python source for *kernel* into *emitter*."""


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONCRETE REWRITE PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

class MatmulConversionPattern(RewritePattern):
    """
    Converts a linalg matmul kernel to an NKI @nki.jit kernel using the
    Tensor Engine (nisa.nc_matmul).

    Source ops handled (each has a dedicated _emit_* method below):

      func.func @name(...)                 →  _emit_func_signature
      arith.constant BM/BN/BK             →  _emit_tile_constants
      memref.reinterpret_cast %arg2 (C)   →  _emit_output_alloc
      arith.divsi/remsi %pid              →  _emit_outer_loops   (loop synthesis)
      tensor.empty + linalg.fill (zero)   →  _emit_psum_init
      scf.for %k = 0 to cdiv(K, BK)      →  _emit_k_loop
      reinterpret_cast+copy (A tile)      →  _emit_a_tile_load
      reinterpret_cast+copy (B tile)      →  _emit_b_tile_load
      linalg.matmul + linalg.add          →  _emit_nc_matmul
      tensor.extract_slice +
        bufferization.materialize_in_dest →  _emit_matmul_store

    NKI semantic note:
      nisa.nc_matmul(dst=psum, stationary=lhsT, moving=rhs)
      computes  psum += lhsT.T @ rhs

      The linalg IR computes  acc += A_tile @ B_tile  (A_tile shape [BM, BK]).
      We load A as lhsT with shape [BK, BM] so that lhsT.T = A_tile, recovering
      the original semantics.  (Matches the nki_matmul_tiled_ tutorial convention.)
    """

    # ── match ─────────────────────────────────────────────────────────────────

    def match(self, kernel: ParsedKernel) -> bool:
        return isinstance(kernel, MatmulInfo)

    # ── rewrite: top-level driver ─────────────────────────────────────────────

    def rewrite(self, kernel: ParsedKernel, emitter: CodeEmitter,
                source_path: str) -> None:
        assert isinstance(kernel, MatmulInfo)
        info = kernel
        (pe_row, pe_col), _ = select_tile_size(info)

        self._emit_file_header(info, pe_row, pe_col, source_path, emitter)
        self._emit_imports(emitter)
        emitter.emit()
        emitter.emit()

        # func.func → @nki.jit def
        self._emit_func_signature(info, emitter)
        emitter.indent()

        # arith.constant → Python constants
        self._emit_tile_constants(info, pe_row, pe_col, emitter)

        # reinterpret_cast %arg2 → nl.shared_hbm output
        self._emit_output_alloc(emitter)

        # program_id dispatch → explicit for-m / for-n loops
        self._emit_outer_loops(info, pe_row, pe_col, emitter)

        emitter.emit()
        emitter.emit("return C")
        emitter.dedent()

    # ── per-op-pattern emit methods ───────────────────────────────────────────

    def _emit_file_header(self, info: MatmulInfo, pe_row: int, pe_col: int,
                          source_path: str, e: CodeEmitter) -> None:
        """Emit the module docstring describing the translation."""
        BM, BN, BK = info.block_m, info.block_n, info.block_k
        row_slots = 128 // pe_row
        col_slots = 128 // pe_col
        e.emit('"""')
        e.emit("Auto-generated NKI kernel.")
        e.emit(f"Source IR : {source_path}")
        e.emit("Generator : linalg_to_nki.py  (MatmulConversionPattern)")
        e.emit("Pipeline  : Triton → TTIR → triton-shared → Linalg → NKI")
        e.emit()
        e.emit("Op-pattern translation table (source → target):")
        e.emit(f"  func.func @{info.func_name}(...)         → @nki.jit def {info.func_name}_nki(A, B)")
        e.emit(f"  arith.constant {BM}/{BN}/{BK}            → TILE_M/N/K constants")
        e.emit(f"  arith.divsi/remsi %pid                   → for m/n in nl.affine_range(...)")
        e.emit(f"  tensor.empty + linalg.fill (zeros)       → nl.ndarray(..., buffer=nl.psum)")
        e.emit(f"  scf.for %k = 0 to cdiv(K,{BK})          → for k in nl.affine_range(K // TILE_K)")
        e.emit(f"  reinterpret_cast + copy (A [{BM}×{BK}]) → nisa.dma_copy → lhsT_tile [TILE_K, TILE_M]")
        e.emit(f"  reinterpret_cast + copy (B [{BK}×{BN}]) → nisa.dma_copy → rhs_tile  [TILE_K, TILE_N]")
        e.emit(f"  linalg.matmul + linalg.add               → nisa.nc_matmul(tile_size=({pe_row},{pe_col}))")
        e.emit(f"  materialize_in_destination               → nisa.tensor_copy + nisa.dma_copy")
        e.emit()
        e.emit(f"Tile sizes:  TILE_M={BM}  TILE_N={BN}  TILE_K={BK}")
        e.emit(f"PE-grid:     tile_size=({pe_row},{pe_col})  →  {row_slots}×{col_slots} parallel slots")
        e.emit('"""')

    def _emit_imports(self, e: CodeEmitter) -> None:
        """Emit NKI import statements."""
        e.emit("import nki")
        e.emit("import nki.isa as nisa")
        e.emit("import nki.language as nl")

    def _emit_func_signature(self, info: MatmulInfo, e: CodeEmitter) -> None:
        """
        Op pattern: func.func @<name>(%arg0: memref<*xf32>, …)

        The linalg function takes raw pointer arguments for A, B, C plus
        integer strides and program IDs.  In NKI these become typed HBM
        tensors passed directly; strides are implicit in tensor layout.
        """
        e.emit("@nki.jit")
        e.emit(f"def {info.func_name}_nki(A, B):")

    def _emit_tile_constants(self, info: MatmulInfo, pe_row: int, pe_col: int,
                             e: CodeEmitter) -> None:
        """
        Op pattern: arith.constant 64 : index  (BLOCK_SIZE_M / _N)
                    arith.constant 32 : i32     (BLOCK_SIZE_K)

        These map directly to Python integer constants.
        """
        BM, BN, BK = info.block_m, info.block_n, info.block_k
        e.emit("# arith.constant ops → tile-size Python constants")
        e.emit(f"TILE_M = {BM}   # %c{BM} = arith.constant {BM} : index")
        e.emit(f"TILE_N = {BN}   # %c{BN} = arith.constant {BN} : index")
        e.emit(f"TILE_K = {BK}   # %c{BK}_i32 = arith.constant {BK} : i32")
        e.emit(f"PE_ROW_SIZE = {pe_row}  # smallest valid PE row slot >= TILE_M")
        e.emit(f"PE_COL_SIZE = {pe_col}  # smallest valid PE col slot >= TILE_N")
        e.emit()
        e.emit("M, K = A.shape")
        e.emit("K_, N = B.shape")
        e.emit('assert K == K_, f"Contraction mismatch: A.K={K} != B.K={K_}"')
        e.emit("assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0, \\")
        e.indent()
        e.emit('"M, N, K must be divisible by their respective tile sizes"')
        e.dedent()
        e.emit("assert TILE_M <= PE_ROW_SIZE and TILE_N <= PE_COL_SIZE, \\")
        e.indent()
        e.emit('"Tile dims must not exceed chosen PE-grid tile_size"')
        e.dedent()
        e.emit()

    def _emit_output_alloc(self, e: CodeEmitter) -> None:
        """
        Op pattern: memref.reinterpret_cast %arg2 (c_ptr) → output memref

        In linalg the output is accessed via a raw pointer + strides.
        In NKI it is a typed ndarray in HBM; the runtime handles layout.
        """
        e.emit("# memref.reinterpret_cast %arg2 (c_ptr) → nl.shared_hbm tensor")
        e.emit("C = nl.ndarray((M, N), dtype=A.dtype, buffer=nl.shared_hbm)")
        e.emit()

    def _emit_outer_loops(self, info: MatmulInfo, pe_row: int, pe_col: int,
                          e: CodeEmitter) -> None:
        """
        Op pattern: arith.divsi %pid, %num_pid_n  →  pid_m
                    arith.remsi %pid, %num_pid_n  →  pid_n

        The linalg kernel is a single-program-id function: one invocation
        handles one (pid_m, pid_n) output tile.  NKI expresses the same
        computation with explicit affine loops.
        """
        e.emit("# arith.divsi/remsi %pid → (pid_m, pid_n)  →  explicit tile loops")
        e.emit("for m in nl.affine_range(M // TILE_M):")
        e.indent()
        e.emit("for n in nl.affine_range(N // TILE_N):")
        e.indent()
        e.emit()
        self._emit_psum_init(e)
        self._emit_k_loop(pe_row, pe_col, e)
        self._emit_matmul_store(e)
        e.dedent()
        e.dedent()

    def _emit_psum_init(self, e: CodeEmitter) -> None:
        """
        Op pattern: %0 = tensor.empty() : tensor<BM×BN×f32>
                    %1 = linalg.fill ins(%cst : f32) outs(%0) → tensor<BM×BN×f32>

        linalg.fill with a zero constant initialises the accumulator.
        PSUM is the NKI accumulator register file; a fresh nl.ndarray into
        nl.psum is implicitly zeroed before the first nc_matmul call.
        """
        e.emit("# tensor.empty + linalg.fill (zeros) → PSUM accumulator")
        e.emit("res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)")
        e.emit()

    def _emit_k_loop(self, pe_row: int, pe_col: int, e: CodeEmitter) -> None:
        """
        Op pattern: scf.for %k = %c0_i32 to %num_k_iters step %c1_i32
                        iter_args(%acc = %1) → tensor<BM×BN×f32>

        The loop-carried accumulator (%acc) maps to the PSUM buffer that
        nc_matmul writes into across iterations.
        """
        e.emit("# scf.for %k = 0 to cdiv(K, TILE_K) step 1 → affine_range K-loop")
        e.emit("for k in nl.affine_range(K // TILE_K):")
        e.indent()
        e.emit()
        self._emit_a_tile_load(e)
        self._emit_b_tile_load(e)
        self._emit_nc_matmul(pe_row, pe_col, e)
        e.dedent()
        e.emit()

    def _emit_a_tile_load(self, e: CodeEmitter) -> None:
        """
        Op pattern (inside scf.for):
          memref.reinterpret_cast %arg0  offset:[rm_start + k_offset*stride_ak]
          memref.subview ... [valid_rows, BK]
          memref.copy subview, alloc
          bufferization.to_tensor %alloc

        NKI: dma_copy from A[k*BK:(k+1)*BK, m*BM:(m+1)*BM] into SBUF as
        lhsT [TILE_K, TILE_M].  Loading as transposed so nc_matmul computes
        lhsT.T @ rhs = A_tile @ B_tile.
        """
        e.emit("# reinterpret_cast %arg0 + memref.copy (A tile [BM×BK])")
        e.emit("# → dma_copy A[k*BK:(k+1)*BK, m*BM:(m+1)*BM] as lhsT [TILE_K, TILE_M]")
        e.emit("lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=A.dtype, buffer=nl.sbuf)")
        e.emit("nisa.dma_copy(")
        e.indent()
        e.emit("dst=lhsT_tile,")
        e.emit("src=A[k * TILE_K : (k + 1) * TILE_K,")
        e.indent()
        e.emit("m * TILE_M : (m + 1) * TILE_M],")
        e.dedent()
        e.dedent()
        e.emit(")")
        e.emit()

    def _emit_b_tile_load(self, e: CodeEmitter) -> None:
        """
        Op pattern (inside scf.for):
          memref.reinterpret_cast %arg1  offset:[k_offset + rn_start*stride_bn]
          memref.subview ... [BK, valid_cols]
          memref.copy subview, alloc
          bufferization.to_tensor %alloc

        Symmetric to the A-tile load.  B is loaded as rhs with shape
        [TILE_K, TILE_N]; the moving dimension spans PE columns.
        """
        e.emit("# reinterpret_cast %arg1 + memref.copy (B tile [BK×BN])")
        e.emit("# → dma_copy B[k*BK:(k+1)*BK, n*BN:(n+1)*BN] as rhs [TILE_K, TILE_N]")
        e.emit("rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=B.dtype, buffer=nl.sbuf)")
        e.emit("nisa.dma_copy(")
        e.indent()
        e.emit("dst=rhs_tile,")
        e.emit("src=B[k * TILE_K : (k + 1) * TILE_K,")
        e.indent()
        e.emit("n * TILE_N : (n + 1) * TILE_N],")
        e.dedent()
        e.dedent()
        e.emit(")")
        e.emit()

    def _emit_nc_matmul(self, pe_row: int, pe_col: int, e: CodeEmitter) -> None:
        """
        Op pattern (inside scf.for):
          %61 = tensor.empty() : tensor<BM×BN×f32>
          %62 = linalg.fill ins(%cst) outs(%61)           ← temp zero tile
          %63 = linalg.matmul ins(%53, %60) outs(%62)     ← tile product
          %64 = linalg.add ins(%acc, %63) outs(%acc)      ← accumulate
          scf.yield %64

        The two-op pattern linalg.matmul + linalg.add (into the loop-carried
        accumulator) maps to a single nc_matmul which accumulates into PSUM.

        nc_matmul semantics: dst += stationary.T @ moving
          stationary = lhsT_tile [TILE_K, TILE_M]  →  .T = A_tile [TILE_M, TILE_K]
          moving     = rhs_tile  [TILE_K, TILE_N]
          result     = res_psum  [TILE_M, TILE_N]   ✓  matches linalg output
        """
        row_slots = 128 // pe_row
        col_slots = 128 // pe_col
        e.emit("# tensor.empty + linalg.fill + linalg.matmul + linalg.add + scf.yield")
        e.emit("# → nisa.nc_matmul (accumulates into PSUM across K iterations)")
        e.emit(f"# tile_size=({pe_row},{pe_col}): {row_slots}×{col_slots}={row_slots*col_slots} parallel PE slots")
        e.emit("nisa.nc_matmul(")
        e.indent()
        e.emit("dst=res_psum,")
        e.emit("stationary=lhsT_tile,   # lhsT_tile.T = A_tile  →  A_tile @ B_tile")
        e.emit("moving=rhs_tile,")
        e.emit(f"tile_size=({pe_row}, {pe_col}),")
        e.dedent()
        e.emit(")")

    def _emit_matmul_store(self, e: CodeEmitter) -> None:
        """
        Op pattern (after scf.for):
          tensor.extract_slice %result [0,0][valid_rows,valid_cols][1,1]
          memref.subview %reinterpret_cast(c_ptr) [0,0][valid_rows,valid_cols][1,1]
          bufferization.materialize_in_destination %slice in %subview

        In NKI this is a two-step move:
          PSUM → SBUF  (nisa.tensor_copy, with dtype cast float32→A.dtype)
          SBUF → HBM   (nisa.dma_copy, into the C output tile)
        """
        e.emit("# tensor.extract_slice + memref.subview")
        e.emit("# + bufferization.materialize_in_destination (store C tile)")
        e.emit("# → nisa.tensor_copy PSUM→SBUF  then  nisa.dma_copy SBUF→HBM")
        e.emit("res_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=A.dtype, buffer=nl.sbuf)")
        e.emit("nisa.tensor_copy(dst=res_sbuf, src=res_psum, dtype=A.dtype)")
        e.emit("nisa.dma_copy(")
        e.indent()
        e.emit("dst=C[m * TILE_M : (m + 1) * TILE_M,")
        e.indent()
        e.emit("n * TILE_N : (n + 1) * TILE_N],")
        e.dedent()
        e.emit("src=res_sbuf,")
        e.dedent()
        e.emit(")")


class VecAddConversionPattern(RewritePattern):
    """
    Converts a linalg 1-D vector-add kernel to an NKI @nki.jit kernel using
    the Vector Engine (nl.add).

    Source ops handled (each has a dedicated _emit_* method below):

      func.func @name(...)                 →  _emit_func_signature
      arith.constant 1024 : index          →  _emit_tile_constant
      arith.muli %pid, %c1024 → offset     →  _emit_outer_loop   (loop synthesis)
      reinterpret_cast + copy (x tile)     →  _emit_x_tile_load
      reinterpret_cast + copy (y tile)     →  _emit_y_tile_load
      linalg.add ins(%x, %y) outs(%x)      →  _emit_vec_add
      tensor.extract_slice +
        bufferization.materialize_in_dest  →  _emit_vec_add_store

    NKI Vector Engine note:
      nl.add(x_tile, y_tile) routes through the Vector Engine (VE), which
      operates on the sbuf partition axis (max 128 elements) × free axis.
      For BLOCK_SIZE=1024 elements, the load/add/store are pipelined over
      the natural partitioning of the 1-D tile buffer.
    """

    # ── match ─────────────────────────────────────────────────────────────────

    def match(self, kernel: ParsedKernel) -> bool:
        return isinstance(kernel, VecAddInfo)

    # ── rewrite: top-level driver ─────────────────────────────────────────────

    def rewrite(self, kernel: ParsedKernel, emitter: CodeEmitter,
                source_path: str) -> None:
        assert isinstance(kernel, VecAddInfo)
        info = kernel

        self._emit_file_header(info, source_path, emitter)
        self._emit_imports(emitter)
        emitter.emit()
        emitter.emit()

        # func.func → @nki.jit def
        self._emit_func_signature(info, emitter)
        emitter.indent()

        # arith.constant → Python constant
        self._emit_tile_constant(info, emitter)

        # arith.muli %pid, %c1024 → explicit loop over blocks
        self._emit_outer_loop(info, emitter)

        emitter.emit()
        emitter.emit("return out")
        emitter.dedent()

    # ── per-op-pattern emit methods ───────────────────────────────────────────

    def _emit_file_header(self, info: VecAddInfo, source_path: str,
                          e: CodeEmitter) -> None:
        """Emit the module docstring describing the translation."""
        BS = info.block_size
        e.emit('"""')
        e.emit("Auto-generated NKI kernel.")
        e.emit(f"Source IR : {source_path}")
        e.emit("Generator : linalg_to_nki.py  (VecAddConversionPattern)")
        e.emit("Pipeline  : Triton → TTIR → triton-shared → Linalg → NKI")
        e.emit()
        e.emit("Op-pattern translation table (source → target):")
        e.emit(f"  func.func @{info.func_name}(...)         → @nki.jit def {info.func_name}_nki(x, y, out)")
        e.emit(f"  arith.constant {BS} : index              → BLOCK_SIZE constant")
        e.emit(f"  arith.muli %pid, %c{BS}                  → for block in nl.affine_range(...)")
        e.emit(f"  reinterpret_cast + copy (x tile [{BS}])  → nisa.dma_copy → x_tile [BLOCK_SIZE]")
        e.emit(f"  reinterpret_cast + copy (y tile [{BS}])  → nisa.dma_copy → y_tile [BLOCK_SIZE]")
        e.emit(f"  linalg.add ins(%x, %y)                  → nl.add (Vector Engine)")
        e.emit(f"  materialize_in_destination               → nisa.dma_copy result → HBM")
        e.emit()
        e.emit(f"Tile size:   BLOCK_SIZE={BS}")
        e.emit(f"Engine:      Vector Engine (nl.add)")
        e.emit('"""')

    def _emit_imports(self, e: CodeEmitter) -> None:
        """Emit NKI import statements."""
        e.emit("import nki")
        e.emit("import nki.isa as nisa")
        e.emit("import nki.language as nl")

    def _emit_func_signature(self, info: VecAddInfo, e: CodeEmitter) -> None:
        """
        Op pattern: func.func @<name>(%arg0: memref<*xf32>, %arg1: memref<*xf32>,
                                       %arg2: memref<*xf32>, %arg3: i32, ...)

        The linalg function takes x_ptr, y_ptr, out_ptr plus n_elements and
        program IDs.  In NKI these become typed HBM tensors; the caller
        supplies n_elements as a scalar.
        """
        e.emit("@nki.jit")
        e.emit(f"def {info.func_name}_nki(x, y, out):")

    def _emit_tile_constant(self, info: VecAddInfo, e: CodeEmitter) -> None:
        """
        Op pattern: %c1024 = arith.constant 1024 : index
                    %c1024_i32 = arith.constant 1024 : i32

        Maps directly to a Python integer constant.
        """
        BS = info.block_size
        e.emit("# arith.constant 1024 : index → BLOCK_SIZE Python constant")
        e.emit(f"BLOCK_SIZE = {BS}   # %c{BS} = arith.constant {BS} : index")
        e.emit()
        e.emit("n_elements = x.shape[0]")
        e.emit("assert n_elements % BLOCK_SIZE == 0, \\")
        e.indent()
        e.emit('"n_elements must be divisible by BLOCK_SIZE"')
        e.dedent()
        e.emit()

    def _emit_outer_loop(self, info: VecAddInfo, e: CodeEmitter) -> None:
        """
        Op pattern: %0  = arith.muli %arg7, %c1024_i32     (pid * BLOCK_SIZE)
                    %1  = arith.index_cast %0               (offset as index)
                    %reinterpret_cast = memref.reinterpret_cast %arg0
                        offset:[%1] sizes:[1024] strides:[1]

        In Triton each program instance handles one block starting at
        pid * BLOCK_SIZE.  NKI loops over all blocks explicitly.
        """
        e.emit("# arith.muli %pid, %c1024 → pid * BLOCK_SIZE  →  explicit block loop")
        e.emit("for block in nl.affine_range(n_elements // BLOCK_SIZE):")
        e.indent()
        e.emit("offset = block * BLOCK_SIZE")
        e.emit()
        self._emit_x_tile_load(e)
        self._emit_y_tile_load(e)
        self._emit_vec_add(e)
        self._emit_vec_add_store(e)
        e.dedent()

    def _emit_x_tile_load(self, e: CodeEmitter) -> None:
        """
        Op pattern:
          %reinterpret_cast = memref.reinterpret_cast %arg0
              offset:[pid*1024] sizes:[1024] strides:[1]
          %alloc = memref.alloc() : memref<1024xf32>
          %subview   = memref.subview %reinterpret_cast[0][valid_n][1]
          %subview_0 = memref.subview %alloc[0][valid_n][1]
          memref.copy %subview, %subview_0
          %8 = bufferization.to_tensor %alloc restrict writable

        NKI: dma_copy the x block from HBM into SBUF.
        """
        e.emit("# reinterpret_cast %arg0 + memref.copy (x block [BLOCK_SIZE])")
        e.emit("# → dma_copy x[offset:offset+BLOCK_SIZE] → x_tile")
        e.emit("x_tile = nl.ndarray((BLOCK_SIZE,), dtype=x.dtype, buffer=nl.sbuf)")
        e.emit("nisa.dma_copy(dst=x_tile, src=x[offset : offset + BLOCK_SIZE])")
        e.emit()

    def _emit_y_tile_load(self, e: CodeEmitter) -> None:
        """
        Symmetric to _emit_x_tile_load for the second operand (%arg1 / y).
        """
        e.emit("# reinterpret_cast %arg1 + memref.copy (y block [BLOCK_SIZE])")
        e.emit("# → dma_copy y[offset:offset+BLOCK_SIZE] → y_tile")
        e.emit("y_tile = nl.ndarray((BLOCK_SIZE,), dtype=y.dtype, buffer=nl.sbuf)")
        e.emit("nisa.dma_copy(dst=y_tile, src=y[offset : offset + BLOCK_SIZE])")
        e.emit()

    def _emit_vec_add(self, e: CodeEmitter) -> None:
        """
        Op pattern:
          %17 = linalg.add ins(%8, %16 : tensor<1024xf32>, tensor<1024xf32>)
                           outs(%8  : tensor<1024xf32>) → tensor<1024xf32>

        Maps to nl.add which routes through the Vector Engine (VE).
        nl.add is a higher-level NKI operation that handles the
        partition-dimension tiling of the 1-D sbuf internally.
        """
        e.emit("# linalg.add ins(%x, %y) outs(%x) → nl.add (Vector Engine)")
        e.emit("result_tile = nl.add(x_tile, y_tile)")
        e.emit()

    def _emit_vec_add_store(self, e: CodeEmitter) -> None:
        """
        Op pattern:
          %extracted_slice = tensor.extract_slice %17[0][valid_n][1]
          %subview_6 = memref.subview %reinterpret_cast_5[0][valid_n][1]
          bufferization.materialize_in_destination %extracted_slice in %subview_6

        NKI: dma_copy the result tile from SBUF back to HBM.
        """
        e.emit("# tensor.extract_slice + bufferization.materialize_in_destination")
        e.emit("# → nisa.dma_copy result_tile → out[offset:offset+BLOCK_SIZE]")
        e.emit("nisa.dma_copy(dst=out[offset : offset + BLOCK_SIZE], src=result_tile)")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CONVERSION PASS
#
#     LinalgToNKIConversionPass orchestrates:
#       1. ConversionTarget declaration
#       2. RewritePattern registration
#       3. IR parsing (extract ParsedKernel)
#       4. Pattern matching + application
#
#     Mirrors the structure of an MLIR pass that calls:
#       populateLinalgToNKIConversionPatterns(patterns, target);
#       applyFullConversion(module, target, std::move(patterns));
# ─────────────────────────────────────────────────────────────────────────────

class LinalgToNKIConversionPass:
    """
    Full conversion pass: Linalg dialect → NKI Python kernel.

    Mirrors an MLIR ConversionPass class.  The constructor declares the
    ConversionTarget (which ops are illegal in the target dialect) and
    registers the concrete RewritePatterns.  run() drives the conversion.

    ConversionTarget for this pass:
      Illegal ops (must be converted):
        linalg.matmul, linalg.add, scf.for,
        memref.reinterpret_cast, memref.copy,
        bufferization.materialize_in_destination,
        bufferization.to_tensor, tensor.empty, tensor.extract_slice
      Legal ops (already in NKI / Python output):
        nki.jit, nisa.nc_matmul, nisa.dma_copy, nisa.tensor_copy, nl.add,
        nl.ndarray, nl.affine_range, nl.psum, nl.sbuf, nl.shared_hbm
    """

    def __init__(self) -> None:
        # ── Conversion target ─────────────────────────────────────────────────
        self.target = ConversionTarget()

        # Ops that exist in the source Linalg IR and must be converted away.
        self.target.add_illegal_op(
            "linalg.matmul",
            "linalg.add",
            "linalg.fill",
            "scf.for",
            "memref.reinterpret_cast",
            "memref.copy",
            "bufferization.materialize_in_destination",
            "bufferization.to_tensor",
            "tensor.empty",
            "tensor.extract_slice",
        )

        # Ops that are already legal in the NKI target.
        self.target.add_legal_op(
            "nki.jit",
            "nisa.nc_matmul",
            "nisa.dma_copy",
            "nisa.tensor_copy",
            "nl.add",
            "nl.ndarray",
            "nl.affine_range",
            "nl.psum",
            "nl.sbuf",
            "nl.shared_hbm",
        )

        # ── Registered patterns (applied in order; first match wins) ──────────
        # Mirrors: populateLinalgToNKIConversionPatterns(patterns, target)
        self._patterns: list[RewritePattern] = [
            MatmulConversionPattern(),
            VecAddConversionPattern(),
        ]

    # ── public entry point ────────────────────────────────────────────────────

    def run(self, ir_text: str, source_path: str = "<unknown>",
            verbose: bool = True) -> str:
        """
        Apply this conversion pass to the linalg IR text.

        Steps:
          1. Parse the IR to extract a ParsedKernel (kind + tile parameters).
          2. Find the first registered pattern that matches the kernel kind.
          3. Apply the pattern (emit NKI Python source).
          4. Return the generated source string.

        Raises ValueError if no registered pattern matches the kernel.
        """
        # Step 1: parse
        kernel = parse_linalg_ir(ir_text)

        if verbose:
            self._log_parsed_kernel(kernel)
            if isinstance(kernel, MatmulInfo):
                for w in validate_nki_constraints(kernel):
                    print(f"[linalg_to_nki] WARNING: {w}")

        # Step 2: match
        pattern = self._select_pattern(kernel)
        if pattern is None:
            kinds = [type(p).__name__ for p in self._patterns]
            raise ValueError(
                f"No registered pattern handles kernel kind {kernel.kind!r}. "
                f"Registered: {kinds}"
            )

        if verbose:
            print(f"[linalg_to_nki] Applying pattern: {type(pattern).__name__}")

        # Step 3: apply
        emitter = CodeEmitter()
        pattern.rewrite(kernel, emitter, source_path)
        return emitter.get_source()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _select_pattern(self, kernel: ParsedKernel) -> Optional[RewritePattern]:
        """Return the first registered pattern whose match() returns True."""
        for p in self._patterns:
            if p.match(kernel):
                return p
        return None

    def _log_parsed_kernel(self, kernel: ParsedKernel) -> None:
        if isinstance(kernel, MatmulInfo):
            print(
                f"[linalg_to_nki] Parsed MATMUL  "
                f"func={kernel.func_name!r}  "
                f"BM={kernel.block_m} BN={kernel.block_n} BK={kernel.block_k}  "
                f"k_loop={kernel.has_k_loop}  bounds_check={kernel.has_bounds_check}"
            )
        elif isinstance(kernel, VecAddInfo):
            print(
                f"[linalg_to_nki] Parsed VEC_ADD  "
                f"func={kernel.func_name!r}  "
                f"BLOCK_SIZE={kernel.block_size}  "
                f"bounds_check={kernel.has_bounds_check}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DRIVER
# ─────────────────────────────────────────────────────────────────────────────

def linalg_to_nki(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    verbose: bool = True,
) -> str:
    """
    Top-level entry point.

    Reads the linalg MLIR file at *input_path*, instantiates
    LinalgToNKIConversionPass, runs it, and optionally writes the result to
    *output_path*.

    Returns the generated NKI source as a string.
    """
    with open(input_path) as f:
        ir_text = f.read()

    pass_ = LinalgToNKIConversionPass()
    nki_src = pass_.run(ir_text, source_path=input_path, verbose=verbose)

    if output_path:
        with open(output_path, "w") as f:
            f.write(nki_src)
        if verbose:
            print(f"[linalg_to_nki] Written → {output_path}")

    return nki_src


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input.linalg|input.mlir> [output_nki.py]")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = (sys.argv[2] if len(sys.argv) > 2
                else re.sub(r'\.(linalg|mlir)$', '', in_path) + "_nki.py")
    linalg_to_nki(in_path, out_path)
