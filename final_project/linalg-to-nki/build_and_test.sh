#!/usr/bin/env bash
#===------------------------------------------------------------------------===
# build_and_test.sh
#
# One-shot configure + build + smoke-test for the linalg-to-nki MLIR pass.
#
# Layout assumed:
#   final_project/
#     linalg-to-nki/        <-- this script lives here
#       build/              <-- created by this script
#     llvm-project/
#       build/lib/cmake/{llvm,mlir}
#
# Usage:
#   ./build_and_test.sh                # configure (if needed), build, test
#   ./build_and_test.sh --clean        # nuke build/ first
#   ./build_and_test.sh --kernel       # also run pass on ../matmul_kernel.linalg
#   LLVM_BUILD_DIR=/path ./build_and_test.sh   # override llvm-project/build
#===------------------------------------------------------------------------===
set -euo pipefail

# --- Resolve paths relative to this script ----------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
FINAL_PROJECT_DIR="$(cd -- "${PROJECT_DIR}/.." &>/dev/null && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${FINAL_PROJECT_DIR}/llvm-project/build}"

# --- Parse args -------------------------------------------------------------
CLEAN=0
RUN_KERNEL=0
for arg in "$@"; do
  case "$arg" in
    --clean)  CLEAN=1 ;;
    --kernel) RUN_KERNEL=1 ;;
    -h|--help)
      sed -n '2,21p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# --- Pretty logging ---------------------------------------------------------
log()  { printf '\033[1;34m[build_and_test]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[build_and_test] %s\033[0m\n' "$*" >&2; exit 1; }

# --- Sanity checks ----------------------------------------------------------
[[ -f "${PROJECT_DIR}/CMakeLists.txt" ]] \
  || fail "CMakeLists.txt not found at ${PROJECT_DIR}"
[[ -d "${LLVM_BUILD_DIR}/lib/cmake/mlir" ]] \
  || fail "MLIR cmake dir not found at ${LLVM_BUILD_DIR}/lib/cmake/mlir
         set LLVM_BUILD_DIR=/path/to/llvm-project/build to override."

command -v cmake >/dev/null  || fail "cmake not on PATH"
GENERATOR="Unix Makefiles"
GENERATOR_FILE="Makefile"
if command -v ninja >/dev/null; then
  GENERATOR="Ninja"
  GENERATOR_FILE="build.ninja"
fi
log "using generator: ${GENERATOR}"

# --- Clean ------------------------------------------------------------------
if [[ "${CLEAN}" -eq 1 ]]; then
  log "removing existing build dir: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

# --- Configure --------------------------------------------------------------
# Skip configure only if BOTH the cache and the generator's build file exist;
# otherwise an aborted prior configure can leave us in a half-state.
NEED_CONFIGURE=1
if [[ -f "${BUILD_DIR}/CMakeCache.txt" && -f "${BUILD_DIR}/${GENERATOR_FILE}" ]]; then
  NEED_CONFIGURE=0
fi

if [[ "${NEED_CONFIGURE}" -eq 1 ]]; then
  if [[ -f "${BUILD_DIR}/CMakeCache.txt" && ! -f "${BUILD_DIR}/${GENERATOR_FILE}" ]]; then
    log "stale build dir (cache present, ${GENERATOR_FILE} missing); wiping ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
  fi
  log "configuring (LLVM_BUILD_DIR=${LLVM_BUILD_DIR})"
  cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -G "${GENERATOR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
    -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"
else
  log "build dir already configured, skipping cmake configure"
fi

# --- Build ------------------------------------------------------------------
log "building linalg-to-nki-opt + linalg-to-nki-translate"
cmake --build "${BUILD_DIR}" --target linalg-to-nki-opt
cmake --build "${BUILD_DIR}" --target linalg-to-nki-translate

OPT_BIN="${BUILD_DIR}/bin/linalg-to-nki-opt"
TRANSLATE_BIN="${BUILD_DIR}/bin/linalg-to-nki-translate"
[[ -x "${OPT_BIN}" ]] || fail "expected binary not found: ${OPT_BIN}"
[[ -x "${TRANSLATE_BIN}" ]] || fail "expected binary not found: ${TRANSLATE_BIN}"
log "built: ${OPT_BIN}"
log "built: ${TRANSLATE_BIN}"

# --- Smoke test 1: parse + print round-trip --------------------------------
TEST_FILE="${PROJECT_DIR}/test/matmul_add_fuse.mlir"
[[ -f "${TEST_FILE}" ]] || fail "test file missing: ${TEST_FILE}"

log "round-trip parse of ${TEST_FILE##*/}"
"${OPT_BIN}" "${TEST_FILE}" >/dev/null

# --- Smoke test 2: run the pass and grep for expected output ---------------
log "running -linalg-to-nki on ${TEST_FILE##*/}"
PASS_OUT="$("${OPT_BIN}" "${TEST_FILE}" -linalg-to-nki)"
echo "----- pass output -----"
echo "${PASS_OUT}"
echo "-----------------------"

if echo "${PASS_OUT}" | grep -q 'nki.nc_matmul'; then
  log "PASS: nki.nc_matmul present in output"
else
  fail "expected 'nki.nc_matmul' in pass output, but it is missing"
fi

if echo "${PASS_OUT}" | grep -qE 'linalg\.(matmul|add)'; then
  fail "leftover linalg.matmul / linalg.add in output (fusion did not fire)"
else
  log "PASS: no residual linalg.matmul / linalg.add"
fi

# --- Smoke test 3: run the dma-fuse pass and grep for expected output ------
DMA_TEST_FILE="${PROJECT_DIR}/test/dma_fuse.mlir"
[[ -f "${DMA_TEST_FILE}" ]] || fail "test file missing: ${DMA_TEST_FILE}"

log "running -nki-fuse-dma on ${DMA_TEST_FILE##*/}"
DMA_OUT="$("${OPT_BIN}" "${DMA_TEST_FILE}" -nki-fuse-dma)"
echo "----- dma-fuse pass output -----"
echo "${DMA_OUT}"
echo "--------------------------------"

if echo "${DMA_OUT}" | grep -q 'nki.dma_copy'; then
  log "PASS: nki.dma_copy present in output"
else
  fail "expected 'nki.dma_copy' in pass output, but it is missing"
fi

if echo "${DMA_OUT}" | grep -qE 'memref\.(reinterpret_cast|alloc|subview|copy)|bufferization\.to_tensor'; then
  fail "leftover memref/to_tensor chain in output (dma fusion did not fire)"
else
  log "PASS: no residual reinterpret_cast / alloc / subview / copy / to_tensor"
fi

# --- Smoke test 4: run the store-fuse pass and grep for expected output ----
STORE_TEST_FILE="${PROJECT_DIR}/test/store_fuse.mlir"
[[ -f "${STORE_TEST_FILE}" ]] || fail "test file missing: ${STORE_TEST_FILE}"

log "running -nki-fuse-store on ${STORE_TEST_FILE##*/}"
STORE_OUT="$("${OPT_BIN}" "${STORE_TEST_FILE}" -nki-fuse-store)"
echo "----- store-fuse pass output -----"
echo "${STORE_OUT}"
echo "----------------------------------"

if echo "${STORE_OUT}" | grep -q 'nki.dma_store'; then
  log "PASS: nki.dma_store present in output"
else
  fail "expected 'nki.dma_store' in pass output, but it is missing"
fi

if echo "${STORE_OUT}" | grep -qE 'memref\.(reinterpret_cast|subview)|tensor\.extract_slice|bufferization\.materialize_in_destination'; then
  fail "leftover store chain in output (store fusion did not fire)"
else
  log "PASS: no residual reinterpret_cast / subview / extract_slice / materialize_in_destination"
fi

# --- Optional: run the pass on the real triton-shared kernel ---------------
if [[ "${RUN_KERNEL}" -eq 1 ]]; then
  KERNEL_FILE="${FINAL_PROJECT_DIR}/matmul_kernel.linalg"
  [[ -f "${KERNEL_FILE}" ]] || fail "kernel file missing: ${KERNEL_FILE}"
  log "running full pipeline on ${KERNEL_FILE}"
  KERNEL_OUT="$("${OPT_BIN}" "${KERNEL_FILE}" \
    -nki-canonicalize-pid-loops \
    -linalg-to-nki \
    -nki-fuse-dma \
    -nki-fuse-store \
    -nki-fold-psum-init)"
  echo "----- kernel pass output -----"
  echo "${KERNEL_OUT}"
  echo "------------------------------"
  if echo "${KERNEL_OUT}" | grep -q 'nki.nc_matmul'; then
    log "PASS: nki.nc_matmul appears in matmul_kernel.linalg lowering"
  else
    fail "no nki.nc_matmul in matmul_kernel.linalg lowering"
  fi
  # Inside the k-loop, both A and B tile loads should have collapsed.
  DMA_COUNT="$(echo "${KERNEL_OUT}" | grep -c 'nki.dma_copy' || true)"
  if [[ "${DMA_COUNT}" -ge 2 ]]; then
    log "PASS: ${DMA_COUNT} nki.dma_copy ops in matmul_kernel.linalg lowering"
  else
    fail "expected >=2 nki.dma_copy ops, found ${DMA_COUNT}"
  fi
  # The output-tile store should collapse into one nki.dma_store.
  if echo "${KERNEL_OUT}" | grep -q 'nki.dma_store'; then
    log "PASS: nki.dma_store appears in matmul_kernel.linalg lowering"
  else
    fail "no nki.dma_store in matmul_kernel.linalg lowering"
  fi
  # And every memref / store-side leftover should be gone.
  if echo "${KERNEL_OUT}" | grep -qE 'memref\.(reinterpret_cast|alloc|subview|copy)|bufferization\.(to_tensor|materialize_in_destination)|tensor\.extract_slice'; then
    fail "leftover memref / bufferization chain in matmul_kernel.linalg lowering"
  else
    log "PASS: no residual memref / bufferization chain in matmul_kernel.linalg lowering"
  fi
  # The PSUM seed should have folded into nki.psum_alloc.
  if echo "${KERNEL_OUT}" | grep -q 'nki.psum_alloc'; then
    log "PASS: nki.psum_alloc appears in matmul_kernel.linalg lowering"
  else
    fail "no nki.psum_alloc (fold-psum-init did not fire)"
  fi
  if echo "${KERNEL_OUT}" | grep -qE 'linalg\.fill|tensor\.empty'; then
    fail "leftover linalg.fill / tensor.empty in matmul_kernel.linalg lowering"
  fi
  # The pid math should be gone -- only loop bound divs/rems remain (none on
  # function args).
  PID_LOOP_COUNT="$(echo "${KERNEL_OUT}" | grep -cE '(^|[[:space:]=])scf\.for' || true)"
  if [[ "${PID_LOOP_COUNT}" -ge 3 ]]; then
    log "PASS: ${PID_LOOP_COUNT} scf.for loops (m, n, k)"
  else
    fail "expected >=3 scf.for loops, found ${PID_LOOP_COUNT}"
  fi

  # --- Translate to Python ---------------------------------------------------
  log "translating lowered IR to NKI Python"
  TMP_LOWERED="$(mktemp --tmpdir lowered.XXXXXX.mlir)"
  TMP_PY="$(mktemp --tmpdir kernel.XXXXXX.py)"
  trap 'rm -f "${TMP_LOWERED}" "${TMP_PY}"' EXIT
  echo "${KERNEL_OUT}" > "${TMP_LOWERED}"
  "${TRANSLATE_BIN}" "${TMP_LOWERED}" -o "${TMP_PY}"
  echo "----- emitted Python -----"
  cat "${TMP_PY}"
  echo "--------------------------"

  for needle in \
    "@nki.jit" \
    "def matmul_kernel_nki(A, B):" \
    "TILE_M = 64" \
    "TILE_N = 64" \
    "TILE_K = 32" \
    "C = nl.ndarray((M, N), dtype=A.dtype, buffer=nl.shared_hbm)" \
    "for m in nl.affine_range(M // TILE_M):" \
    "for n in nl.affine_range(N // TILE_N):" \
    "for k in nl.affine_range(K // TILE_K):" \
    "res_psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)" \
    "nisa.dma_copy(" \
    "nisa.nc_matmul(" \
    "nisa.tensor_copy(dst=res_sbuf, src=res_psum, dtype=A.dtype)" \
    "return C" \
  ; do
    if grep -qF "${needle}" "${TMP_PY}"; then
      log "PASS: emitted Python contains: ${needle}"
    else
      fail "emitted Python missing: ${needle}"
    fi
  done
fi

log "all checks passed"
