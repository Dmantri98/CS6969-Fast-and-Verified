//===----------------------------------------------------------------------===//
// linalg-to-nki-opt: an mlir-opt-style driver that registers all upstream
// MLIR dialects and passes plus the local `nki` dialect and conversion
// passes. Useful for running the conversion pass against a `.linalg` file
// from the command line.
//===----------------------------------------------------------------------===//
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "IR/NKIDialect.h"
#include "Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::nki::NKIDialect>();

  mlir::registerAllPasses();
  mlir::nki::registerLinalgToNKIPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "linalg-to-nki optimizer driver\n", registry));
}
