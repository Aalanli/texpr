#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "gdsl/Dialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllPasses();
  registry.insert<
    mlir::gpu::GPUDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
    mlir::math::MathDialect, mlir::nvgpu::NVGPUDialect, mlir::linalg::LinalgDialect,
    mlir::affine::AffineDialect, mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect>();
  registry.insert<mlir::gdsl::GdslDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "GDSL optimizer driver\n",
                                                  registry));
}
