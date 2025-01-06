#include "gdsl/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "gdsl/Ops.h"


#define GET_OP_CLASSES
#include "gdsl/Ops.cpp.inc"

namespace mlir {


namespace gdsl {

void ComputeOp::build(
    ::mlir::OpBuilder &odsBuilder, 
    ::mlir::OperationState &odsState, 
    int64_t size
) {

}

void ComputeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange sizes) {
    
}
}
}