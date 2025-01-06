#ifndef GDSL_GDSLOPS_H
#define GDSL_GDSLOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"


#define GET_OP_CLASSES
#include "gdsl/Ops.h.inc"

#endif // HELLO_HELLOOPS_H

