#ifndef GDSL_DIALECT
#define GDSL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Gdsl_Dialect : Dialect {
    let name = "gdsl";
    let summary = "Gdsl out-of-tree MLIR dialect.";
    let description = [{
        TODO
    }];
    let cppNamespace = "::mlir::gdsl";
    let hasConstantMaterializer = 1;
}

#endif
