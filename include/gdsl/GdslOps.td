#ifndef GDSL_OPS
#define GDSL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "gdsl/GdslDialect.td"


class Gdsl_Op<string mnemonic, list<Trait> traits = []> :
        Op<Gdsl_Dialect, mnemonic, traits>;


def ConstantOp : Gdsl_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "hello.constant"()
      { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
      : () -> tensor<2x3xf64>
    ```
  }];

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

//  let parser = [{ return ::parseConstantOp(parser, result); }];
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}

def ComputeOp : Gdsl_Op<"compute", [Pure]> {
  let summary = "compute op";
  let description = [{todo}];
  let arguments = (ins Variadic<Index>:$sizes);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  let builders = [
    OpBuilder<(ins "int64_t":$size)>,
    OpBuilder<(ins "ValueRange":$sizes)>,
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;
  }];
}

#endif
