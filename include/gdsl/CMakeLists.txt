add_mlir_dialect(HelloOps hello)
add_mlir_doc(HelloDialect HelloDialect Hello/ -gen-dialect-doc)
add_mlir_doc(HelloOps HelloOps Hello/ -gen-op-doc)

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS GdslDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(GdslDialect GdslDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS GdslOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
# add_mlir_doc(TritonOps TritonOps dialects/ -gen-op-doc)

add_public_tablegen_target(GdslTableGen)
