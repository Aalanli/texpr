add_mlir_dialect_library(MLIRHello
		HelloDialect.cpp
		HelloOps.cpp
		LowerToAffine.cpp
		LowerToLLVM.cpp
		GdslDialect.cpp
		GdslOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/gdsl

        DEPENDS
        MLIRHelloOpsIncGen
		GdslTableGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
