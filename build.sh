LLVM_REPO=$(pwd)/thirdparty/llvm-project-llvmorg-18.1.6

if ![ -d build ]; then
    mkdir build
fi

cmake -G Ninja -B build \
    -DLLVM_DIR=$LLVM_REPO/build/lib/cmake/llvm \
    -DMLIR_DIR=$LLVM_REPO/build/lib/cmake/mlir \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cp build/compile_commands.json .

cmake --build build --target gdsl-opt
