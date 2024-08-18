LLVM_REPO=$(pwd)/thirdparty/llvm-project-llvmorg-18.1.6

if [ -d build ]; then
    echo "build directory already exists"
    cd build
else
    mkdir build
    cd build
    cmake -G Ninja .. \
        -DLLVM_DIR=$LLVM_REPO/build/lib/cmake/llvm \
        -DMLIR_DIR=$LLVM_REPO/build/lib/cmake/mlir \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cp compile_commands.json ..
fi

cmake --build . --target gdsl-opt
