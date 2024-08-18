if [ -d thirdparty ]; then
    echo "thirdparty directory already exists"
else
    mkdir thirdparty
fi

cd thirdparty
# save current path
export THIRD_PARTY_PATH=$(pwd)

echo "Downloading LLVM 18.1.6 to $THIRD_PARTY_PATH"
if [ -f llvmorg-18.1.6.tar.gz ]; then
    echo "File already exists"
else
    wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-18.1.6.tar.gz
fi
if [ -d llvm-project-llvmorg-18.1.6 ]; then
    echo "LLVM Directory already exists"
else
    tar -xvzf llvmorg-18.1.6.tar.gz 
fi

cd llvm-project-llvmorg-18.1.6

LLVM_REPO=$(pwd)
BUILD_DIR=$LLVM_REPO/build
INSTALL_DIR=$LLVM_REPO/install

cmake "-H$LLVM_REPO/llvm" \
     "-B$BUILD_DIR" \
     -DLLVM_INSTALL_UTILS=ON \
     -DLLVM_ENABLE_PROJECTS="mlir;clang" \
     -DLLVM_INCLUDE_TOOLS=ON \
     -DLLVM_BUILD_EXAMPLES=ON \
     -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLVM_ENABLE_ASSERTIONS=ON \
     -DLLVM_ENABLE_RTTI=ON \
     -DCMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR \
     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
     

cmake --build $BUILD_DIR --target check-mlir -j 10

cd $BUILD_DIR
make lli # lli needs to be build separately for testing
cd ..