#!/bin/bash
# Build llama.cpp with AME support for RISC-V

function usage() {
    # 运行大模型
    qemu-riscv64 -cpu rv64,v=true,vlen=1024,h=true,zvfh=true,zvfhmin=true,x-matrix=true,rlen=512,mlen=65536,melen=32 ./build/bin/llama-cli -m stories110M-q8_0.gguf -p "One day, Lily met a Shoggoth" -n 500 -c 256
    # DEBUG kernel
    llvm-objdump -d build/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/riscv/ame/gemm_q8_0_m128k64n128.c.o  --mattr=+matrix-xuantie-0.6 
}

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default configuration
DISABLE_OPENMP=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-openmp)
            DISABLE_OPENMP=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--no-openmp]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== Building llama.cpp with RISC-V AME support ===${NC}"
echo -e "${YELLOW}Build Mode: Fully static linking${NC}"
if [ "$DISABLE_OPENMP" = true ]; then
    echo -e "${YELLOW}OpenMP: Disabled${NC}"
else
    echo -e "${YELLOW}OpenMP: Enabled (static)${NC}"
fi

# Configuration
BUILD_DIR="build"
TOOLCHAIN_FILE="cmake/riscv64-toolchain.cmake"

# Check required environment variables
if [ -z "$XS_PROJECT_ROOT" ]; then
    echo -e "${RED}Error: XS_PROJECT_ROOT environment variable not set${NC}"
    exit 1
fi

# Clean previous build (optional)
# Uncomment if you want to clean before building:
# if [ -d "$BUILD_DIR" ]; then
#     echo "Cleaning previous build..."
#     rm -rf "$BUILD_DIR"
# fi

# Configure with CMake
echo -e "${GREEN}Configuring with CMake...${NC}"

# Build CMake options array
CMAKE_OPTS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE"
    -DRISCV_ROOT_PATH="$XS_PROJECT_ROOT/local/llvm"
    -DRISCV_SYSROOT="/opt/riscv/sysroot"
    -DRISCV_TRIPLE="riscv64-unknown-linux-gnu"
    -DRISCV_MARCH="rv64gc_zba_zicbop"
    -DRISCV_MABI="lp64d"
    -DRISCV_USE_LLVM=ON
    -DGGML_CCACHE=OFF
    -DLLAMA_CURL=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_RV_AME=ON
    -DGGML_RV_ZFH=ON
    -DGGML_RV_ZVFH=ON
    -DGGML_RVV=ON
)

# Add OpenMP option if disabled
if [ "$DISABLE_OPENMP" = true ]; then
    CMAKE_OPTS+=(-DGGML_OPENMP=OFF)
fi

cmake -S . -B "$BUILD_DIR" "${CMAKE_OPTS[@]}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Configuration successful!${NC}"
else
    echo -e "${RED}Configuration failed!${NC}"
    exit 1
fi

# Build
echo -e "${GREEN}Building...${NC}"
cmake --build "$BUILD_DIR" --config Release -j$(nproc)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${GREEN}Binaries are in: $BUILD_DIR/bin/${NC}"
    
    # List key binaries
    echo -e "\n${GREEN}Key executables:${NC}"
    ls -lh "$BUILD_DIR/bin/llama-cli" 2>/dev/null || echo "  llama-cli not found"
    ls -lh "$BUILD_DIR/bin/llama-server" 2>/dev/null || echo "  llama-server not found"
    ls -lh "$BUILD_DIR/bin/llama-bench" 2>/dev/null || echo "  llama-bench not found"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}=== Build complete ===${NC}"
echo -e "\n${YELLOW}Build Configuration:${NC}"
echo -e "  • Toolchain: LLVM/Clang (RISC-V)"
echo -e "  • OpenMP: $([ "$DISABLE_OPENMP" = true ] && echo "Disabled" || echo "Enabled (static)")"
echo -e "  • AME Backend: Enabled"
echo -e "\n${YELLOW}AME Integration Details:${NC}"
echo -e "  • AME backend registered as 'RISCV_AME' buffer type"
echo -e "  • Automatic dispatch for Q8_0 and Q4_0 matrix multiplication"
echo -e "  • Q4_0 weights are pre-unpacked during model loading (1.89x memory)"
echo -e "  • Dimensions must be >= 64 for AME acceleration"
echo -e "  • Falls back to RVV/scalar for unsupported operations"
echo -e "\n${YELLOW}To use on RISC-V hardware with AME support:${NC}"
echo -e "  1. Copy the build directory to your RISC-V system"
echo -e "  2. Run inference: ${GREEN}./build/bin/llama-cli -m model.gguf -p 'prompt'${NC}"
echo -e "  3. Benchmark: ${GREEN}./build/bin/llama-bench -m model.gguf${NC}"
echo -e "\n${YELLOW}Performance Notes:${NC}"
echo -e "  • Q4_0 repacking: +89% memory, zero runtime unpacking cost"
echo -e "  • Q8_0 native: Direct hardware acceleration"
echo -e "  • AME activates for matrices with M,N,K >= 64"

