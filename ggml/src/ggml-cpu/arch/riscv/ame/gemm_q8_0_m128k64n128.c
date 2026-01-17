#include "ame.h"
#include <stddef.h>
#include <stdio.h>
// INT8 GEMM using RISC-V AME instructions with fixed tile size M=128, K=64, N=128
// C(128×128) = A(128×64) × B(64×128), where B is transposed in memory
// This function computes a single 128x128 tile output
void ggml_ame_gemm_m128k64n128_i8_i32_bT(
    const int8_t * A,      // Input matrix A: 128x64
    const int8_t * B,      // Input matrix B (transposed): 128x64  
    int32_t * C            // Output matrix C: 128x128
) {
    // Fixed tile dimensions
    const int TILE_M = AME_TILE_M;
    const int TILE_K = AME_TILE_K;
    const int TILE_N = AME_TILE_N;
    AME_LOG("ame_gemm m128k64n128_i8_i32_bT called");
    // Configure matrix dimensions
    int tmp;
    MSETTILEM(tmp, TILE_M);
    MSETTILEK(tmp, TILE_K);
    MSETTILEN(tmp, TILE_N);

    // Preload C matrix to initialize accumulator 
    // MLCE32 will zero accumulator if C is zero, or add to it if non-zero
    int32_t *addr_c = C;
    int stride_c = TILE_N; // Row stride (in elements)
    MLCE32(acc0, addr_c, stride_c * 4);

    // Load left matrix A tile: 128x64
    const int8_t *addr_a = A;
    MLAE8(tr0, addr_a, TILE_K);

    // Load right matrix B tile (transposed): 128x64
    const int8_t *addr_b = B;
    MLBE8(tr1, addr_b, TILE_K);

    // INT8 matrix multiply-accumulate: C(128x128) = A(128x64) × B^T(128x64)
    MQMA(acc0, tr0, tr1);

    // Store INT32 result to C (128x128)
    MSCE32(acc0, addr_c, stride_c * 4);
}
