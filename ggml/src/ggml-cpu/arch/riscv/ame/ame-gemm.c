#include "ame.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

// Forward declaration of the atomic GEMM function
extern void ggml_ame_gemm_m128k64n128_i8_i32_bT(
    const int8_t * A,      // Input matrix A: 128x64
    const int8_t * B,      // Input matrix B (transposed): 128x64  
    int32_t * C            // Output matrix C: 128x128
);

// INT8 GEMM using RISC-V AME instructions
// C(M×N) = A(M×K) × B(K×N), where B is transposed in memory
// This implementation tiles over the atomic 128x64x128 GEMM
void ggml_ame_gemm_q8_0(
    const int8_t * A,
    const int8_t * B,
    int32_t * C,
    int M,
    int K,
    int N
) {
    // Fixed tile dimensions for the atomic operation
    const int TILE_M = AME_TILE_M;
    const int TILE_K = AME_TILE_K;
    const int TILE_N = AME_TILE_N;
    
    // fprintf(stderr, "[AME] GEMM atomic function: M=%d, N=%d, K=%d (tiles: %dx%dx%d)\n",
    //         M, N, K, TILE_M, TILE_K, TILE_N);
    // fflush(stderr);

    // Loop over M dimension in chunks of TILE_M
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        const int M_tile = (i0 + TILE_M <= M) ? TILE_M : (M - i0);
        
        // Loop over N dimension in chunks of TILE_N
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            const int N_tile = (j0 + TILE_N <= N) ? TILE_N : (N - j0);
            
            // Initialize accumulator for this M×N tile
            int32_t C_tile[TILE_M * TILE_N];
            memset(C_tile, 0, sizeof(C_tile));
            
            // Loop over K dimension in chunks of TILE_K
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                const int K_tile = (k0 + TILE_K <= K) ? TILE_K : (K - k0);
                
                // If all dimensions match exactly, call atomic function directly
                if (M_tile == TILE_M && K_tile == TILE_K && N_tile == TILE_N) {
                    // Extract pointers for this tile
                    const int8_t * A_tile = A + i0 * K + k0;
                    const int8_t * B_tile = B + j0 * K + k0;
                    
                    // fprintf(stderr, "[AME] Direct call: M=%d, N=%d, K=%d (no padding)\n",
                    //         M_tile, N_tile, K_tile);
                    // fflush(stderr);
                    
                    // Call atomic GEMM: C_tile += A_tile × B_tile^T
                    ggml_ame_gemm_m128k64n128_i8_i32_bT(A_tile, B_tile, C_tile);
                } else {
                    // Handle edge cases with padding
                    // fprintf(stderr, "[AME] Padding required: actual(%d,%d,%d) -> padded(%d,%d,%d)\n",
                    //         M_tile, N_tile, K_tile, TILE_M, TILE_N, TILE_K);
                    // fflush(stderr);
                    
                    int8_t A_padded[TILE_M * TILE_K];
                    int8_t B_padded[TILE_N * TILE_K];
                    int32_t C_padded[TILE_M * TILE_N];
                    
                    // Zero padding buffers
                    memset(A_padded, 0, sizeof(A_padded));
                    memset(B_padded, 0, sizeof(B_padded));
                    memset(C_padded, 0, sizeof(C_padded));
                    
                    // Copy A tile with padding (M_tile × K_tile -> TILE_M × TILE_K)
                    for (int i = 0; i < M_tile; i++) {
                        memcpy(A_padded + i * TILE_K,
                               A + (i0 + i) * K + k0,
                               K_tile * sizeof(int8_t));
                    }
                    
                    // Copy B tile with padding (N_tile × K_tile -> TILE_N × TILE_K)
                    for (int j = 0; j < N_tile; j++) {
                        memcpy(B_padded + j * TILE_K,
                               B + (j0 + j) * K + k0,
                               K_tile * sizeof(int8_t));
                    }
                    
                    // Copy current accumulator to padded buffer
                    for (int i = 0; i < M_tile; i++) {
                        memcpy(C_padded + i * TILE_N,
                               C_tile + i * TILE_N,
                               N_tile * sizeof(int32_t));
                    }
                    
                    // Call atomic GEMM with padded data
                    ggml_ame_gemm_m128k64n128_i8_i32_bT(A_padded, B_padded, C_padded);
                    
                    // Copy result back (only valid portion)
                    for (int i = 0; i < M_tile; i++) {
                        memcpy(C_tile + i * TILE_N,
                               C_padded + i * TILE_N,
                               N_tile * sizeof(int32_t));
                    }
                }
            }
            
            // Write accumulated tile to output C matrix
            for (int i = 0; i < M_tile; i++) {
                for (int j = 0; j < N_tile; j++) {
                    C[(i0 + i) * N + (j0 + j)] = C_tile[i * TILE_N + j];
                }
            }
        }
    }
}
