#include "ame.h"
#include "common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

#if defined(__riscv_v)
#include <riscv_vector.h>
#endif

// Dot product using RVV for Q8_0 blocks
// Compatible with block_q8_0 and block_q4_0_ame (same layout)
#if defined(__riscv_v)
static void ame_vec_dot_q8_0_rvv(int n, float * s, const void * vx, const void * vy) {
    const int qk = 32;
    const int nb = n / qk;
    const block_q8_0 * restrict x = (const block_q8_0 *)vx;
    const block_q8_0 * restrict y = (const block_q8_0 *)vy;
    
    float sumf = 0;
    size_t vl = qk;
    
    for (int i = 0; i < nb; ++i) {
        // load elements
        vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[i].qs, vl);
        vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[i].qs, vl);

        vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);

        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);

        sumf += sumi * (GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));
    }
    *s = sumf;
}
#endif

// Quantize a row of F32 values to Q8_0 format
// This creates one Q8_0 block (32 int8 values + 1 FP16 scale)
static void quantize_row_f32_to_q8_0(const float * x, block_q8_0 * y, int k) {
    assert(k % 32 == 0);
    const int nb = k / 32;  // Number of blocks

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        // Find max absolute value in this block
        for (int j = 0; j < 32; j++) {
            const float v = x[i * 32 + j];
            const float av = fabsf(v);
            if (av > amax) {
                amax = av;
            }
        }

        // Compute scale factor (use 127 for symmetric quantization)
        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        // Quantize values to int8 [-127, 127]
        for (int j = 0; j < 32; j++) {
            const float x0 = x[i * 32 + j] * id;
            const int8_t q = roundf(x0);
            y[i].qs[j] = q;
        }
    }
}

// Repack Q4_0 blocks to AME-optimized format (pre-unpack to int8)
// This is called once during set_tensor, avoiding repeated unpacking
void ggml_ame_repack_q4_0(
    void * dst,
    const void * src,
    int64_t nblocks
) {
    const block_q4_0 * restrict src_blocks = (const block_q4_0 *)src;
    block_q4_0_ame * restrict dst_blocks = (block_q4_0_ame *)dst;
    
    for (int64_t i = 0; i < nblocks; i++) {
        // Copy scale factor
        dst_blocks[i].d = src_blocks[i].d;
        
        // Unpack 4-bit values to int8
        for (int j = 0; j < 16; j++) {
            const uint8_t byte = src_blocks[i].qs[j];
            // Low nibble: bits [0:3]
            const uint8_t v0 = byte & 0x0F;
            // High nibble: bits [4:7]
            const uint8_t v1 = (byte >> 4) & 0x0F;
            
            // Convert unsigned [0,15] to signed [-8,7]
            dst_blocks[i].qs[j * 2 + 0] = (int8_t)(v0) - 8;
            dst_blocks[i].qs[j * 2 + 1] = (int8_t)(v1) - 8;
        }
    }
}

// Pad matrix to AME tile size
static int8_t * ggml_ame_pad_matrix(
    const int8_t * src,
    int rows,
    int cols,
    int padded_rows,
    int padded_cols
) {
    if (!src || rows > padded_rows || cols > padded_cols) {
        return NULL;
    }
    
    size_t total = (size_t)padded_rows * (size_t)padded_cols;
    int8_t * dst = (int8_t *)calloc(total, sizeof(int8_t));
    if (!dst) {
        return NULL;
    }
    
    for (int r = 0; r < rows; r++) {
        memcpy(dst + (size_t)r * padded_cols,
               src + (size_t)r * cols,
               (size_t)cols * sizeof(int8_t));
    }
    
    return dst;
}

// Wrapper for AME-accelerated quantized matrix multiplication
// src0: Q8_0 weight matrix (quantized)
// src1: F32 input matrix (will be quantized on-the-fly)
// dst: F32 output matrix
void ggml_ame_mul_mat_q8_0(
    const void * src0,  // Weight matrix (Q8_0 quantized)
    const void * src1,  // Input matrix (F32)
    void * dst,         // Output (F32)
    int64_t ne00,       // K dimension
    int64_t ne01,       // M dimension (rows)
    int64_t ne10,       // K dimension
    int64_t ne11        // N dimension (columns)
) {
    const int64_t M = ne01;
    const int64_t N = ne11;
    const int64_t K = ne00;
    
    (void)ne10; // Unused parameter
    
    // fprintf(stderr, "[AME] mul_mat_q8_0: M=%d, N=%d, K=%d\n", M, N, K);//改成mul_mat_q8_0
    
    const block_q8_0 * restrict x = (const block_q8_0 *)src0;
    const float * restrict y_f32 = (const float *)src1;  // F32 input
    float * restrict out = (float *)dst;
    
    // Q8_0 block size is 32, but AME_TILE_K is 64, so we need 2 blocks
    const int qk = 32;
    const int64_t nb_x = K / qk;  // number of blocks per row in x
    const int k_blocks_per_tile = AME_TILE_K / qk;  // 64/32 = 2 blocks per tile
    
    // Allocate buffer for quantized input (row-major)
    // We quantize N rows at a time (one row per output column)
    const int64_t y_q8_size = N * nb_x;  // Number of Q8_0 blocks needed
    block_q8_0 * y_q8 = (block_q8_0 *)malloc(y_q8_size * sizeof(block_q8_0));
    if (!y_q8) return;
    
    // Quantize F32 input to Q8_0 (by rows)
    for (int64_t i = 0; i < N; i++) {
        quantize_row_f32_to_q8_0(y_f32 + i * K, y_q8 + i * nb_x, K);
    }
    
    const block_q8_0 * restrict y = y_q8;
    
    // Process tiles with AME
    for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
        const int64_t imax = (i0 + AME_TILE_M < M) ? AME_TILE_M : (M - i0);
        
        for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
            const int64_t jmax = (j0 + AME_TILE_N < N) ? AME_TILE_N : (N - j0);

#if defined(__riscv_v)
             // If the tile is partial (tail), use RVV implementation directly
            if (imax < AME_TILE_M || jmax < AME_TILE_N) {
                for (int64_t i = 0; i < imax; i++) {
                    const block_q8_0 * xi = &x[(i0 + i) * nb_x];
                    for (int64_t j = 0; j < jmax; j++) {
                        const block_q8_0 * yj = &y[(j0 + j) * nb_x];
                        ame_vec_dot_q8_0_rvv(K, &out[(i0 + i) * N + (j0 + j)], xi, yj);
                    }
                }
                continue;
            }
#endif
            
            // Accumulate for this output tile
            float tile_out[AME_TILE_M * AME_TILE_N] = {0};
            
            // Tile over K dimension: process k_blocks_per_tile Q8_0 blocks at a time
            for (int64_t kb = 0; kb < nb_x; kb += k_blocks_per_tile) {
                // Determine how many blocks we can process in this iteration
                const int64_t kb_end = (kb + k_blocks_per_tile < nb_x) ? k_blocks_per_tile : (nb_x - kb);
                const int64_t k_size = kb_end * qk;  // Actual K dimension for this tile
                
                // Extract int8 data for current K-tile from both matrices
                int8_t tile_a[AME_TILE_M * AME_TILE_K];
                int8_t tile_b[AME_TILE_N * AME_TILE_K];
                float scales_a[AME_TILE_M];
                float scales_b[AME_TILE_N];
                
                memset(tile_a, 0, sizeof(tile_a));
                memset(tile_b, 0, sizeof(tile_b));
                
                // Load A tile: accumulate blocks and compute combined scale
                for (int64_t i = 0; i < imax; i++) {
                    float combined_scale = 1.0f;
                    for (int64_t b = 0; b < kb_end; b++) {
                        const block_q8_0 * block = &x[(i0 + i) * nb_x + kb + b];
                        float scale = GGML_FP16_TO_FP32(block->d);
                        if (b == 0) combined_scale = scale;
                        memcpy(&tile_a[i * k_size + b * qk], block->qs, qk);
                    }
                    scales_a[i] = combined_scale;
                }
                
                // Load B tile: accumulate blocks and compute combined scale
                for (int64_t j = 0; j < jmax; j++) {
                    float combined_scale = 1.0f;
                    for (int64_t b = 0; b < kb_end; b++) {
                        const block_q8_0 * block = &y[(j0 + j) * nb_x + kb + b];
                        float scale = GGML_FP16_TO_FP32(block->d);
                        if (b == 0) combined_scale = scale;
                        memcpy(&tile_b[j * k_size + b * qk], block->qs, qk);
                    }
                    scales_b[j] = combined_scale;
                }
                
                // Compute int8 GEMM: C += A * B^T
                int32_t acc[AME_TILE_M * AME_TILE_N] = {0};
                
                // AME GEMM with proper K dimension (64)
                ggml_ame_gemm_q8_0(tile_a, tile_b, acc, imax, k_size, jmax);
                
                // Apply scales and accumulate to float output
                for (int64_t i = 0; i < imax; i++) {
                    for (int64_t j = 0; j < jmax; j++) {
                        tile_out[i * AME_TILE_N + j] += 
                            (float)acc[i * jmax + j] * scales_a[i] * scales_b[j];
                    }
                }
            }
            
            // Write tile to output
            for (int64_t i = 0; i < imax; i++) {
                for (int64_t j = 0; j < jmax; j++) {
                    out[(i0 + i) * N + (j0 + j)] = tile_out[i * AME_TILE_N + j];
                }
            }
        }
    }
    
    // Free temporary quantized buffer
    free(y_q8);
}

// GGML integration wrapper for Q4_0 quantized matrix multiplication
// src0: block_q4_0_ame* (repacked format)
// src1: F32 input matrix (will be quantized on-the-fly)
void ggml_ame_mul_mat_q4_0(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11
) {
    const int64_t M = ne01;
    const int64_t N = ne11;
    const int64_t K = ne00;
    
    (void)ne10; // Unused parameter
    
    fprintf(stderr, "[AME] mul_mat_q4_0: M=%d, N=%d, K=%d\n", M, N, K);
    
    const block_q4_0_ame * restrict x = (const block_q4_0_ame *)src0;
    const float * restrict y_f32 = (const float *)src1;  // F32 input
    float * restrict out = (float *)dst;
    
    // Q4_0 block size is 32
    const int qk = 32;
    const int64_t nb_x = K / qk;  // number of blocks per row in x
    const int k_blocks_per_tile = AME_TILE_K / qk;  // 64/32 = 2 blocks
    
    // Allocate buffer for quantized input
    const int64_t y_q8_size = N * nb_x;
    block_q8_0 * y_q8 = (block_q8_0 *)malloc(y_q8_size * sizeof(block_q8_0));
    if (!y_q8) return;
    
    // Quantize F32 input to Q8_0 (for int8 GEMM compatibility)
    for (int64_t i = 0; i < N; i++) {
        quantize_row_f32_to_q8_0(y_f32 + i * K, y_q8 + i * nb_x, K);
    }
    
    const block_q8_0 * restrict y = y_q8;
    
    // Process tiles with AME
    for (int64_t i0 = 0; i0 < M; i0 += AME_TILE_M) {
        const int64_t imax = (i0 + AME_TILE_M < M) ? AME_TILE_M : (M - i0);
        
        for (int64_t j0 = 0; j0 < N; j0 += AME_TILE_N) {
            const int64_t jmax = (j0 + AME_TILE_N < N) ? AME_TILE_N : (N - j0);

#if defined(__riscv_v)
            if (imax < AME_TILE_M || jmax < AME_TILE_N) {
                for (int64_t i = 0; i < imax; i++) {
                    const block_q4_0_ame * xi = &x[(i0 + i) * nb_x];
                    for (int64_t j = 0; j < jmax; j++) {
                        const block_q8_0 * yj = &y[(j0 + j) * nb_x];
                        // Compatible layout: block_q4_0_ame -> block_q8_0
                        ame_vec_dot_q8_0_rvv(K, &out[(i0 + i) * N + (j0 + j)], (const void*)xi, (const void*)yj);
                    }
                }
                continue;
            }
#endif
            
            // Accumulate for this output tile
            float tile_out[AME_TILE_M * AME_TILE_N] = {0};
            
            // Tile over K dimension in Q4_0 blocks
            for (int64_t kb = 0; kb < nb_x; kb++) {
                // Extract pre-unpacked int8 data for current K-tile
                int8_t tile_a[AME_TILE_M * qk];
                int8_t tile_b[AME_TILE_N * qk];
                float scales_a[AME_TILE_M];
                float scales_b[AME_TILE_N];
                
                // Load A tile (M x qk) - data already unpacked
                for (int64_t i = 0; i < imax; i++) {
                    const block_q4_0_ame * block = &x[(i0 + i) * nb_x + kb];
                    scales_a[i] = GGML_FP16_TO_FP32(block->d);
                    // Direct copy - no unpacking needed!
                    for (int k = 0; k < qk; k++) {
                        tile_a[i * qk + k] = block->qs[k];
                    }
                }
                
                // Load B tile (N x qk) - from Q8_0 quantized input
                for (int64_t j = 0; j < jmax; j++) {
                    const block_q8_0 * block = &y[(j0 + j) * nb_x + kb];
                    scales_b[j] = GGML_FP16_TO_FP32(block->d);
                    // Copy int8 data
                    for (int k = 0; k < qk; k++) {
                        tile_b[j * qk + k] = block->qs[k];
                    }
                }
                
                // Compute int8 GEMM: tile_out += A * B^T (int32 accumulation)
                int32_t acc[AME_TILE_M * AME_TILE_N] = {0};
                
                // AME GEMM expects A: imax x qk, B: jmax x qk, C: imax x jmax
                ggml_ame_gemm_q8_0(tile_a, tile_b, acc, imax, qk, jmax);
                
                // Apply scales and accumulate to float output
                for (int64_t i = 0; i < imax; i++) {
                    for (int64_t j = 0; j < jmax; j++) {
                        tile_out[i * AME_TILE_N + j] += 
                            (float)acc[i * jmax + j] * scales_a[i] * scales_b[j];
                    }
                }
            }
            
            // Write tile to output
            for (int64_t i = 0; i < imax; i++) {
                for (int64_t j = 0; j < jmax; j++) {
                    out[(i0 + i) * N + (j0 + j)] = tile_out[i * AME_TILE_N + j];
                }
            }
        }
    }
    
    // Free quantized buffer
    free(y_q8);
}
