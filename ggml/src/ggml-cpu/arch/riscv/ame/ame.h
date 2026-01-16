#ifndef GGML_RISCV_AME_H
#define GGML_RISCV_AME_H

#include <stdint.h>

// AME debug logging (default on for development)
#ifndef AME_DEBUG
#define AME_DEBUG 0
#endif

#if AME_DEBUG
#include <stdio.h>
#define AME_LOG(fmt, ...)                             \
    do {                                              \
        fprintf(stderr, "[AME] " fmt "\n", ##__VA_ARGS__);   \
        fflush(stderr);                              \
    } while (0)
#else
#define AME_LOG(fmt, ...) do { (void)sizeof(fmt); } while (0)
#endif

// Fixed tile dimensions for AME atomic operation
// Matches ggml_ame_gemm_q8_0_m128k64n128: M=128, K=64, N=128
#define AME_TILE_M 128
#define AME_TILE_K 64
#define AME_TILE_N 128

// Helper function to check if AME can be used for given dimensions
// We remove minimum size checks to properly support Q4_0 repacked weights
// which must use AME backend even for small batches (N=1)
// Also require K to be a multiple of 32 (Q8_0/Q4_0 block size) for validity
static inline int ggml_ame_can_use(int M, int N, int K) {
    if (K % 32 != 0) return 0;

    // AME kernels handle padding/tiling for small dimensions,
    // so we accept any size provided K is aligned.
    return 1;
}

// Repacked Q4_0 format for AME (pre-unpacked to int8)
// This avoids unpacking overhead during every matmul
typedef struct {
    uint16_t d;         // scale factor FP16 (same format as block_q4_0)
    int8_t qs[32];      // pre-unpacked 4-bit values to int8 [-8, 7]
} block_q4_0_ame;

// Matrix configuration instructions
#ifdef STC
#define MSETSEW(RD, SEW) \
    asm volatile ( \
        "msetsew %0, %1" \
        : "=r"(RD) \
        : "i"(SEW) \
        : \
    )

#define MSETINT8(RD, VAL) \
    asm volatile ( \
        "msetint8 %0, %1" \
        : "=r"(RD) \
        : "i"(VAL) \
        : \
    )

#define MSETTILEM(RD, VAL) \
    asm volatile ( \
        "msettilem %0, %1" \
        : "=r"(RD) \
        : "r"(VAL) \
        : \
    )

#define MSETTILEK(RD, VAL) \
    asm volatile ( \
        "msettilek %0, %1" \
        : "=r"(RD) \
        : "r"(VAL) \
        : \
    )

#define MSETTILEN(RD, VAL) \
    asm volatile ( \
        "msettilen %0, %1" \
        : "=r"(RD) \
        : "r"(VAL) \
        : \
    )

// Matrix accumulator zero instruction
#define MZERO_ACC(ACC) \
    asm volatile ( \
        "mzero.acc.m " #ACC \
        : \
        : \
        : \
    )

// Matrix load instructions
#define MLAE8(REG, SRC, N) \
    asm volatile ( \
        "mlae8.m " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLBE8(REG, SRC, N) \
    asm volatile ( \
        "mlbe8.m " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLCE32(REG, SRC, N) \
    asm volatile ( \
        "mlce32.m " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

// Matrix store instruction
#define MSCE32(REG, DST, N) \
    asm volatile ( \
        "msce32.m " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

// Matrix multiply-accumulate instruction
#define MMA(ACC, TR0, TR2) \
    asm volatile ( \
        "mmau.mm " #ACC ", " #TR0 ", " #TR2 "\n" \
        : \
        : \
        : \
    )

#define MQMA(ACC, TR0, TR2) \
    asm volatile ( \
        "mqma.mm " #ACC ", " #TR0 ", " #TR2 "\n" \
        : \
        : \
        : \
    )
#else
#define MSETSEW(RD, SEW) ((void)0) //非STC没有这条指令, 空指令

#define MSETINT8(RD, VAL) ((void)0) //非STC没有这条指令, 空指令

#define MSETTILEM(RD, VAL) \
    asm volatile ( \
        "msettilem %0" \
        : \
        : "r"(VAL) \
        : \
    );(RD)=VAL;

#define MSETTILEK(RD, VAL) \
    asm volatile ( \
        "msettilek %0" \
        : \
        : "r"(VAL) \
        : \
    );(RD)=VAL;

#define MSETTILEN(RD, VAL) \
    asm volatile ( \
        "msettilen %0" \
        : \
        : "r"(VAL) \
        : \
    );(RD)=VAL;

// Matrix accumulator zero instruction
#define MZERO_ACC(ACC) ((void)0)

// Matrix load instructions
#define MLAE8(REG, SRC, N) \
    asm volatile ( \
        "mlae8 " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLBE8(REG, SRC, N) \
    asm volatile ( \
        "mlbe8 " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

#define MLCE32(REG, SRC, N) \
    asm volatile ( \
        "mlce32 " #REG ", (%0), %1" \
        : \
        : "r"(SRC), "r"(N) \
        : \
    )

// Matrix store instruction
#define MSCE32(REG, DST, N) \
    asm volatile ( \
        "msce32 " #REG ", (%0), %1" \
        : \
        : "r"(DST), "r"(N) \
        : "memory" \
    )

// Matrix multiply-accumulate instruction
#define MMAU(ACC, TR0, TR2) \
    asm volatile ( \
        "mmaccu.w.b" #ACC ", " #TR0 ", " #TR2 "\n" \
        : \
        : \
        : \
    )

#define MQMA(ACC, TR0, TR2) \
    asm volatile ( \
        "mmacc.w.b " #ACC ", " #TR0 ", " #TR2 "\n" \
        : \
        : \
        : \
    )
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Core AME GEMM function for INT8 matrix multiplication
// C(M×N) += A(M×K) × B(K×N), where B is transposed in memory
void ggml_ame_gemm_q8_0(
    const int8_t * A,
    const int8_t * B,
    int32_t * C,
    int M,
    int K,
    int N
);

// Q4_0 weight repacking (called once during set_tensor)
void ggml_ame_repack_q4_0(
    void * dst,              // Output: block_q4_0_ame array
    const void * src,        // Input: block_q4_0 array
    int64_t nblocks          // Number of Q4_0 blocks
);

// GGML integration wrapper for Q8_0 quantized matrix multiplication
void ggml_ame_mul_mat_q8_0(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11
);

// GGML integration wrapper for Q4_0 quantized matrix multiplication
void ggml_ame_mul_mat_q4_0(
    const void * src0,
    const void * src1,
    void * dst,
    int64_t ne00,
    int64_t ne01,
    int64_t ne10,
    int64_t ne11
);

#ifdef __cplusplus
}
#endif

#endif // GGML_RISCV_AME_H
